"""PyTorch reference implementation of the RaDe-GS geometry rasterization.

RaDe-GS (arXiv:2406.01467) derives a closed-form per-Gaussian linearization of
the ray--Gaussian intersection: for a pixel at offset ``d = mean2d - pix`` the
intersection *distance* along the pixel ray is ``t = rp.x * d.x + rp.y * d.y +
rp.z`` where ``rp`` (the "ray plane") and the camera-space intersection-plane
normal are per-Gaussian quantities computed from the camera-space mean and the
inverse camera-space covariance (``computeCov2D`` in the reference
``render_forward.cu``).

This module provides:

- :func:`compute_ray_planes`: the per-Gaussian closed form (differentiable
  torch; this is also used at render time by ``rasterization(...,
  render_geometry=True)`` -- its backward comes from autograd, not CUDA).
- :func:`_rasterize_geometry_torch`: a slow, fully differentiable reference
  rasterizer producing the per-pixel geometry outputs (expected plane depth,
  median plane depth, blended normal, GOF-style distortion) with the exact
  compositing semantics of the CUDA kernel. Used as the test oracle.

It also provides the *eval3d* (GOF, arXiv:2404.10772) counterparts, where the
compositing weight is the peak **3D** Gaussian response along the pixel ray
instead of the 2D splat, and depth/normal come from the exact per-ray optimum
``t* = -B/(2A)`` of the camera-space quadric ``f(t r) = A t^2 + B t + C``
(``f`` = squared Mahalanobis distance; GOF ``renderCUDA`` in ``forward.cu``):

- :func:`compute_view2gaussians`: the 10-float per-(camera, Gaussian) quadric
  coefficients (``computeView2Gaussian``); matches the fork's
  ``view_to_gaussians`` CUDA op.
- :func:`_rasterize_geometry_eval3d_torch`: reference rasterizer for the
  eval3d geometry kernels.

Conventions are gsplat's, not INRIA's: pixel centers at integer + 0.5,
``means2d`` in gsplat's pixel space, alpha clamped at 0.999, skip below
1/255, terminate at T <= 1e-4.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from gsplat.cuda._torch_impl import _quat_to_rotmat

# GOF hardcodes these for the distortion NDC mapping (auxiliary.h).
GOF_DISTORTION_NEAR = 0.2
GOF_DISTORTION_FAR = 100.0


def compute_ray_planes(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4] wxyz, need not be normalized
    scales: Tensor,  # [N, 3]
    viewmats: Tensor,  # [C, 4, 4] world-to-camera
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    min_scale: float = 1e-7,
    frustum_clamp: float = 1.3,
    near_plane: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    """Per-Gaussian ray planes and intersection-plane normals (RaDe-GS Eq. 5-12).

    Port of the geometry half of the reference ``computeCov2D``
    (``render_forward.cu``), re-derived against row-major conventions.

    Unlike the reference (which preprocesses only Gaussians that survived the
    near-plane cull), this runs over the full [C, N] grid, so Gaussians at or
    behind the camera plane (``tz <= max(near_plane, 1e-6)``) are masked to
    safe values and emit the degenerate defaults (``rp = 0``, ``normal =
    (0, 0, -1)``). The masking must keep every forward intermediate finite:
    computing through ``tz == 0`` puts Inf/NaN into the graph, and autograd
    turns that into NaN parameter gradients even when the downstream gradient
    is zero (``0 * inf``) — i.e. even for Gaussians culled from
    rasterization. Callers should pass the same ``near_plane`` used for
    projection culling so every composited Gaussian has a real ray plane.

    Returns:
        ray_planes: [C, N, 3]. Per-pixel intersection *distance* (not z-depth)
            is ``t = rp.x * (mean2d.x - px) + rp.y * (mean2d.y - py) + rp.z``
            with the offset in gsplat pixel units.
        normals: [C, N, 3]. Unit camera-space normals of the intersection
            plane at the Gaussian center ray (pointing towards the camera,
            i.e. z-component negative for front-facing).
    """
    C = viewmats.shape[0]
    N = means.shape[0]

    R_cw = viewmats[:, :3, :3]  # [C, 3, 3]
    t_cw = viewmats[:, :3, 3]  # [C, 3]
    means_c = torch.einsum("cij,nj->cni", R_cw, means) + t_cw[:, None, :]  # [C, N, 3]

    # Inverse camera-space covariance. Sigma_cam = (R_cw R_g) S^2 (R_cw R_g)^T
    # so Sigma_cam^{-1} = A A^T with A = R_cam_gauss S^{-1}. When the smallest
    # scale is degenerate the reference falls back to the rank-1 outer product
    # of the thinnest axis (unit magnitude, S dropped): the Gaussian is treated
    # as a plane orthogonal to that axis.
    R_g = _quat_to_rotmat(F.normalize(quats, dim=-1))  # [N, 3, 3]
    R_cg = torch.einsum("cij,njk->cnik", R_cw, R_g)  # [C, N, 3, 3]

    well_conditioned = scales.amin(dim=-1) > min_scale  # [N]
    safe_scales = scales.clamp_min(min_scale)
    A = R_cg / safe_scales[None, :, None, :]  # [C, N, 3, 3] columns scaled
    cov_inv_full = A @ A.transpose(-1, -2)  # [C, N, 3, 3]

    min_axis_id = scales.argmin(dim=-1)  # [N]
    min_axis = torch.gather(
        R_cg, 3, min_axis_id[None, :, None, None].expand(C, N, 3, 1)
    ).squeeze(-1)  # [C, N, 3] camera-space thinnest axis
    cov_inv_rank1 = min_axis[..., :, None] * min_axis[..., None, :]  # [C, N, 3, 3]

    cov_inv = torch.where(
        well_conditioned[None, :, None, None], cov_inv_full, cov_inv_rank1
    )

    # Reference clamps the view-space point used for the linearization to
    # 1.3x the frustum (same clamp EWA uses for the projection Jacobian).
    fx = Ks[:, 0, 0][:, None]  # [C, 1]
    fy = Ks[:, 1, 1][:, None]
    tan_fovx = 0.5 * width / fx  # [C, 1]
    tan_fovy = 0.5 * height / fy
    # Near-plane mask (see docstring): culled Gaussians compute with a safe
    # stand-in view-space point so no intermediate is Inf/NaN, and their
    # outputs are replaced with the degenerate defaults below.
    valid = means_c[..., 2] > max(near_plane, 1e-6)  # [C, N]
    safe_point = means_c.new_tensor([0.0, 0.0, 1.0])
    means_c = torch.where(valid[..., None], means_c, safe_point)
    tz = means_c[..., 2]
    tc = torch.linalg.norm(means_c, dim=-1)  # unclamped distance -> rp.z
    u = torch.clamp(
        means_c[..., 0] / tz,
        min=-frustum_clamp * tan_fovx,
        max=frustum_clamp * tan_fovx,
    )
    v = torch.clamp(
        means_c[..., 1] / tz,
        min=-frustum_clamp * tan_fovy,
        max=frustum_clamp * tan_fovy,
    )
    tx = u * tz  # clamped view-space point
    ty = v * tz
    l = torch.sqrt(tx * tx + ty * ty + tz * tz)  # clamped distance

    uvh = torch.stack([u, v, torch.ones_like(u)], dim=-1)  # [C, N, 3]
    uvh_m = torch.einsum("cnij,cnj->cni", cov_inv, uvh)  # [C, N, 3]
    uvh_m_norm = torch.linalg.norm(uvh_m, dim=-1, keepdim=True)
    # Degenerate: Sigma^{-1} r == 0 (reference emits rp=0, normal=(0,0,-1)),
    # plus the near-plane-culled lanes masked above.
    degenerate = (uvh_m_norm.squeeze(-1) < 1e-15) | ~valid
    uvh_mn = uvh_m / uvh_m_norm.clamp_min(1e-15)

    vbn = (uvh_mn * uvh).sum(dim=-1).clamp_min(1e-7)  # [C, N]
    w = uvh_mn / vbn[..., None]  # [C, N, 3]

    # plane = nJ_inv @ w with nJ_inv rows
    #   ( v^2+1, -uv, -u )
    #   ( -uv, u^2+1, -v )
    plane0 = (v * v + 1.0) * w[..., 0] - u * v * w[..., 1] - u * w[..., 2]
    plane1 = -u * v * w[..., 0] + (u * u + 1.0) * w[..., 1] - v * w[..., 2]

    ray_len2 = u * u + v * v + 1.0
    factor = l / ray_len2  # [C, N]

    rp = torch.stack(
        [plane0 * factor / fx, plane1 * factor / fy, tc], dim=-1
    )  # [C, N, 3]

    # normal = normalize(nJ @ (-plane0*factor, -plane1*factor, -1)); the glm
    # matrix in the reference is column-major, so nJ acts as
    #   n0 = v0/tz + v2*tx/l
    #   n1 = v1/tz + v2*ty/l
    #   n2 = -v0*tx/tz^2 - v1*ty/tz^2 + v2*tz/l
    rnv0 = -plane0 * factor
    rnv1 = -plane1 * factor
    n0 = rnv0 / tz - tx / l
    n1 = rnv1 / tz - ty / l
    n2 = -rnv0 * tx / (tz * tz) - rnv1 * ty / (tz * tz) - tz / l
    normal = torch.stack([n0, n1, n2], dim=-1)
    normal = F.normalize(normal, dim=-1)

    default_normal = torch.tensor([0.0, 0.0, -1.0], device=means.device)
    rp = torch.where(degenerate[..., None], torch.zeros_like(rp), rp)
    normal = torch.where(
        degenerate[..., None], default_normal.expand_as(normal), normal
    )
    return rp, normal


def ray_depth_to_z_map(
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
) -> Tensor:
    """Per-pixel factor converting ray *distance* to z-depth: 1/|(pixnf, 1)|.

    ``pixnf = ((px - cx)/fx, (py - cy)/fy)`` at gsplat pixel centers
    (integer + 0.5). Returns [C, H, W].
    """
    device = Ks.device
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=Ks.dtype) + 0.5,
        torch.arange(width, device=device, dtype=Ks.dtype) + 0.5,
        indexing="ij",
    )
    px = (xs[None] - Ks[:, 0, 2][:, None, None]) / Ks[:, 0, 0][:, None, None]
    py = (ys[None] - Ks[:, 1, 2][:, None, None]) / Ks[:, 1, 1][:, None, None]
    return torch.rsqrt(px * px + py * py + 1.0)  # [C, H, W]


def _rasterize_geometry_torch(
    means2d: Tensor,  # [C, N, 2]
    conics: Tensor,  # [C, N, 3]
    colors: Tensor,  # [C, N, channels]
    opacities: Tensor,  # [C, N]
    ray_planes: Tensor,  # [C, N, 3]
    normals: Tensor,  # [C, N, 3]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    backgrounds: Optional[Tensor] = None,  # [C, channels]
    sort_keys: Optional[Tensor] = None,  # [C, N] compositing order key
    distort_near: float = GOF_DISTORTION_NEAR,
    distort_far: float = GOF_DISTORTION_FAR,
):
    """Reference rasterizer with RaDe-GS geometry outputs.

    Composites all N Gaussians per camera in ``sort_keys`` order (default:
    ``ray_planes[..., 2]``, i.e. center distance -- callers comparing against
    the CUDA kernel must pass the projection z-depths used for the isect
    sort). Small scenes only ([N, P] dense tensors).

    Returns a dict:
        render:   [C, H, W, channels]
        alpha:    [C, H, W, 1]
        expected_depth: [C, H, W, 1] raw accumulated ``sum(t_z * w)``
            (z-depth, alpha-weighted, NOT normalized by alpha)
        median_depth:   [C, H, W, 1] z-depth of the last composited Gaussian
            with pre-composite transmittance > 0.5 (no alpha weighting; the
            selection mask carries no gradient)
        normal:   [C, H, W, 3] accumulated camera-space normal (unnormalized)
        distort:  [C, H, W, 1] raw accumulated GOF distortion (before the
            ``/(1-T)^2`` normalization, which is done by the caller)
    """
    device, dtype = means2d.device, means2d.dtype
    C, N, channels = colors.shape

    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    pix = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)  # [P, 2]
    P = pix.shape[0]
    rln = ray_depth_to_z_map(Ks, width, height).reshape(C, P)  # [C, P]

    # Per (gaussian, pixel) alpha and plane depth.
    d = means2d[:, :, None, :] - pix[None, None, :, :]  # [C, N, P, 2]
    sigma = (
        0.5
        * (
            conics[..., 0, None] * d[..., 0] ** 2
            + conics[..., 2, None] * d[..., 1] ** 2
        )
        + conics[..., 1, None] * d[..., 0] * d[..., 1]
    )  # [C, N, P]
    alpha = (opacities[..., None] * torch.exp(-sigma)).clamp_max(0.999)

    t_ray = (
        ray_planes[..., 0, None] * d[..., 0]
        + ray_planes[..., 1, None] * d[..., 1]
        + ray_planes[..., 2, None]
    )  # [C, N, P] intersection distance
    t_z = t_ray * rln[:, None, :]  # z-depth

    # GOF skips Gaussians whose plane depth is at or below the near plane
    # (forward.cu:518): no contribution to any channel, T untouched.
    skip = (sigma < 0) | (alpha < 1.0 / 255.0) | (t_z <= distort_near)

    if sort_keys is None:
        sort_keys = ray_planes[..., 2]
    order = torch.argsort(sort_keys.detach(), dim=-1)  # [C, N]

    trans = torch.ones(C, P, device=device, dtype=dtype)
    active = torch.ones(C, P, dtype=torch.bool, device=device)
    out_color = torch.zeros(C, P, channels, device=device, dtype=dtype)
    out_alpha = torch.zeros(C, P, device=device, dtype=dtype)
    out_edepth = torch.zeros(C, P, device=device, dtype=dtype)
    out_mdepth = torch.zeros(C, P, device=device, dtype=dtype)
    out_normal = torch.zeros(C, P, 3, device=device, dtype=dtype)
    out_distort = torch.zeros(C, P, device=device, dtype=dtype)
    dist1 = torch.zeros(C, P, device=device, dtype=dtype)
    dist2 = torch.zeros(C, P, device=device, dtype=dtype)
    cam_idx = torch.arange(C, device=device)

    for k in range(N):
        g = order[:, k]  # [C]
        a = alpha[cam_idx, g]  # [C, P]
        s = skip[cam_idx, g]
        contrib = active & ~s
        next_trans = trans * (1.0 - a)
        term = contrib & (next_trans <= 1e-4)
        comp = contrib & ~term
        w = torch.where(comp, a * trans, torch.zeros_like(a))  # [C, P]

        out_color = out_color + colors[cam_idx, g][:, None, :] * w[..., None]
        out_normal = out_normal + normals[cam_idx, g][:, None, :] * w[..., None]
        tz_g = t_z[cam_idx, g]  # [C, P]
        out_edepth = out_edepth + tz_g * w

        med_update = comp & (trans > 0.5)
        out_mdepth = torch.where(med_update, tz_g, out_mdepth)

        # GOF distortion (2DGS NDC mapping, forward.cu:543-557); the running
        # sums only advance where the Gaussian composites (tz > near there;
        # the clamp only guards the masked-out lanes).
        m = (
            distort_far
            * (tz_g - distort_near)
            / ((distort_far - distort_near) * tz_g.clamp_min(1e-8))
        )
        A_acc = 1.0 - trans
        error = m * m * A_acc + dist2 - 2.0 * m * dist1
        out_distort = out_distort + torch.where(comp, error * w, torch.zeros_like(w))
        dist1 = dist1 + torch.where(comp, m * w, torch.zeros_like(w))
        dist2 = dist2 + torch.where(comp, m * m * w, torch.zeros_like(w))

        out_alpha = out_alpha + w
        trans = torch.where(comp, next_trans, trans)
        active = active & ~term

    if backgrounds is not None:
        out_color = out_color + trans[..., None] * backgrounds[:, None, :]

    return {
        "render": out_color.reshape(C, height, width, channels),
        "alpha": out_alpha.reshape(C, height, width, 1),
        "expected_depth": out_edepth.reshape(C, height, width, 1),
        "median_depth": out_mdepth.reshape(C, height, width, 1),
        "normal": out_normal.reshape(C, height, width, 3),
        "distort": out_distort.reshape(C, height, width, 1),
    }


def compute_view2gaussians(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4] wxyz, need not be normalized
    scales: Tensor,  # [N, 3]
    camtoworlds: Tensor,  # [C, 4, 4] camera-to-world
    eps: float = 1e-10,
) -> Tensor:
    """Per-(camera, Gaussian) quadric coefficients of the 3D response.

    Port of GOF's ``computeView2Gaussian`` (``forward.cu``) / the fork's
    ``view_to_gaussians`` CUDA op (which this matches bit-for-bit up to
    float order, including the ``eps=1e-10`` scale regularizer). For a
    camera-space ray point ``p(t) = t * r`` with ``r = (x, y, 1)``, the
    squared Mahalanobis distance to the Gaussian is the quadric

        ``f(t) = AA * t^2 + BB * t + CC``

    with ``AA = r^T M r``, ``BB = 2 b^T r``, ``CC = c`` packed as

        ``[M00, M01, M02, M11, M12, M22, b0, b1, b2, c]``

    where ``M = R_vg^T S^-2 R_vg`` (camera-space inverse covariance),
    ``b = R_vg^T S^-2 t_vg``, ``c = t_vg^T S^-2 t_vg``, ``R_vg/t_vg`` the
    view-to-gaussian transform (``t_vg`` = camera origin in the Gaussian
    frame). Returns [C, N, 10]. Fully differentiable.
    """
    C = camtoworlds.shape[0]
    N = means.shape[0]

    R_g = _quat_to_rotmat(F.normalize(quats, dim=-1))  # [N, 3, 3] gaussian->world
    R_wc = camtoworlds[:, :3, :3]  # [C, 3, 3] camera->world
    t_wc = camtoworlds[:, :3, 3]  # [C, 3]

    # view -> gaussian: R_vg = R_g^T R_wc, t_vg = R_g^T (cam_origin - mean)
    R_vg = torch.einsum("nji,cjk->cnik", R_g, R_wc)  # [C, N, 3, 3]
    t_vg = torch.einsum(
        "nji,cnj->cni", R_g, t_wc[:, None, :] - means[None, :, :]
    )  # [C, N, 3]

    sinv2 = 1.0 / (scales * scales + eps)  # [N, 3]
    sinv2_cn = sinv2[None, :, :].expand(C, N, 3)

    M = torch.einsum("cnji,cnj,cnjk->cnik", R_vg, sinv2_cn, R_vg)  # [C, N, 3, 3]
    b = torch.einsum("cnji,cnj,cnj->cni", R_vg, sinv2_cn, t_vg)  # [C, N, 3]
    c = (t_vg * t_vg * sinv2_cn).sum(dim=-1)  # [C, N]

    return torch.cat(
        [
            M[..., 0, 0, None],
            M[..., 0, 1, None],
            M[..., 0, 2, None],
            M[..., 1, 1, None],
            M[..., 1, 2, None],
            M[..., 2, 2, None],
            b,
            c[..., None],
        ],
        dim=-1,
    )  # [C, N, 10]


def _rasterize_geometry_eval3d_torch(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4] wxyz, need not be normalized
    scales: Tensor,  # [N, 3]
    colors: Tensor,  # [C, N, channels]
    opacities: Tensor,  # [C, N]
    viewmats: Tensor,  # [C, 4, 4] world-to-camera
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    backgrounds: Optional[Tensor] = None,  # [C, channels]
    sort_keys: Optional[Tensor] = None,  # [C, N] compositing order key
    distort_near: float = GOF_DISTORTION_NEAR,
    distort_far: float = GOF_DISTORTION_FAR,
):
    """Reference rasterizer for the eval3d (GOF) geometry outputs.

    Same compositing loop and output dict as
    :func:`_rasterize_geometry_torch`, but per GOF's ``renderCUDA``
    (``forward.cu:409``) the per-(gaussian, pixel) quantities come from the
    exact 3D response along the camera-space ray ``p(t) = t * (x, y, 1)``
    (see :func:`compute_view2gaussians`):

    - weight: ``alpha = opacity * exp(-0.5 * min_t f(t))`` -- the *peak* 3D
      response along the ray, no 2D projection involved;
    - depth: ``t* = -BB / (2 AA)``, which is a z-depth directly (the ray is
      parametrized with unit z), so no ray-to-z conversion is applied;
    - normal: ``-normalize(M r)`` per pixel (the reference evaluates the
      inverse-covariance product at the ray direction, not at the optimum);
    - skip: ``t* <= distort_near`` (GOF's ``NEAR_PLANE`` skip), before the
      alpha threshold, T untouched;
    - distortion: identical NDC-mapped accumulator.

    gsplat conventions where GOF differs: pixel rays use the true principal
    point ``(cx, cy)`` (GOF hardcodes ``W/2, H/2``), alpha is clamped at
    0.999 (GOF: 0.99) and the termination threshold is ``T <= 1e-4`` (GOF:
    ``< 1e-4``), matching the fork's ``rasterize_to_pixels_eval3d`` kernel.

    Default compositing order is the camera-space center z-depth (what the
    projection stage sorts by); callers comparing against CUDA must pass the
    exact isect-sort keys.

    Every intermediate is finite for any Gaussian pose (including behind the
    camera: the quadric is globally well-defined and ``AA >= lambda_min > 0``),
    so unlike :func:`compute_ray_planes` no validity masking is needed for
    autograd safety.
    """
    device, dtype = means.device, means.dtype
    C, N, channels = colors.shape

    # Algebraic inverse of the world-to-camera transform (exactly what the
    # CUDA side does; avoids torch.linalg.inv numerics).
    R_cw = viewmats[:, :3, :3]  # [C, 3, 3]
    t_cw = viewmats[:, :3, 3]  # [C, 3]
    camtoworlds = torch.zeros_like(viewmats)
    camtoworlds[:, :3, :3] = R_cw.transpose(-1, -2)
    camtoworlds[:, :3, 3] = -torch.einsum("cji,cj->ci", R_cw, t_cw)
    camtoworlds[:, 3, 3] = 1.0

    v2g = compute_view2gaussians(means, quats, scales, camtoworlds)  # [C, N, 10]

    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    px = (xs.reshape(-1)[None, :] - Ks[:, 0, 2][:, None]) / Ks[:, 0, 0][:, None]
    py = (ys.reshape(-1)[None, :] - Ks[:, 1, 2][:, None]) / Ks[:, 1, 1][:, None]
    r = torch.stack([px, py, torch.ones_like(px)], dim=-1)  # [C, P, 3]
    P = r.shape[1]

    Msym = torch.stack(
        [
            torch.stack([v2g[..., 0], v2g[..., 1], v2g[..., 2]], dim=-1),
            torch.stack([v2g[..., 1], v2g[..., 3], v2g[..., 4]], dim=-1),
            torch.stack([v2g[..., 2], v2g[..., 4], v2g[..., 5]], dim=-1),
        ],
        dim=-2,
    )  # [C, N, 3, 3]
    b = v2g[..., 6:9]  # [C, N, 3]
    c = v2g[..., 9]  # [C, N]

    Mr = torch.einsum("cnij,cpj->cnpi", Msym, r)  # [C, N, P, 3]
    AA = torch.einsum("cnpi,cpi->cnp", Mr, r)  # [C, N, P] > 0 (M is PD)
    BB = 2.0 * torch.einsum("cni,cpi->cnp", b, r)  # [C, N, P]

    t_z = -BB / (2.0 * AA)  # exact optimum, z-depth
    # min_t f(t) = CC - BB^2 / (4 AA); GOF clamps power at 0 (numerics only).
    min_value = (c[..., None] - BB * BB / (4.0 * AA)).clamp_min(0.0)
    alpha = (opacities[..., None] * torch.exp(-0.5 * min_value)).clamp_max(0.999)

    # GOF's normalization epsilon convention (backward.cu:813:
    # len = sqrt(|u|^2 + 1e-7)), matched by the CUDA kernel.
    normal = -Mr / torch.sqrt(
        (Mr * Mr).sum(dim=-1, keepdim=True) + 1e-7
    )  # [C, N, P, 3]

    # GOF skips at or below the near plane before the alpha test
    # (forward.cu:518); order does not matter as both only gate contribution.
    skip = (t_z <= distort_near) | (alpha < 1.0 / 255.0)

    if sort_keys is None:
        means_c = (
            torch.einsum("cij,nj->cni", R_cw, means) + t_cw[:, None, :]
        )  # [C, N, 3]
        sort_keys = means_c[..., 2]
    order = torch.argsort(sort_keys.detach(), dim=-1)  # [C, N]

    trans = torch.ones(C, P, device=device, dtype=dtype)
    active = torch.ones(C, P, dtype=torch.bool, device=device)
    out_color = torch.zeros(C, P, channels, device=device, dtype=dtype)
    out_alpha = torch.zeros(C, P, device=device, dtype=dtype)
    out_edepth = torch.zeros(C, P, device=device, dtype=dtype)
    out_mdepth = torch.zeros(C, P, device=device, dtype=dtype)
    out_normal = torch.zeros(C, P, 3, device=device, dtype=dtype)
    out_distort = torch.zeros(C, P, device=device, dtype=dtype)
    dist1 = torch.zeros(C, P, device=device, dtype=dtype)
    dist2 = torch.zeros(C, P, device=device, dtype=dtype)
    cam_idx = torch.arange(C, device=device)

    for k in range(N):
        g = order[:, k]  # [C]
        a = alpha[cam_idx, g]  # [C, P]
        s = skip[cam_idx, g]
        contrib = active & ~s
        next_trans = trans * (1.0 - a)
        term = contrib & (next_trans <= 1e-4)
        comp = contrib & ~term
        w = torch.where(comp, a * trans, torch.zeros_like(a))  # [C, P]

        out_color = out_color + colors[cam_idx, g][:, None, :] * w[..., None]
        out_normal = out_normal + normal[cam_idx, g] * w[..., None]
        tz_g = t_z[cam_idx, g]  # [C, P]
        out_edepth = out_edepth + tz_g * w

        med_update = comp & (trans > 0.5)
        out_mdepth = torch.where(med_update, tz_g, out_mdepth)

        # GOF distortion (2DGS NDC mapping, forward.cu:543-557); the running
        # sums only advance where the Gaussian composites (t_z > near there;
        # the clamp only guards the masked-out lanes).
        m = (
            distort_far
            * (tz_g - distort_near)
            / ((distort_far - distort_near) * tz_g.clamp_min(1e-8))
        )
        A_acc = 1.0 - trans
        error = m * m * A_acc + dist2 - 2.0 * m * dist1
        out_distort = out_distort + torch.where(comp, error * w, torch.zeros_like(w))
        dist1 = dist1 + torch.where(comp, m * w, torch.zeros_like(w))
        dist2 = dist2 + torch.where(comp, m * m * w, torch.zeros_like(w))

        out_alpha = out_alpha + w
        trans = torch.where(comp, next_trans, trans)
        active = active & ~term

    if backgrounds is not None:
        out_color = out_color + trans[..., None] * backgrounds[:, None, :]

    return {
        "render": out_color.reshape(C, height, width, channels),
        "alpha": out_alpha.reshape(C, height, width, 1),
        "expected_depth": out_edepth.reshape(C, height, width, 1),
        "median_depth": out_mdepth.reshape(C, height, width, 1),
        "normal": out_normal.reshape(C, height, width, 3),
        "distort": out_distort.reshape(C, height, width, 1),
    }
