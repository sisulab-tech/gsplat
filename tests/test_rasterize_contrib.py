"""Tests for the contribution-tracking forward rasterizer.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

import math
from typing import Optional, Tuple

import pytest
import torch
from torch import Tensor

device = torch.device("cuda:0")

ALPHA_THRESHOLD = 1.0 / 255.0
ALPHA_CAP = 0.99
T_EPS = 1e-4
FLOAT_MAX = torch.finfo(torch.float32).max


def _reference(
    means2d: Tensor,  # [C, N, 2]
    conics: Tensor,  # [C, N, 3]
    colors: Tensor,  # [C, N, channels]
    opacities: Tensor,  # [C, N]
    depths: Tensor,  # [C, N]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
    backgrounds: Optional[Tensor] = None,
    pixel_masks: Optional[Tensor] = None,
    calculate_surface_distance: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Brute-force torch reference, one tile and one Gaussian at a time.

    Returns the five kernel outputs plus a boolean mask of the Gaussians whose
    winning pixel is unambiguous, which the pixel-id comparison uses.
    """
    C, N = opacities.shape
    channels = colors.shape[-1]
    tile_height, tile_width = isect_offsets.shape[1:3]
    block_size = tile_size * tile_size
    n_isects = flatten_ids.numel()

    means2d_f = means2d.reshape(-1, 2)
    conics_f = conics.reshape(-1, 3)
    colors_f = colors.reshape(-1, channels)
    opacities_f = opacities.reshape(-1)
    depths_f = depths.reshape(-1)
    offsets_f = isect_offsets.reshape(-1)

    render = torch.zeros(C, image_height, image_width, channels, device=device)
    edepth = torch.zeros(C, image_height, image_width, device=device)
    min_surface_dist = torch.full((C * N,), FLOAT_MAX, device=device)

    vote_gauss, vote_pixel, vote_contrib = [], [], []

    for c in range(C):
        for ty in range(tile_height):
            for tx in range(tile_width):
                flat_tile = c * tile_height * tile_width + ty * tile_width + tx
                start = int(offsets_f[flat_tile])
                end = (
                    n_isects
                    if flat_tile == C * tile_height * tile_width - 1
                    else int(offsets_f[flat_tile + 1])
                )

                ys = ty * tile_size + torch.arange(tile_size, device=device)
                xs = tx * tile_size + torch.arange(tile_size, device=device)
                pix_y, pix_x = torch.meshgrid(ys, xs, indexing="ij")
                pix_y, pix_x = pix_y.reshape(-1), pix_x.reshape(-1)
                px = pix_x.float() + 0.5
                py = pix_y.float() + 0.5

                inside = (pix_y < image_height) & (pix_x < image_width)
                pix_id = pix_y.clamp(max=image_height - 1) * image_width + pix_x.clamp(
                    max=image_width - 1
                )
                if pixel_masks is None:
                    masked = torch.zeros_like(inside)
                else:
                    masked = inside & ~pixel_masks[c].reshape(-1)[pix_id]
                done = ~inside | masked

                T = torch.ones(block_size, device=device)
                out = torch.zeros(block_size, channels, device=device)
                ed = torch.zeros(block_size, device=device)

                num_batches = math.ceil((end - start) / block_size)
                for b in range(num_batches):
                    if bool(done.all()):
                        break
                    batch_start = start + b * block_size
                    batch_size = min(block_size, end - batch_start)
                    batch_ids = flatten_ids[batch_start : batch_start + batch_size]

                    for t in range(batch_size):
                        g = int(batch_ids[t])
                        conic = conics_f[g]
                        mean = means2d_f[g]
                        dx = mean[0] - px
                        dy = mean[1] - py
                        sigma = (
                            0.5 * (conic[0] * dx * dx + conic[2] * dy * dy)
                            + conic[1] * dx * dy
                        )
                        alpha = (opacities_f[g] * torch.exp(-sigma)).clamp(
                            max=ALPHA_CAP
                        )
                        passes = ~done & (sigma >= 0) & (alpha >= ALPHA_THRESHOLD)
                        next_T = T * (1.0 - alpha)
                        stop = passes & (next_T < T_EPS)
                        active = passes & ~stop

                        vis = alpha * T
                        out[active] += colors_f[g] * vis[active, None]
                        ed[active] += depths_f[g] * vis[active]
                        T = torch.where(active, next_T, T)
                        done = done | stop

                        if bool(active.any()):
                            vote_gauss.append(
                                torch.full(
                                    (int(active.sum()),),
                                    g,
                                    dtype=torch.long,
                                    device=device,
                                )
                            )
                            vote_pixel.append(pix_id[active])
                            vote_contrib.append(vis[active])

                    if calculate_surface_distance:
                        scoring = inside & ~masked
                        for t in range(batch_size):
                            g = int(batch_ids[t])
                            dist = (depths_f[g] - ed).abs()[scoring].min()
                            min_surface_dist[g] = torch.minimum(
                                min_surface_dist[g], dist
                            )

                keep = inside & ~masked
                flat_render = render[c].reshape(-1, channels)
                flat_edepth = edepth[c].reshape(-1)
                if backgrounds is not None:
                    out = out + T[:, None] * backgrounds[c][None, :]
                flat_render[pix_id[keep]] = out[keep]
                flat_edepth[pix_id[keep]] = ed[keep]
                render[c] = flat_render.reshape(image_height, image_width, channels)
                edepth[c] = flat_edepth.reshape(image_height, image_width)

    max_contrib = torch.zeros(C * N, device=device)
    max_pixel = torch.zeros(C * N, dtype=torch.int32, device=device)
    unique = torch.zeros(C * N, dtype=torch.bool, device=device)

    if vote_gauss:
        g_idx = torch.cat(vote_gauss)
        p_idx = torch.cat(vote_pixel)
        contrib = torch.cat(vote_contrib)

        # same packing as the kernel: contribution in the high word, the
        # complemented pixel id in the low word so ties pick the smallest id
        key = (contrib.view(torch.int32).to(torch.int64) << 32) | (
            (~p_idx.to(torch.int32)).to(torch.int64) & 0xFFFFFFFF
        )
        best_key = torch.zeros(C * N, dtype=torch.int64, device=device).scatter_reduce(
            0, g_idx, key, reduce="amax"
        )
        seen = best_key != 0
        max_contrib[seen] = (best_key[seen] >> 32).to(torch.int32).view(torch.float32)
        max_pixel[seen] = ~(best_key[seen] & 0xFFFFFFFF).to(torch.int32)

        # a Gaussian's winning pixel is only comparable when its top
        # contribution is attained once and is clear of the runner-up
        best = torch.zeros(C * N, device=device).scatter_reduce(
            0, g_idx, contrib, reduce="amax"
        )
        is_best = contrib == best[g_idx]
        n_best = torch.zeros(C * N, device=device).scatter_add(
            0, g_idx, is_best.float()
        )
        runner_up = torch.full((C * N,), -1.0, device=device).scatter_reduce(
            0,
            g_idx,
            torch.where(is_best, torch.full_like(contrib, -1.0), contrib),
            reduce="amax",
        )
        unique = seen & (n_best == 1) & ((best - runner_up) > 1e-4 * best)

    return (
        render,
        edepth,
        max_contrib.reshape(C, N),
        max_pixel.reshape(C, N),
        min_surface_dist.reshape(C, N),
        unique.reshape(C, N),
    )


def _scene(n_gauss: int, width: int, height: int, seed: int = 7):
    """Random projected Gaussians spread over the image."""
    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles

    torch.manual_seed(seed)
    C = 1
    means2d = torch.rand(C, n_gauss, 2, device=device)
    means2d[..., 0] *= width
    means2d[..., 1] *= height
    # conics are the inverse of an isotropic-ish 2D covariance
    radii_px = torch.rand(C, n_gauss, device=device) * 6.0 + 2.0
    var = radii_px**2
    conics = torch.zeros(C, n_gauss, 3, device=device)
    conics[..., 0] = 1.0 / var
    conics[..., 2] = 1.0 / var
    colors = torch.rand(C, n_gauss, 3, device=device)
    opacities = torch.rand(C, n_gauss, device=device) * 0.8 + 0.1
    depths = torch.rand(C, n_gauss, device=device) * 5.0 + 1.0

    tile_size = 16
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)
    radii = (3.0 * radii_px).ceil().to(torch.int32)
    radii = torch.stack([radii, radii], dim=-1)
    _, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)
    return dict(
        means2d=means2d,
        conics=conics,
        colors=colors,
        opacities=opacities,
        depths=depths,
        tile_size=tile_size,
        isect_offsets=isect_offsets,
        flatten_ids=flatten_ids,
        width=width,
        height=height,
    )


def _run(scene, **kwargs):
    from gsplat.cuda._wrapper import rasterize_to_pixels_contrib

    return rasterize_to_pixels_contrib(
        scene["means2d"],
        scene["conics"],
        scene["colors"],
        scene["opacities"],
        scene["depths"],
        scene["width"],
        scene["height"],
        scene["tile_size"],
        scene["isect_offsets"],
        scene["flatten_ids"],
        **kwargs,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_matches_reference():
    scene = _scene(n_gauss=200, width=64, height=48)
    backgrounds = torch.ones(1, 3, device=device)

    colors, edepth, max_contrib, max_pixel, min_sd = _run(
        scene, backgrounds=backgrounds, calculate_surface_distance=True
    )
    (
        ref_colors,
        ref_edepth,
        ref_max_contrib,
        ref_max_pixel,
        ref_min_sd,
        unique,
    ) = _reference(
        scene["means2d"],
        scene["conics"],
        scene["colors"],
        scene["opacities"],
        scene["depths"],
        scene["width"],
        scene["height"],
        scene["tile_size"],
        scene["isect_offsets"],
        scene["flatten_ids"],
        backgrounds=backgrounds,
        calculate_surface_distance=True,
    )

    torch.testing.assert_close(colors, ref_colors, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(edepth, ref_edepth, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(max_contrib, ref_max_contrib, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(min_sd, ref_min_sd, atol=1e-4, rtol=1e-4)

    assert unique.any(), "no Gaussian had an unambiguous winning pixel"
    assert torch.equal(max_pixel[unique], ref_max_pixel[unique])

    # every tile here holds fewer than one batch, so the surface distance is
    # taken against the fully accumulated depth
    assert (min_sd < FLOAT_MAX).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_multi_batch_surface_distance():
    # a single 16x16 tile with more than 256 Gaussians, so the surface
    # distances of the later batches see a partially accumulated depth
    scene = _scene(n_gauss=600, width=16, height=16, seed=11)
    assert scene["flatten_ids"].numel() > 256

    _, _, max_contrib, max_pixel, min_sd = _run(scene, calculate_surface_distance=True)
    (
        _,
        _,
        ref_max_contrib,
        ref_max_pixel,
        ref_min_sd,
        unique,
    ) = _reference(
        scene["means2d"],
        scene["conics"],
        scene["colors"],
        scene["opacities"],
        scene["depths"],
        scene["width"],
        scene["height"],
        scene["tile_size"],
        scene["isect_offsets"],
        scene["flatten_ids"],
        calculate_surface_distance=True,
    )

    torch.testing.assert_close(max_contrib, ref_max_contrib, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(min_sd, ref_min_sd, atol=1e-4, rtol=1e-4)
    assert unique.any()
    assert torch.equal(max_pixel[unique], ref_max_pixel[unique])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_masked_pixels_cast_no_votes():
    scene = _scene(n_gauss=200, width=64, height=48)
    masks = torch.ones(1, 48, 64, dtype=torch.bool, device=device)
    masks[:, :, 32:] = False

    colors, edepth, max_contrib, max_pixel, _ = _run(scene, pixel_masks=masks)

    assert torch.all(colors[:, :, 32:] == 0)
    assert torch.all(edepth[:, :, 32:] == 0)

    voted = max_contrib > 0
    winner_x = max_pixel % 64
    assert torch.all(winner_x[voted] < 32)

    (
        ref_colors,
        ref_edepth,
        ref_max_contrib,
        ref_max_pixel,
        _,
        unique,
    ) = _reference(
        scene["means2d"],
        scene["conics"],
        scene["colors"],
        scene["opacities"],
        scene["depths"],
        scene["width"],
        scene["height"],
        scene["tile_size"],
        scene["isect_offsets"],
        scene["flatten_ids"],
        pixel_masks=masks,
    )
    torch.testing.assert_close(colors, ref_colors, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(edepth, ref_edepth, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(max_contrib, ref_max_contrib, atol=1e-5, rtol=1e-4)
    assert torch.equal(max_pixel[unique], ref_max_pixel[unique])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_deterministic():
    scene = _scene(n_gauss=400, width=64, height=48, seed=3)
    first = _run(scene, calculate_surface_distance=True)
    second = _run(scene, calculate_surface_distance=True)
    for a, b in zip(first, second):
        assert torch.equal(a, b)
