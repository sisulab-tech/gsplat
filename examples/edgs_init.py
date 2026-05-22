"""EDGS (Eliminating Densification for Efficient Convergence of 3DGS) initialization.

Ports the correspondence-based dense initialization from the EDGS paper to work
with gsplat's data structures (Parser/Dataset from datasets/colmap.py).

Reference: third_party/EDGS/source/corr_init.py
"""

import math
import time
from collections import defaultdict
from typing import List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.cluster.vq import kmeans, vq
from scipy.spatial.distance import cdist
from tqdm import tqdm

from datasets.colmap import Dataset, Parser


# ---------------------------------------------------------------------------
# Helper: camera selection and nearest neighbors
# ---------------------------------------------------------------------------


def k_closest_vectors(matrix: torch.Tensor, k: int) -> torch.Tensor:
    """Find the k-closest vectors for each vector based on Euclidean distance.

    Args:
        matrix: Input matrix of shape [N, D].
        k: Number of closest vectors to return.

    Returns:
        Indices of the k-closest vectors for each vector, shape [N, k].
    """
    distances = torch.cdist(matrix, matrix, p=2)
    distances.fill_diagonal_(float("inf"))
    _, indices = torch.topk(distances, k, largest=False, dim=1)
    return indices


def select_cameras_kmeans(cameras: np.ndarray, K: int) -> List[int]:
    """Select K cameras from a set using K-means clustering.

    Args:
        cameras: Array of shape (N, 16), flattened 4x4 camera matrices.
        K: Number of clusters (cameras to select).

    Returns:
        List of indices of the cameras closest to the cluster centers.
    """
    if not isinstance(cameras, np.ndarray):
        cameras = np.asarray(cameras)
    if cameras.shape[1] != 16:
        raise ValueError(
            "Each camera must have 16 values corresponding to a flattened 4x4 matrix."
        )

    cluster_centers, _ = kmeans(cameras, K)
    cluster_assignments, _ = vq(cameras, cluster_centers)

    selected_indices = []
    for k_idx in range(K):
        cluster_members = cameras[cluster_assignments == k_idx]
        if len(cluster_members) == 0:
            continue
        distances = cdist([cluster_centers[k_idx]], cluster_members)[0]
        nearest_camera_idx = np.where(cluster_assignments == k_idx)[0][
            np.argmin(distances)
        ]
        selected_indices.append(nearest_camera_idx)

    return selected_indices


# ---------------------------------------------------------------------------
# Helper: projection matrix construction
# ---------------------------------------------------------------------------


def _get_projection_matrix(
    znear: float, zfar: float, fovX: float, fovY: float
) -> torch.Tensor:
    """Build OpenGL-style projection matrix (column-vector convention).

    Exact port of getProjectionMatrix from
    third_party/EDGS/submodules/gaussian-splatting/utils/graphics_utils.py
    """
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)
    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def build_full_proj_transform(
    w2c: torch.Tensor,
    K: torch.Tensor,
    W: int,
    H: int,
    znear: float = 0.01,
    zfar: float = 100.0,
) -> torch.Tensor:
    """Build 4x4 row-vector projection matrix matching original 3DGS convention.

    Replicates: world_view_transform @ projection_matrix
    where world_view_transform = W2C.T and projection_matrix = P.T

    Args:
        w2c: [4, 4] world-to-camera transform (column-vector convention).
        K: [3, 3] camera intrinsics matrix.
        W: Image width.
        H: Image height.
        znear: Near clipping plane.
        zfar: Far clipping plane.

    Returns:
        full_proj: [4, 4] matrix for row-vector convention: [X,Y,Z,1] @ P
    """
    fx = K[0, 0].item() if isinstance(K, torch.Tensor) else float(K[0, 0])
    fy = K[1, 1].item() if isinstance(K, torch.Tensor) else float(K[1, 1])

    fovX = 2.0 * math.atan(W / (2.0 * fx))
    fovY = 2.0 * math.atan(H / (2.0 * fy))

    P = _get_projection_matrix(znear, zfar, fovX, fovY)

    w2c_t = w2c.clone().float()
    # Row-vector convention: world_view_transform = W2C^T, projection = P^T
    world_view_transform = w2c_t.T
    projection_matrix = P.T

    full_proj = world_view_transform @ projection_matrix
    return full_proj


# ---------------------------------------------------------------------------
# Helper: tensor preparation
# ---------------------------------------------------------------------------


def prepare_tensor(
    input_array, device: str = "cuda"
) -> torch.Tensor:
    """Convert input to a float32 tensor on the specified device."""
    if not isinstance(input_array, torch.Tensor):
        return (
            torch.tensor(input_array, dtype=torch.float32).to(device).clone().detach()
        )
    return input_array.clone().detach().to(device).to(torch.float32)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def load_image_for_roma(
    parser: Parser, image_idx: int
) -> Tuple[Image.Image, np.ndarray]:
    """Load and undistort an image for RoMa matching.

    Args:
        parser: COLMAP parser with image paths and distortion maps.
        image_idx: Global image index (into parser arrays).

    Returns:
        pil_image: PIL Image for RoMa input.
        np_image: numpy uint8 array [H, W, 3] for color extraction.
    """
    image = imageio.imread(parser.image_paths[image_idx])[..., :3]
    camera_id = parser.camera_ids[image_idx]
    params = parser.params_dict[camera_id]

    if len(params) > 0 and camera_id in parser.mapx_dict:
        mapx = parser.mapx_dict[camera_id]
        mapy = parser.mapy_dict[camera_id]
        image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
        x, y, w, h = parser.roi_undist_dict[camera_id]
        image = image[y : y + h, x : x + w]

    pil_image = Image.fromarray(image)
    return pil_image, image


# ---------------------------------------------------------------------------
# RoMa matching: multi-NN variant helpers
# ---------------------------------------------------------------------------


def compute_warp_and_confidence(
    img_pil_A: Image.Image,
    img_pil_B: Image.Image,
    roma_model,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Compute warp and confidence between two images using RoMa.

    Args:
        img_pil_A: Source PIL image.
        img_pil_B: Target PIL image.
        roma_model: Pre-trained RoMa model.
        device: Computation device.

    Returns:
        certainty: [H, W] confidence tensor.
        warp: [H, W, 4] warp tensor (source coords + target coords).
        imB_np: Target image as numpy array.
    """
    from romatch.utils import get_tuple_transform_ops

    ws, hs = roma_model.w_resized, roma_model.h_resized
    test_transform = get_tuple_transform_ops(resize=(hs, ws), normalize=True)
    im_A, im_B = test_transform((img_pil_A, img_pil_B))
    batch = {"im_A": im_A[None].to(device), "im_B": im_B[None].to(device)}

    corresps = (
        roma_model.forward(batch)
        if not roma_model.symmetric
        else roma_model.forward_symmetric(batch)
    )
    finest_scale = 1
    hs_out, ws_out = (
        roma_model.upsample_res if roma_model.upsample_preds else (hs, ws)
    )

    certainty = corresps[finest_scale]["certainty"]
    im_A_to_im_B = corresps[finest_scale]["flow"]
    if roma_model.attenuate_cert:
        low_res_certainty = F.interpolate(
            corresps[16]["certainty"],
            size=(hs_out, ws_out),
            align_corners=False,
            mode="bilinear",
        )
        certainty -= 0.5 * low_res_certainty * (low_res_certainty < 0)

    if roma_model.upsample_preds:
        im_A_to_im_B = F.interpolate(
            im_A_to_im_B,
            size=(hs_out, ws_out),
            align_corners=False,
            mode="bilinear",
        )
        certainty = F.interpolate(
            certainty,
            size=(hs_out, ws_out),
            align_corners=False,
            mode="bilinear",
        )

    im_A_to_im_B = im_A_to_im_B.permute(0, 2, 3, 1)
    im_A_coords = (
        torch.stack(
            torch.meshgrid(
                torch.linspace(
                    -1 + 1 / hs_out, 1 - 1 / hs_out, hs_out, device=device
                ),
                torch.linspace(
                    -1 + 1 / ws_out, 1 - 1 / ws_out, ws_out, device=device
                ),
                indexing="ij",
            ),
            dim=0,
        )
        .permute(1, 2, 0)
        .unsqueeze(0)
        .expand(im_A_to_im_B.size(0), -1, -1, -1)
    )

    warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
    certainty = certainty.sigmoid()

    return certainty[0, 0], warp[0], np.array(img_pil_B)


def resize_batch(
    tensors_3d: torch.Tensor,
    tensors_4d: torch.Tensor,
    target_shape: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resize a batch of certainty [B,H,W] and warp [B,H,W,4] tensors."""
    target_H, target_W = target_shape

    resized_3d = F.interpolate(
        tensors_3d.unsqueeze(1),
        size=(target_H, target_W),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    resized_4d = F.interpolate(
        tensors_4d.permute(0, 3, 1, 2),
        size=(target_H, target_W),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)

    return resized_3d, resized_4d


def aggregate_confidences_and_warps(
    parser: Parser,
    train_indices: np.ndarray,
    closest_indices: np.ndarray,
    roma_model,
    source_local_idx: int,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, List[np.ndarray]]:
    """Aggregate confidences and warps across nearest neighbors.

    Args:
        parser: COLMAP parser.
        train_indices: Global indices of training images.
        closest_indices: [N_train, K] local indices of nearest neighbors.
        roma_model: Pre-trained RoMa model.
        source_local_idx: Local index of the source viewpoint in train_indices.
        device: Computation device.

    Returns:
        certainties_max: [H, W] max confidence across neighbors.
        warps_max: [H, W, 4] warp for best neighbor per pixel.
        certainties_max_idcs: [H, W] which neighbor is best per pixel.
        imA_np: Source image numpy array.
        imB_compound: List of neighbor images.
    """
    certainties_all, warps_all, imB_compound = [], [], []
    source_global_idx = train_indices[source_local_idx]

    img_pil_A, imA_np = load_image_for_roma(parser, source_global_idx)

    for nn_local_idx in closest_indices[source_local_idx]:
        nn_global_idx = train_indices[nn_local_idx]
        img_pil_B, _ = load_image_for_roma(parser, nn_global_idx)

        certainty, warp, imB = compute_warp_and_confidence(
            img_pil_A, img_pil_B, roma_model, device=device
        )
        certainties_all.append(certainty)
        warps_all.append(warp)
        imB_compound.append(imB)

    certainties_all = torch.stack(certainties_all, dim=0)
    target_shape = imB_compound[0].shape[:2]

    certainties_all_resized, warps_all_resized = resize_batch(
        certainties_all, torch.stack(warps_all, dim=0), target_shape
    )

    certainties_max, certainties_max_idcs = torch.max(
        certainties_all_resized, dim=0
    )
    H, W = certainties_max.shape

    warps_max = warps_all_resized[
        certainties_max_idcs, torch.arange(H).unsqueeze(1), torch.arange(W)
    ]

    return (
        certainties_max,
        warps_max,
        certainties_max_idcs,
        imA_np,
        imB_compound,
        certainties_all_resized,
        warps_all_resized,
    )


# ---------------------------------------------------------------------------
# Keypoint extraction
# ---------------------------------------------------------------------------


def extract_keypoints_and_colors(
    imA: np.ndarray,
    imB_compound: List[np.ndarray],
    certainties_max: torch.Tensor,
    certainties_max_idcs: torch.Tensor,
    matches: torch.Tensor,
    roma_model,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract keypoints and colors from source and multiple target images (multi-NN).

    Args:
        imA: Source image [H_A, W_A, 3] uint8.
        imB_compound: List of target images [H_B, W_B, 3] uint8.
        certainties_max: [H, W] max confidence.
        certainties_max_idcs: [H, W] best neighbor index per pixel.
        matches: [N, 4] matches in normalized coordinates.
        roma_model: RoMa model instance.

    Returns:
        kptsA_np: [N, 2] normalized keypoints in imA (x, y in NDC).
        kptsB_np: [N, 2] pixel keypoints in imB.
        kptsB_proj_matrices_idx: [N] which neighbor image for each keypoint.
        kptsA_color: [N, 3] colors in imA.
        kptsB_color: [N, 3] colors in imB.
    """
    H_A, W_A, _ = imA.shape
    H, W = certainties_max.shape

    # Convert matches to pixel coordinates
    kptsA, kptsB = roma_model.to_pixel_coordinates(matches, W_A, H_A, H, W)

    kptsA_np = kptsA.detach().cpu().numpy()
    kptsB_np = kptsB.detach().cpu().numpy()
    kptsA_np = kptsA_np[:, [1, 0]]

    # Re-get pixel coords for color extraction
    kptsA_np = kptsA.detach().cpu().numpy()
    kptsB_np = kptsB.detach().cpu().numpy()

    # Extract colors for keypoints in imA
    kptsA_x = np.round(kptsA_np[:, 0] / 1.0).astype(int)
    kptsA_y = np.round(kptsA_np[:, 1] / 1.0).astype(int)
    kptsA_color = imA[np.clip(kptsA_x, 0, H - 1), np.clip(kptsA_y, 0, W - 1)]

    # Extract colors for keypoints in imB using certainties_max_idcs
    imB_compound_np = np.stack(imB_compound, axis=0)
    H_B, W_B, _ = imB_compound[0].shape

    kptsB_x = np.round(kptsB_np[:, 0]).astype(int)
    kptsB_y = np.round(kptsB_np[:, 1]).astype(int)

    certainties_max_idcs_np = certainties_max_idcs.detach().cpu().numpy()
    kptsB_proj_matrices_idx = certainties_max_idcs_np[
        np.clip(kptsA_x, 0, H - 1), np.clip(kptsA_y, 0, W - 1)
    ]
    kptsB_color = imB_compound_np[
        kptsB_proj_matrices_idx,
        np.clip(kptsB_y, 0, H - 1),
        np.clip(kptsB_x, 0, W - 1),
    ]

    # Normalize keypoints to [-1, 1]
    kptsA_np[:, 0] = kptsA_np[:, 0] / H * 2.0 - 1.0
    kptsA_np[:, 1] = kptsA_np[:, 1] / W * 2.0 - 1.0
    kptsB_np[:, 0] = kptsB_np[:, 0] / W_B * 2.0 - 1.0
    kptsB_np[:, 1] = kptsB_np[:, 1] / H_B * 2.0 - 1.0

    return (
        kptsA_np[:, [1, 0]],
        kptsB_np,
        kptsB_proj_matrices_idx,
        kptsA_color,
        kptsB_color,
    )


def extract_keypoints_and_colors_single(
    imA: np.ndarray,
    imB: np.ndarray,
    matches: torch.Tensor,
    roma_model,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract keypoints and colors from a source and single target image (fast variant).

    Args:
        imA: Source image [H_A, W_A, 3] uint8.
        imB: Target image [H_B, W_B, 3] uint8.
        matches: [N, 4] matches in normalized coordinates (x1, y1, x2, y2).
        roma_model: RoMa model instance.

    Returns:
        kptsA_np_norm: [N, 2] normalized keypoints in imA.
        kptsB_np_norm: [N, 2] normalized keypoints in imB.
        kptsA_color: [N, 3] colors in imA (uint8).
        kptsB_color: [N, 3] colors in imB (uint8).
    """
    H_A, W_A, _ = imA.shape
    H_B, W_B, _ = imB.shape

    kptsA = matches[:, :2]  # [N, 2]
    kptsB = matches[:, 2:]  # [N, 2]

    # Scale normalized coordinates [-1, 1] to pixel coordinates
    kptsA_pix = torch.zeros_like(kptsA)
    kptsB_pix = torch.zeros_like(kptsB)

    kptsA_pix[:, 0] = (kptsA[:, 0] + 1) * (W_A - 1) / 2
    kptsA_pix[:, 1] = (kptsA[:, 1] + 1) * (H_A - 1) / 2

    kptsB_pix[:, 0] = (kptsB[:, 0] + 1) * (W_B - 1) / 2
    kptsB_pix[:, 1] = (kptsB[:, 1] + 1) * (H_B - 1) / 2

    kptsA_np = kptsA_pix.detach().cpu().numpy()
    kptsB_np = kptsB_pix.detach().cpu().numpy()

    # Extract colors
    kptsA_x = np.round(kptsA_np[:, 0]).astype(int)
    kptsA_y = np.round(kptsA_np[:, 1]).astype(int)
    kptsB_x = np.round(kptsB_np[:, 0]).astype(int)
    kptsB_y = np.round(kptsB_np[:, 1]).astype(int)

    kptsA_color = imA[
        np.clip(kptsA_y, 0, H_A - 1), np.clip(kptsA_x, 0, W_A - 1)
    ]
    kptsB_color = imB[
        np.clip(kptsB_y, 0, H_B - 1), np.clip(kptsB_x, 0, W_B - 1)
    ]

    # Normalize keypoints to [-1, 1] for triangulation
    kptsA_np_norm = np.zeros_like(kptsA_np)
    kptsB_np_norm = np.zeros_like(kptsB_np)

    kptsA_np_norm[:, 0] = kptsA_np[:, 0] / (W_A - 1) * 2.0 - 1.0
    kptsA_np_norm[:, 1] = kptsA_np[:, 1] / (H_A - 1) * 2.0 - 1.0

    kptsB_np_norm[:, 0] = kptsB_np[:, 0] / (W_B - 1) * 2.0 - 1.0
    kptsB_np_norm[:, 1] = kptsB_np[:, 1] / (H_B - 1) * 2.0 - 1.0

    return kptsA_np_norm, kptsB_np_norm, kptsA_color, kptsB_color


# ---------------------------------------------------------------------------
# Triangulation
# ---------------------------------------------------------------------------


def triangulate_points(
    P1: torch.Tensor,
    P2: torch.Tensor,
    k1_x: torch.Tensor,
    k1_y: torch.Tensor,
    k2_x: torch.Tensor,
    k2_y: torch.Tensor,
    device: str = "cuda",
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Triangulate 3D points from two-view correspondences via linear least squares.

    Args:
        P1, P2: Projection matrices [batch, 4, 4] or [4, 4].
        k1_x, k1_y: NDC coordinates in camera 1, shape [batch].
        k2_x, k2_y: NDC coordinates in camera 2, shape [batch].
        device: Computation device.

    Returns:
        X: Homogeneous 3D points [batch, 4].
        errors_proj1: Reprojection errors in camera 1 [batch].
        errors_proj2: Reprojection errors in camera 2 [batch].
    """
    EPS = 1e-4

    P1 = prepare_tensor(P1, device)
    P2 = prepare_tensor(P2, device)
    k1_x = prepare_tensor(k1_x, device)
    k1_y = prepare_tensor(k1_y, device)
    k2_x = prepare_tensor(k2_x, device)
    k2_y = prepare_tensor(k2_y, device)
    batch_size = k1_x.shape[0]

    if P1.ndim == 2:
        P1 = P1.unsqueeze(0).expand(batch_size, -1, -1)
    if P2.ndim == 2:
        P2 = P2.unsqueeze(0).expand(batch_size, -1, -1)

    # Extract columns from P1 and P2
    P1_0, P1_1, P1_2 = P1[:, :, 0], P1[:, :, 1], P1[:, :, 2]
    P2_0, P2_1, P2_2 = P2[:, :, 0], P2[:, :, 1], P2[:, :, 2]

    k1_x = k1_x.view(-1, 1)
    k1_y = k1_y.view(-1, 1)
    k2_x = k2_x.view(-1, 1)
    k2_y = k2_y.view(-1, 1)

    # Construct equations: A * [x, y, z]^T = -b
    A1 = P1_0 - k1_x * P1_2
    A2 = P1_1 - k1_y * P1_2
    A3 = P2_0 - k2_x * P2_2
    A4 = P2_1 - k2_y * P2_2

    A = torch.stack([A1, A2, A3, A4], dim=1)  # [batch, 4, 4]

    b = -A[:, :, 3]  # [batch, 4]
    A_reduced = A[:, :, :3]  # [batch, 4, 3]

    X_xyz = torch.linalg.lstsq(A_reduced, b.unsqueeze(2)).solution.squeeze(
        2
    )  # [batch, 3]

    ones = torch.ones((batch_size, 1), dtype=torch.float32, device=X_xyz.device)
    X = torch.cat([X_xyz, ones], dim=1)  # [batch, 4]

    # Compute reprojection errors
    seeked_splats_proj1 = (X.unsqueeze(1) @ P1).squeeze(1)
    seeked_splats_proj1 = seeked_splats_proj1 / (
        EPS + seeked_splats_proj1[:, [3]]
    )
    seeked_splats_proj2 = (X.unsqueeze(1) @ P2).squeeze(1)
    seeked_splats_proj2 = seeked_splats_proj2 / (
        EPS + seeked_splats_proj2[:, [3]]
    )
    proj1_target = torch.cat([k1_x, k1_y], dim=1)
    proj2_target = torch.cat([k2_x, k2_y], dim=1)
    errors_proj1 = (
        torch.abs(seeked_splats_proj1[:, :2] - proj1_target)
        .sum(1)
        .detach()
        .cpu()
        .numpy()
    )
    errors_proj2 = (
        torch.abs(seeked_splats_proj2[:, :2] - proj2_target)
        .sum(1)
        .detach()
        .cpu()
        .numpy()
    )

    return X, errors_proj1, errors_proj2


def select_best_keypoints(
    NNs_triangulated_points: torch.Tensor,
    NNs_errors_proj1: np.ndarray,
    NNs_errors_proj2: np.ndarray,
    device: str = "cuda",
) -> Tuple[torch.Tensor, np.ndarray]:
    """Select the best triangulated points across nearest neighbors.

    For each point, selects the triangulation with the lowest max reprojection error.

    Args:
        NNs_triangulated_points: [num_nns, num_points, dim] triangulated points.
        NNs_errors_proj1: [num_nns, num_points] reprojection errors cam 1.
        NNs_errors_proj2: [num_nns, num_points] reprojection errors cam 2.

    Returns:
        selected_points: [num_points, dim] best triangulated points.
        selected_errors: [num_points] min max-reprojection-error.
    """
    NNs_errors_proj = np.maximum(NNs_errors_proj1, NNs_errors_proj2)
    indices = torch.from_numpy(np.argmin(NNs_errors_proj, axis=0)).long().to(device)
    n_indices = torch.arange(NNs_triangulated_points.shape[1]).long().to(device)

    selected = NNs_triangulated_points[indices, n_indices, :]
    return selected, np.min(NNs_errors_proj, axis=0)


# ---------------------------------------------------------------------------
# Main EDGS initialization
# ---------------------------------------------------------------------------


def init_edgs(
    parser: Parser,
    trainset: Dataset,
    device: str = "cuda",
    num_refs: int = 180,
    nns_per_ref: int = 1,
    matches_per_ref: int = 15_000,
    scaling_factor: float = 0.001,
    proj_err_tolerance: float = 0.01,
    roma_model_type: str = "outdoors",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """EDGS dense initialization from multi-view correspondences.

    Args:
        parser: COLMAP parser with camera data.
        trainset: Training dataset (used for training indices).
        device: Computation device.
        num_refs: Number of reference views selected by K-means.
        nns_per_ref: Number of nearest neighbors per reference (1=fast variant).
        matches_per_ref: Number of keypoint matches sampled per reference view.
        scaling_factor: Scale factor for Gaussian size relative to camera distance.
        proj_err_tolerance: Reprojection error threshold for opacity masking.
        roma_model_type: RoMa model variant ("outdoors" or "indoors").

    Returns:
        points: [N, 3] triangulated 3D positions.
        rgbs: [N, 3] colors in [0, 1].
        scales: [N, 3] log-space scales.
        opacities: [N] logit-space opacities.
        quats: [N, 4] random quaternions.
    """
    from romatch import roma_indoor, roma_outdoor

    timings = defaultdict(list)

    # Load RoMa model
    print(f"[EDGS] Loading RoMa model ({roma_model_type})...")
    if roma_model_type == "indoors":
        roma_model = roma_indoor(device=device, use_custom_corr=False)
    else:
        roma_model = roma_outdoor(device=device, use_custom_corr=False)
    roma_model.upsample_preds = False
    roma_model.symmetric = False

    M = matches_per_ref
    upper_thresh = roma_model.sample_thresh
    expansion_factor = 1

    # Get training image indices
    train_indices = trainset.indices  # global indices into parser arrays

    # Build flattened W2C matrices for all training cameras
    w2c_flat = []
    for idx in train_indices:
        w2c = parser.worldtocams[idx]
        w2c_flat.append(w2c.flatten())
    w2c_flat = torch.from_numpy(np.stack(w2c_flat, axis=0)).float()

    # Select reference cameras via K-means
    NUM_REFERENCE_FRAMES = min(num_refs, len(train_indices))
    selected_local_indices = select_cameras_kmeans(
        cameras=w2c_flat.numpy(), K=NUM_REFERENCE_FRAMES
    )
    selected_local_indices = sorted(selected_local_indices)
    print(
        f"[EDGS] Selected {len(selected_local_indices)} reference frames "
        f"from {len(train_indices)} training images."
    )

    # Find nearest neighbors
    NUM_NNS = min(nns_per_ref, len(train_indices) - 1)
    closest_indices = k_closest_vectors(w2c_flat, NUM_NNS)
    closest_indices_np = closest_indices.detach().cpu().numpy()

    # Warmup RoMa
    print("[EDGS] Warming up RoMa model...")
    with torch.no_grad():
        img_pil_0, _ = load_image_for_roma(parser, train_indices[0])
        idx1 = 1 if len(train_indices) > 1 else 0
        img_pil_1, _ = load_image_for_roma(parser, train_indices[idx1])
        warp, certainty_warp = roma_model.match(img_pil_0, img_pil_1, device=device)
        del warp, certainty_warp
        torch.cuda.empty_cache()

    # Accumulators
    all_new_xyz = []
    all_new_colors = []
    all_new_opacities = []
    all_new_scaling = []

    if nns_per_ref == 1:
        _init_edgs_fast(
            parser=parser,
            train_indices=train_indices,
            selected_local_indices=selected_local_indices,
            closest_indices_np=closest_indices_np,
            roma_model=roma_model,
            M=M,
            upper_thresh=upper_thresh,
            expansion_factor=expansion_factor,
            scaling_factor=scaling_factor,
            proj_err_tolerance=proj_err_tolerance,
            device=device,
            all_new_xyz=all_new_xyz,
            all_new_colors=all_new_colors,
            all_new_opacities=all_new_opacities,
            all_new_scaling=all_new_scaling,
            timings=timings,
        )
    else:
        _init_edgs_multi_nn(
            parser=parser,
            train_indices=train_indices,
            selected_local_indices=selected_local_indices,
            closest_indices_np=closest_indices_np,
            roma_model=roma_model,
            M=M,
            upper_thresh=upper_thresh,
            expansion_factor=expansion_factor,
            scaling_factor=scaling_factor,
            proj_err_tolerance=proj_err_tolerance,
            device=device,
            all_new_xyz=all_new_xyz,
            all_new_colors=all_new_colors,
            all_new_opacities=all_new_opacities,
            all_new_scaling=all_new_scaling,
            timings=timings,
        )

    # Concatenate results
    points = torch.cat(all_new_xyz, dim=0)  # [N, 3]
    rgbs = torch.cat(all_new_colors, dim=0)  # [N, 3] float [0,1]
    scales = torch.cat(all_new_scaling, dim=0)  # [N, 3] log-space
    opacities = torch.cat(all_new_opacities, dim=0)  # [N] logit-space

    # Post-init scale halving (from EDGS trainer.py)
    scales = scales + math.log(0.5)

    # Random quaternions
    N = points.shape[0]
    quats = torch.rand((N, 4))

    print(f"[EDGS] Initialized {N} Gaussians.")
    if timings:
        print("\n=== EDGS Profiling Summary (average per frame) ===")
        for key, times in timings.items():
            print(
                f"  {key:35s}: {sum(times) / len(times):.4f} sec "
                f"(total {sum(times):.2f} sec)"
            )

    return points, rgbs, scales, opacities, quats


# ---------------------------------------------------------------------------
# Fast variant (nns_per_ref == 1)
# ---------------------------------------------------------------------------


def _init_edgs_fast(
    parser: Parser,
    train_indices: np.ndarray,
    selected_local_indices: List[int],
    closest_indices_np: np.ndarray,
    roma_model,
    M: int,
    upper_thresh: float,
    expansion_factor: int,
    scaling_factor: float,
    proj_err_tolerance: float,
    device: str,
    all_new_xyz: list,
    all_new_colors: list,
    all_new_opacities: list,
    all_new_scaling: list,
    timings: dict,
):
    """Fast EDGS initialization using single nearest neighbor per reference."""
    for source_local_idx in tqdm(
        selected_local_indices, desc="[EDGS fast] Processing reference frames"
    ):
        source_global_idx = train_indices[source_local_idx]
        source_camera_id = parser.camera_ids[source_global_idx]

        # Step 1: Compute warp and certainty
        start = time.time()
        NNs = closest_indices_np.shape[1]
        nn_local_idx = closest_indices_np[
            source_local_idx, np.random.randint(NNs)
        ]
        nn_global_idx = train_indices[nn_local_idx]

        img_pil_A, imA_np = load_image_for_roma(parser, source_global_idx)
        img_pil_B, imB_np = load_image_for_roma(parser, nn_global_idx)

        with torch.no_grad():
            warp, certainty_warp = roma_model.match(
                img_pil_A, img_pil_B, device=device
            )
        timings["aggregation_warp_certainty"].append(time.time() - start)

        # Step 2: Good samples selection
        start = time.time()
        certainty = certainty_warp.reshape(-1).clone()
        certainty[certainty > upper_thresh] = 1
        good_samples = torch.multinomial(
            certainty,
            num_samples=min(expansion_factor * M, len(certainty)),
            replacement=False,
        )
        timings["good_samples_selection"].append(time.time() - start)

        # Step 3: Extract keypoints and triangulate
        start = time.time()
        matches_NN = warp.reshape(-1, 4)[good_samples]

        kptsA_np, kptsB_np, kptsA_color, kptsB_color = (
            extract_keypoints_and_colors_single(
                imA_np, imB_np, matches_NN, roma_model
            )
        )

        # Build projection matrices
        source_K = parser.Ks_dict[source_camera_id]
        source_W, source_H = parser.imsize_dict[source_camera_id]
        source_w2c = torch.from_numpy(parser.worldtocams[source_global_idx]).float()
        proj_A = build_full_proj_transform(
            source_w2c,
            torch.from_numpy(source_K).float(),
            source_W,
            source_H,
        )

        nn_camera_id = parser.camera_ids[nn_global_idx]
        nn_K = parser.Ks_dict[nn_camera_id]
        nn_W, nn_H = parser.imsize_dict[nn_camera_id]
        nn_w2c = torch.from_numpy(parser.worldtocams[nn_global_idx]).float()
        proj_B = build_full_proj_transform(
            nn_w2c,
            torch.from_numpy(nn_K).float(),
            nn_W,
            nn_H,
        )

        with torch.no_grad():
            triangulated_points, errors_proj1, errors_proj2 = triangulate_points(
                P1=torch.stack([proj_A] * M, dim=0),
                P2=torch.stack([proj_B] * M, dim=0),
                k1_x=kptsA_np[:M, 0],
                k1_y=kptsA_np[:M, 1],
                k2_x=kptsB_np[:M, 0],
                k2_y=kptsB_np[:M, 1],
                device=device,
            )
        timings["triangulation_per_NN"].append(time.time() - start)

        # Step 4: Select best (only 1 NN, so trivial)
        start = time.time()
        selected_points, selected_errors = select_best_keypoints(
            NNs_triangulated_points=triangulated_points.unsqueeze(0),
            NNs_errors_proj1=errors_proj1[np.newaxis, :],
            NNs_errors_proj2=errors_proj2[np.newaxis, :],
            device=device,
        )
        timings["select_best_keypoints"].append(time.time() - start)

        # Step 5: Create Gaussian parameters
        start = time.time()
        new_xyz = selected_points[:, :3]  # Drop homogeneous coord
        all_new_xyz.append(new_xyz)

        # Colors as float [0, 1]
        all_new_colors.append(
            torch.from_numpy(kptsA_color[:M].astype(np.float32) / 255.0)
        )

        # Opacities: 0 for good points, -10 for bad (logit space)
        mask_bad = torch.tensor(
            selected_errors[:M] > proj_err_tolerance,
            dtype=torch.float32,
        )
        all_new_opacities.append(-mask_bad * 10.0)

        # Scales: log(distance_to_camera * scaling_factor), isotropic
        cam_center = torch.from_numpy(
            parser.camtoworlds[source_global_idx][:3, 3]
        ).float()
        dist_to_cam = torch.linalg.norm(
            cam_center.to(new_xyz.device) - new_xyz, dim=1, ord=2
        )
        all_new_scaling.append(
            torch.log(
                (dist_to_cam * scaling_factor).unsqueeze(1).repeat(1, 3)
            )
        )
        timings["save_gaussians"].append(time.time() - start)

        # Free memory
        del warp, certainty_warp
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Multi-NN variant (nns_per_ref > 1)
# ---------------------------------------------------------------------------


def _init_edgs_multi_nn(
    parser: Parser,
    train_indices: np.ndarray,
    selected_local_indices: List[int],
    closest_indices_np: np.ndarray,
    roma_model,
    M: int,
    upper_thresh: float,
    expansion_factor: int,
    scaling_factor: float,
    proj_err_tolerance: float,
    device: str,
    all_new_xyz: list,
    all_new_colors: list,
    all_new_opacities: list,
    all_new_scaling: list,
    timings: dict,
):
    """Multi-NN EDGS initialization using multiple nearest neighbors per reference."""
    for source_local_idx in tqdm(
        selected_local_indices, desc="[EDGS multi-NN] Processing reference frames"
    ):
        source_global_idx = train_indices[source_local_idx]
        source_camera_id = parser.camera_ids[source_global_idx]

        # Step 1: Aggregate warps and confidences across neighbors
        with torch.no_grad():
            (
                certainties_max,
                warps_max,
                certainties_max_idcs,
                imA_np,
                imB_compound,
                certainties_all,
                warps_all,
            ) = aggregate_confidences_and_warps(
                parser=parser,
                train_indices=train_indices,
                closest_indices=closest_indices_np,
                roma_model=roma_model,
                source_local_idx=source_local_idx,
                device=device,
            )

        # Step 2: Sample high-confidence matches
        with torch.no_grad():
            matches = warps_max
            certainty = certainties_max.clone()
            certainty[certainty > upper_thresh] = 1
            matches, certainty = matches.reshape(-1, 4), certainty.reshape(-1)

            good_samples = torch.multinomial(
                certainty,
                num_samples=min(expansion_factor * M, len(certainty)),
                replacement=False,
            )

        # Step 3: Triangulate across all NNs
        reference_image_dict = {
            "triangulated_points": [],
            "errors_proj1": [],
            "errors_proj2": [],
        }

        source_K = parser.Ks_dict[source_camera_id]
        source_W, source_H = parser.imsize_dict[source_camera_id]
        source_w2c = torch.from_numpy(parser.worldtocams[source_global_idx]).float()
        proj_A = build_full_proj_transform(
            source_w2c,
            torch.from_numpy(source_K).float(),
            source_W,
            source_H,
        )

        with torch.no_grad():
            for NN_idx in range(len(warps_all)):
                matches_NN = warps_all[NN_idx].reshape(-1, 4)[good_samples]

                kptsA_np, kptsB_np, kptsB_proj_idx, kptsA_color, kptsB_color = (
                    extract_keypoints_and_colors(
                        imA_np,
                        imB_compound,
                        certainties_max,
                        certainties_max_idcs,
                        matches_NN,
                        roma_model,
                    )
                )

                nn_local_idx = closest_indices_np[source_local_idx, NN_idx]
                nn_global_idx = train_indices[nn_local_idx]
                nn_camera_id = parser.camera_ids[nn_global_idx]
                nn_K = parser.Ks_dict[nn_camera_id]
                nn_W, nn_H = parser.imsize_dict[nn_camera_id]
                nn_w2c = torch.from_numpy(
                    parser.worldtocams[nn_global_idx]
                ).float()
                proj_B = build_full_proj_transform(
                    nn_w2c,
                    torch.from_numpy(nn_K).float(),
                    nn_W,
                    nn_H,
                )

                tri_pts, err1, err2 = triangulate_points(
                    P1=torch.stack([proj_A] * M, dim=0),
                    P2=torch.stack([proj_B] * M, dim=0),
                    k1_x=kptsA_np[:M, 0],
                    k1_y=kptsA_np[:M, 1],
                    k2_x=kptsB_np[:M, 0],
                    k2_y=kptsB_np[:M, 1],
                    device=device,
                )

                reference_image_dict["triangulated_points"].append(tri_pts)
                reference_image_dict["errors_proj1"].append(err1)
                reference_image_dict["errors_proj2"].append(err2)

        # Step 4: Select best triangulated points
        with torch.no_grad():
            selected_points, selected_errors = select_best_keypoints(
                NNs_triangulated_points=torch.stack(
                    reference_image_dict["triangulated_points"], dim=0
                ),
                NNs_errors_proj1=np.stack(
                    reference_image_dict["errors_proj1"], axis=0
                ),
                NNs_errors_proj2=np.stack(
                    reference_image_dict["errors_proj2"], axis=0
                ),
                device=device,
            )

        # Step 5: Create Gaussian parameters
        new_xyz = selected_points[:, :3]
        all_new_xyz.append(new_xyz)

        all_new_colors.append(
            torch.from_numpy(kptsA_color[:M].astype(np.float32) / 255.0)
        )

        mask_bad = torch.tensor(
            selected_errors[:M] > proj_err_tolerance,
            dtype=torch.float32,
        )
        all_new_opacities.append(-mask_bad * 10.0)

        cam_center = torch.from_numpy(
            parser.camtoworlds[source_global_idx][:3, 3]
        ).float()
        dist_to_cam = torch.linalg.norm(
            cam_center.to(new_xyz.device) - new_xyz, dim=1, ord=2
        )
        all_new_scaling.append(
            torch.log(
                (dist_to_cam * scaling_factor).unsqueeze(1).repeat(1, 3)
            )
        )

        torch.cuda.empty_cache()
