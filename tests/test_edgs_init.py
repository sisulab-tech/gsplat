"""Tests for EDGS initialization functions.

Tests cover:
- Camera selection helpers (k_closest_vectors, select_cameras_kmeans)
- Projection matrix construction (build_full_proj_transform)
- Triangulation (triangulate_points)
- Best keypoint selection (select_best_keypoints)
- Keypoint extraction (extract_keypoints_and_colors_single)
- Projection-triangulation roundtrip consistency

Most tests run on CPU since the EDGS helpers are pure PyTorch/NumPy.

Usage:
```bash
pytest tests/test_edgs_init.py -s
```
"""

import math
import sys
import os

import numpy as np
import pytest
import torch

# Add examples/ to path so we can import edgs_init
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from edgs_init import (
    build_full_proj_transform,
    extract_keypoints_and_colors_single,
    k_closest_vectors,
    prepare_tensor,
    select_best_keypoints,
    select_cameras_kmeans,
    triangulate_points,
)


# ---------------------------------------------------------------------------
# Tests for k_closest_vectors
# ---------------------------------------------------------------------------


class TestKClosestVectors:
    def test_basic(self):
        """K-nearest neighbors returns correct indices for simple arrangement."""
        # 4 points in 2D forming a line: 0, 1, 2, 3
        matrix = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
        )
        indices = k_closest_vectors(matrix, k=2)
        assert indices.shape == (4, 2)

        # Point 0's 2 closest should be 1, 2
        assert set(indices[0].tolist()) == {1, 2}
        # Point 3's 2 closest should be 2, 1
        assert set(indices[3].tolist()) == {1, 2}
        # Point 1's 2 closest should be 0, 2
        assert set(indices[1].tolist()) == {0, 2}

    def test_k_equals_one(self):
        """Single nearest neighbor is correct."""
        matrix = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]]
        )
        indices = k_closest_vectors(matrix, k=1)
        assert indices.shape == (3, 1)
        assert indices[0].item() == 1  # closest to 0 is 1
        assert indices[1].item() == 0  # closest to 1 is 0
        assert indices[2].item() == 1  # closest to 10 is 1

    def test_excludes_self(self):
        """Each vector's own index should not appear in its neighbors."""
        N = 10
        matrix = torch.randn(N, 5)
        indices = k_closest_vectors(matrix, k=3)
        for i in range(N):
            assert i not in indices[i].tolist()


# ---------------------------------------------------------------------------
# Tests for select_cameras_kmeans
# ---------------------------------------------------------------------------


class TestSelectCamerasKmeans:
    def test_returns_correct_count(self):
        """Should return K indices (or close if clusters are empty)."""
        N = 20
        cameras = np.random.randn(N, 16).astype(np.float32)
        K = 5
        selected = select_cameras_kmeans(cameras, K)
        assert len(selected) <= K
        assert len(selected) > 0

    def test_indices_in_range(self):
        """All returned indices should be valid camera indices."""
        N = 10
        cameras = np.random.randn(N, 16).astype(np.float32)
        K = 3
        selected = select_cameras_kmeans(cameras, K)
        for idx in selected:
            assert 0 <= idx < N

    def test_unique_indices(self):
        """Selected cameras should be unique."""
        # Use well-separated cameras to ensure distinct clusters
        cameras = np.zeros((6, 16), dtype=np.float32)
        for i in range(6):
            cameras[i, 0] = i * 100.0  # very separated
        selected = select_cameras_kmeans(cameras, K=3)
        assert len(selected) == len(set(selected))

    def test_invalid_shape_raises(self):
        """Should raise ValueError if cameras don't have 16 columns."""
        cameras = np.random.randn(5, 10).astype(np.float32)
        with pytest.raises(ValueError, match="16 values"):
            select_cameras_kmeans(cameras, K=2)


# ---------------------------------------------------------------------------
# Tests for build_full_proj_transform
# ---------------------------------------------------------------------------


class TestBuildFullProjTransform:
    def _make_camera(self):
        """Create a synthetic camera setup for testing."""
        fx, fy = 500.0, 500.0
        cx, cy = 320.0, 240.0
        W, H = 640, 480

        K = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
        )

        # Camera at origin looking down -Z
        w2c = torch.eye(4)

        return K, w2c, W, H

    def test_identity_camera_projection(self):
        """Point on camera axis projects to NDC center."""
        K, w2c, W, H = self._make_camera()
        full_proj = build_full_proj_transform(w2c, K, W, H)

        # Point at (0, 0, 5, 1) in world space — on the camera axis
        point = torch.tensor([[0.0, 0.0, 5.0, 1.0]])
        result = (point @ full_proj)
        ndc = result[0, :3] / result[0, 3]

        # With centered principal point, this should project to NDC ~(0, 0, ?)
        # ndc_x = 2*fx*0/(W*5) = 0, ndc_y = 2*fy*0/(H*5) = 0
        assert abs(ndc[0].item()) < 1e-5
        assert abs(ndc[1].item()) < 1e-5

    def test_known_pixel_projection(self):
        """3D point projects to expected NDC coordinates."""
        fx, fy = 500.0, 500.0
        cx, cy = 320.0, 240.0
        W, H = 640, 480

        K = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
        )
        w2c = torch.eye(4)
        full_proj = build_full_proj_transform(w2c, K, W, H)

        # Point at (1, 0.5, 10) in camera/world space
        Xc, Yc, Zc = 1.0, 0.5, 10.0
        point = torch.tensor([[Xc, Yc, Zc, 1.0]])

        # Manual pixel projection
        px = fx * Xc / Zc + cx  # 500*1/10 + 320 = 370
        py = fy * Yc / Zc + cy  # 500*0.5/10 + 240 = 265

        # Expected NDC (using the 3DGS convention: 2*fx/W normalization)
        expected_ndc_x = 2.0 * fx * Xc / (W * Zc)  # = 2*500*1/(640*10) = 0.15625
        expected_ndc_y = 2.0 * fy * Yc / (H * Zc)  # = 2*500*0.5/(480*10) = 0.10417

        result = point @ full_proj
        ndc = result[0, :3] / result[0, 3]

        torch.testing.assert_close(
            ndc[0], torch.tensor(expected_ndc_x), atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            ndc[1], torch.tensor(expected_ndc_y), atol=1e-5, rtol=1e-5
        )

    def test_w_equals_z_cam(self):
        """The w component of the projection should equal Z_camera."""
        K, w2c, W, H = self._make_camera()
        full_proj = build_full_proj_transform(w2c, K, W, H)

        Zc = 7.0
        point = torch.tensor([[2.0, -1.0, Zc, 1.0]])
        result = point @ full_proj
        # w should equal Zc (since camera is at identity)
        torch.testing.assert_close(
            result[0, 3], torch.tensor(Zc), atol=1e-5, rtol=1e-5
        )

    def test_translated_camera(self):
        """Projection works correctly with a non-identity W2C transform."""
        fx, fy = 500.0, 500.0
        cx, cy = 320.0, 240.0
        W, H = 640, 480

        K = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
        )
        # Camera translated to (3, 0, 0) in world, looking down -Z
        w2c = torch.eye(4)
        w2c[0, 3] = -3.0  # translation component: -R^T @ t, for t=[3,0,0]

        full_proj = build_full_proj_transform(w2c, K, W, H)

        # World point at (3, 0, 5) — directly in front of camera
        point = torch.tensor([[3.0, 0.0, 5.0, 1.0]])
        result = point @ full_proj
        ndc = result[0, :3] / result[0, 3]

        # In camera space this is (0, 0, 5), should project to center
        assert abs(ndc[0].item()) < 1e-5
        assert abs(ndc[1].item()) < 1e-5

    def test_matches_3dgs_convention(self):
        """Verify full_proj matches the original 3DGS getProjectionMatrix formula."""
        fx, fy = 600.0, 600.0
        W, H = 800, 600
        fovX = 2.0 * math.atan(W / (2.0 * fx))
        fovY = 2.0 * math.atan(H / (2.0 * fy))

        # Build using our function
        K = torch.tensor(
            [[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]]
        )
        w2c = torch.eye(4)
        full_proj = build_full_proj_transform(w2c, K, W, H)

        # Build manually using original 3DGS formula
        tanHalfFovX = math.tan(fovX / 2.0)
        tanHalfFovY = math.tan(fovY / 2.0)
        znear, zfar = 0.01, 100.0

        P = torch.zeros(4, 4)
        P[0, 0] = 2.0 * znear / (2.0 * tanHalfFovX * znear)
        P[1, 1] = 2.0 * znear / (2.0 * tanHalfFovY * znear)
        P[0, 2] = 0.0  # symmetric
        P[1, 2] = 0.0  # symmetric
        P[3, 2] = 1.0
        P[2, 2] = zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)

        # full_proj should equal W2C^T @ P^T = I^T @ P^T = P^T
        expected = P.T
        torch.testing.assert_close(full_proj, expected, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Tests for prepare_tensor
# ---------------------------------------------------------------------------


class TestPrepareTensor:
    def test_numpy_input(self):
        arr = np.array([1.0, 2.0, 3.0])
        t = prepare_tensor(arr, device="cpu")
        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.float32
        np.testing.assert_array_almost_equal(t.numpy(), arr)

    def test_tensor_input(self):
        t_in = torch.tensor([1.0, 2.0], dtype=torch.float64)
        t_out = prepare_tensor(t_in, device="cpu")
        assert t_out.dtype == torch.float32
        assert not t_out.requires_grad

    def test_list_input(self):
        t = prepare_tensor([4.0, 5.0], device="cpu")
        assert isinstance(t, torch.Tensor)
        assert t.shape == (2,)


# ---------------------------------------------------------------------------
# Tests for triangulate_points
# ---------------------------------------------------------------------------


class TestTriangulatePoints:
    def _make_two_camera_setup(self):
        """Create two cameras with known geometry for triangulation testing.

        Camera 1: at origin, looking down -Z.
        Camera 2: translated 1 unit along X, looking down -Z.
        """
        fx, fy = 500.0, 500.0
        W, H = 640, 480
        K = torch.tensor(
            [[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]]
        )

        w2c1 = torch.eye(4)
        w2c2 = torch.eye(4)
        w2c2[0, 3] = -1.0  # camera 2 at world position (1, 0, 0)

        P1 = build_full_proj_transform(w2c1, K, W, H)
        P2 = build_full_proj_transform(w2c2, K, W, H)

        return P1, P2, K, W, H

    def _project_point(self, P, point_world):
        """Project a 3D world point to NDC using the full projection matrix."""
        point_h = torch.tensor(
            [point_world[0], point_world[1], point_world[2], 1.0]
        ).unsqueeze(0)
        result = (point_h @ P).squeeze(0)
        ndc = result[:3] / result[3]
        return ndc[0].item(), ndc[1].item()

    def test_recovers_known_point(self):
        """Triangulation should recover a known 3D point from two views."""
        P1, P2, K, W, H = self._make_two_camera_setup()

        # Known 3D point
        world_point = [0.5, 0.3, 5.0]

        # Project to both cameras
        ndc1_x, ndc1_y = self._project_point(P1, world_point)
        ndc2_x, ndc2_y = self._project_point(P2, world_point)

        # Triangulate
        X, err1, err2 = triangulate_points(
            P1=P1.unsqueeze(0),
            P2=P2.unsqueeze(0),
            k1_x=np.array([ndc1_x]),
            k1_y=np.array([ndc1_y]),
            k2_x=np.array([ndc2_x]),
            k2_y=np.array([ndc2_y]),
            device="cpu",
        )

        # Check recovered point (Z has less precision due to baseline/depth ratio)
        recovered = X[0, :3].numpy()
        np.testing.assert_array_almost_equal(
            recovered, world_point, decimal=1
        )

        # Reprojection errors should be small
        assert err1[0] < 0.01
        assert err2[0] < 0.01

    def test_batch_triangulation(self):
        """Triangulation works for multiple points simultaneously."""
        P1, P2, K, W, H = self._make_two_camera_setup()

        # Multiple known 3D points
        world_points = [
            [0.0, 0.0, 3.0],
            [1.0, -0.5, 8.0],
            [-0.3, 0.7, 4.0],
        ]

        k1_x_list, k1_y_list = [], []
        k2_x_list, k2_y_list = [], []

        for wp in world_points:
            ndc1_x, ndc1_y = self._project_point(P1, wp)
            ndc2_x, ndc2_y = self._project_point(P2, wp)
            k1_x_list.append(ndc1_x)
            k1_y_list.append(ndc1_y)
            k2_x_list.append(ndc2_x)
            k2_y_list.append(ndc2_y)

        B = len(world_points)
        X, err1, err2 = triangulate_points(
            P1=P1.unsqueeze(0).expand(B, -1, -1),
            P2=P2.unsqueeze(0).expand(B, -1, -1),
            k1_x=np.array(k1_x_list),
            k1_y=np.array(k1_y_list),
            k2_x=np.array(k2_x_list),
            k2_y=np.array(k2_y_list),
            device="cpu",
        )

        assert X.shape == (B, 4)
        for i, wp in enumerate(world_points):
            np.testing.assert_array_almost_equal(
                X[i, :3].numpy(), wp, decimal=1
            )
            assert err1[i] < 0.01
            assert err2[i] < 0.01

    def test_noisy_observations(self):
        """Triangulation should still work with small noise, with bounded error."""
        P1, P2, K, W, H = self._make_two_camera_setup()

        world_point = [0.5, 0.3, 5.0]
        ndc1_x, ndc1_y = self._project_point(P1, world_point)
        ndc2_x, ndc2_y = self._project_point(P2, world_point)

        # Add small noise to observations
        noise = 0.002
        ndc1_x += noise
        ndc2_y -= noise

        X, err1, err2 = triangulate_points(
            P1=P1.unsqueeze(0),
            P2=P2.unsqueeze(0),
            k1_x=np.array([ndc1_x]),
            k1_y=np.array([ndc1_y]),
            k2_x=np.array([ndc2_x]),
            k2_y=np.array([ndc2_y]),
            device="cpu",
        )

        # Point should still be close, but with some error
        recovered = X[0, :3].numpy()
        distance = np.linalg.norm(np.array(world_point) - recovered)
        assert distance < 0.5  # reasonable tolerance for noisy observations

        # Reprojection errors should be non-zero but bounded
        assert err1[0] < 0.1
        assert err2[0] < 0.1


# ---------------------------------------------------------------------------
# Tests for select_best_keypoints
# ---------------------------------------------------------------------------


class TestSelectBestKeypoints:
    def test_selects_lowest_error(self):
        """Should select the triangulation with lowest max error for each point."""
        N = 5
        dim = 4

        # 2 NNs, 5 points
        points_nn0 = torch.randn(N, dim)
        points_nn1 = torch.randn(N, dim)
        NNs_points = torch.stack([points_nn0, points_nn1], dim=0)

        # NN0 has lower error for points 0,2,4; NN1 for points 1,3
        err1_nn0 = np.array([0.01, 0.5, 0.02, 0.8, 0.03])
        err2_nn0 = np.array([0.01, 0.5, 0.02, 0.8, 0.03])
        err1_nn1 = np.array([0.5, 0.01, 0.5, 0.01, 0.5])
        err2_nn1 = np.array([0.5, 0.01, 0.5, 0.01, 0.5])

        NNs_err1 = np.stack([err1_nn0, err1_nn1], axis=0)
        NNs_err2 = np.stack([err2_nn0, err2_nn1], axis=0)

        selected, sel_errors = select_best_keypoints(
            NNs_points, NNs_err1, NNs_err2, device="cpu"
        )

        assert selected.shape == (N, dim)
        # Points 0,2,4 should come from NN0
        torch.testing.assert_close(selected[0], points_nn0[0])
        torch.testing.assert_close(selected[2], points_nn0[2])
        torch.testing.assert_close(selected[4], points_nn0[4])
        # Points 1,3 should come from NN1
        torch.testing.assert_close(selected[1], points_nn1[1])
        torch.testing.assert_close(selected[3], points_nn1[3])

    def test_single_nn(self):
        """With a single NN, all points should be selected from it."""
        N = 3
        dim = 4
        points = torch.randn(1, N, dim)
        err1 = np.array([[0.1, 0.2, 0.3]])
        err2 = np.array([[0.05, 0.15, 0.25]])

        selected, sel_errors = select_best_keypoints(
            points, err1, err2, device="cpu"
        )

        assert selected.shape == (N, dim)
        torch.testing.assert_close(selected, points[0])


# ---------------------------------------------------------------------------
# Tests for extract_keypoints_and_colors_single
# ---------------------------------------------------------------------------


class TestExtractKeypointsAndColorsSingle:
    def _make_mock_roma_model(self):
        """Create a minimal mock roma model (not used in the single variant)."""

        class MockRoma:
            pass

        return MockRoma()

    def test_center_pixel_color(self):
        """A match pointing to the center pixel should extract the center color."""
        H_A, W_A = 100, 200
        H_B, W_B = 100, 200

        # Create test images with known colors
        imA = np.zeros((H_A, W_A, 3), dtype=np.uint8)
        imA[50, 100] = [255, 0, 0]  # red at center
        imB = np.zeros((H_B, W_B, 3), dtype=np.uint8)
        imB[50, 100] = [0, 255, 0]  # green at center

        # Match at NDC (0, 0) — should map to pixel center
        matches = torch.tensor([[0.0, 0.0, 0.0, 0.0]])

        roma_model = self._make_mock_roma_model()
        kptsA_norm, kptsB_norm, kptsA_color, kptsB_color = (
            extract_keypoints_and_colors_single(imA, imB, matches, roma_model)
        )

        # NDC (0,0) -> pixel ((W-1)/2, (H-1)/2) = (99.5, 49.5) -> rounded (100, 50)
        # But in the function: (0+1)*(W-1)/2 = 99.5 -> rounded to 100
        np.testing.assert_array_equal(kptsA_color[0], [255, 0, 0])
        np.testing.assert_array_equal(kptsB_color[0], [0, 255, 0])

    def test_ndc_roundtrip(self):
        """Normalized coordinates should roundtrip correctly."""
        H_A, W_A = 480, 640
        imA = np.zeros((H_A, W_A, 3), dtype=np.uint8)
        imB = np.zeros((H_A, W_A, 3), dtype=np.uint8)

        # Several NDC coordinates
        ndc_coords = torch.tensor(
            [
                [-1.0, -1.0, -1.0, -1.0],  # top-left
                [1.0, 1.0, 1.0, 1.0],  # bottom-right
                [0.0, 0.0, 0.0, 0.0],  # center
            ]
        )

        roma_model = self._make_mock_roma_model()
        kptsA_norm, kptsB_norm, _, _ = extract_keypoints_and_colors_single(
            imA, imB, ndc_coords, roma_model
        )

        # The function converts ndc->pixel->ndc, so output should match input
        np.testing.assert_array_almost_equal(
            kptsA_norm[:, 0], ndc_coords[:, 0].numpy(), decimal=5
        )
        np.testing.assert_array_almost_equal(
            kptsA_norm[:, 1], ndc_coords[:, 1].numpy(), decimal=5
        )

    def test_output_shapes(self):
        """Output shapes should match the number of matches."""
        N = 10
        H, W = 100, 100
        imA = np.zeros((H, W, 3), dtype=np.uint8)
        imB = np.zeros((H, W, 3), dtype=np.uint8)
        matches = torch.rand(N, 4) * 2 - 1  # random NDC in [-1, 1]

        roma_model = self._make_mock_roma_model()
        kptsA, kptsB, colorsA, colorsB = extract_keypoints_and_colors_single(
            imA, imB, matches, roma_model
        )

        assert kptsA.shape == (N, 2)
        assert kptsB.shape == (N, 2)
        assert colorsA.shape == (N, 3)
        assert colorsB.shape == (N, 3)


# ---------------------------------------------------------------------------
# Tests for projection-triangulation roundtrip
# ---------------------------------------------------------------------------


class TestProjectionTriangulationRoundtrip:
    """End-to-end tests verifying that project -> triangulate recovers points."""

    def test_roundtrip_identity_cameras(self):
        """Two cameras at different positions should allow triangulation."""
        fx, fy = 500.0, 500.0
        W, H = 640, 480
        K = torch.tensor(
            [[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]]
        )

        # Camera 1 at origin
        w2c1 = torch.eye(4)
        # Camera 2 translated 2 units right
        w2c2 = torch.eye(4)
        w2c2[0, 3] = -2.0

        P1 = build_full_proj_transform(w2c1, K, W, H)
        P2 = build_full_proj_transform(w2c2, K, W, H)

        # Generate random 3D points in front of both cameras
        torch.manual_seed(42)
        N = 50
        world_points = torch.randn(N, 3)
        world_points[:, 2] = world_points[:, 2].abs() + 3.0  # ensure positive Z

        # Project to both cameras
        points_h = torch.cat(
            [world_points, torch.ones(N, 1)], dim=1
        )  # [N, 4]
        proj1 = points_h @ P1  # [N, 4]
        ndc1 = proj1[:, :2] / proj1[:, 3:4]  # [N, 2]

        proj2 = points_h @ P2
        ndc2 = proj2[:, :2] / proj2[:, 3:4]

        # Triangulate
        X, err1, err2 = triangulate_points(
            P1=P1.unsqueeze(0).expand(N, -1, -1),
            P2=P2.unsqueeze(0).expand(N, -1, -1),
            k1_x=ndc1[:, 0].numpy(),
            k1_y=ndc1[:, 1].numpy(),
            k2_x=ndc2[:, 0].numpy(),
            k2_y=ndc2[:, 1].numpy(),
            device="cpu",
        )

        # All points should be recovered accurately
        recovered = X[:, :3]
        max_error = (recovered - world_points).abs().max().item()
        assert max_error < 0.1, f"Max triangulation error {max_error} too large"

        # All reprojection errors should be small
        assert np.max(err1) < 0.01
        assert np.max(err2) < 0.01

    def test_roundtrip_rotated_camera(self):
        """Triangulation works when camera 2 has a rotation."""
        fx, fy = 500.0, 500.0
        W, H = 640, 480
        K = torch.tensor(
            [[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]]
        )

        # Camera 1 at origin
        w2c1 = torch.eye(4)

        # Camera 2: translated and slightly rotated
        angle = 0.1  # ~5.7 degrees
        c, s = math.cos(angle), math.sin(angle)
        w2c2 = torch.tensor(
            [
                [c, 0.0, s, -1.5],
                [0.0, 1.0, 0.0, 0.0],
                [-s, 0.0, c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        P1 = build_full_proj_transform(w2c1, K, W, H)
        P2 = build_full_proj_transform(w2c2, K, W, H)

        # Known 3D points
        world_points = torch.tensor(
            [
                [0.0, 0.0, 5.0],
                [1.0, 1.0, 7.0],
                [-0.5, -0.3, 4.0],
            ]
        )
        N = world_points.shape[0]

        points_h = torch.cat([world_points, torch.ones(N, 1)], dim=1)
        proj1 = points_h @ P1
        ndc1 = proj1[:, :2] / proj1[:, 3:4]
        proj2 = points_h @ P2
        ndc2 = proj2[:, :2] / proj2[:, 3:4]

        X, err1, err2 = triangulate_points(
            P1=P1.unsqueeze(0).expand(N, -1, -1),
            P2=P2.unsqueeze(0).expand(N, -1, -1),
            k1_x=ndc1[:, 0].numpy(),
            k1_y=ndc1[:, 1].numpy(),
            k2_x=ndc2[:, 0].numpy(),
            k2_y=ndc2[:, 1].numpy(),
            device="cpu",
        )

        recovered = X[:, :3]
        max_error = (recovered - world_points).abs().max().item()
        assert max_error < 0.01, f"Max triangulation error {max_error} too large"


# ---------------------------------------------------------------------------
# Tests for Gaussian parameter computation
# ---------------------------------------------------------------------------


class TestGaussianParameterComputation:
    """Tests verifying that EDGS-style Gaussian parameters are computed correctly."""

    def test_opacity_masking(self):
        """Points with high reprojection error get negative opacity."""
        errors = np.array([0.001, 0.5, 0.005, 1.0, 0.002])
        tolerance = 0.01

        mask_bad = torch.tensor(
            errors > tolerance, dtype=torch.float32
        )
        opacities = -mask_bad * 10.0

        # Good points (0, 2, 4) should have opacity 0 (sigmoid(0) = 0.5)
        assert opacities[0].item() == 0.0
        assert opacities[2].item() == 0.0
        assert opacities[4].item() == 0.0

        # Bad points (1, 3) should have opacity -10 (sigmoid(-10) ≈ 0)
        assert opacities[1].item() == -10.0
        assert opacities[3].item() == -10.0

    def test_scale_computation(self):
        """Scales should be log(distance * factor), isotropic."""
        points = torch.tensor(
            [[0.0, 0.0, 5.0], [1.0, 0.0, 3.0]]
        )
        cam_center = torch.tensor([0.0, 0.0, 0.0])
        scaling_factor = 0.001

        dist = torch.linalg.norm(cam_center - points, dim=1, ord=2)
        scales = torch.log(
            (dist * scaling_factor).unsqueeze(1).repeat(1, 3)
        )

        # Point 0: dist=5, scale=log(0.005)
        expected_0 = math.log(5.0 * 0.001)
        assert abs(scales[0, 0].item() - expected_0) < 1e-6
        # Isotropic: all 3 scale dimensions equal
        assert scales[0, 0].item() == scales[0, 1].item() == scales[0, 2].item()

        # Point 1: dist=sqrt(10), scale=log(sqrt(10)*0.001)
        expected_1 = math.log(math.sqrt(10.0) * 0.001)
        assert abs(scales[1, 0].item() - expected_1) < 1e-5

    def test_scale_halving(self):
        """Post-init scale halving should add log(0.5) to scales."""
        scales = torch.tensor([[1.0, 1.0, 1.0], [-2.0, -2.0, -2.0]])
        scales_halved = scales + math.log(0.5)

        # exp(scale_halved) = exp(scale) * 0.5
        actual_sizes = torch.exp(scales_halved)
        expected_sizes = torch.exp(scales) * 0.5
        torch.testing.assert_close(actual_sizes, expected_sizes, atol=1e-6, rtol=1e-6)

    def test_rgb_to_sh_consistency(self):
        """RGB to SH conversion should be consistent with gsplat's rgb_to_sh."""
        from utils import rgb_to_sh

        # The EDGS code uses RGB2SH which is: (rgb - 0.5) / C0
        C0 = 0.28209479177387814
        rgbs = torch.tensor([[1.0, 0.0, 0.5], [0.3, 0.7, 0.1]])

        sh_gsplat = rgb_to_sh(rgbs)
        sh_edgs = (rgbs - 0.5) / C0

        torch.testing.assert_close(sh_gsplat, sh_edgs, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-v"])
