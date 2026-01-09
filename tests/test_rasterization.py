"""Tests for the functions in the CUDA extension.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

from typing import Optional

import pytest
import torch

device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("per_view_color", [True, False])
@pytest.mark.parametrize("sh_degree", [None, 3])
@pytest.mark.parametrize("render_mode", ["RGB", "RGB+D", "D"])
@pytest.mark.parametrize("packed", [True, False])
def test_rasterization(
    per_view_color: bool, sh_degree: Optional[int], render_mode: str, packed: bool
):
    from gsplat.rendering import _rasterization, rasterization

    torch.manual_seed(42)

    C, N = 2, 10_000
    means = torch.rand(N, 3, device=device)
    quats = torch.randn(N, 4, device=device)
    scales = torch.rand(N, 3, device=device)
    opacities = torch.rand(N, device=device)
    if per_view_color:
        if sh_degree is None:
            colors = torch.rand(C, N, 3, device=device)
        else:
            colors = torch.rand(C, N, (sh_degree + 1) ** 2, 3, device=device)
    else:
        if sh_degree is None:
            colors = torch.rand(N, 3, device=device)
        else:
            colors = torch.rand(N, (sh_degree + 1) ** 2, 3, device=device)

    width, height = 300, 200
    focal = 300.0
    Ks = torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        device=device,
    ).expand(C, -1, -1)
    viewmats = torch.eye(4, device=device).expand(C, -1, -1)

    renders, alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode=render_mode,
        packed=packed,
    )

    if render_mode == "D":
        assert renders.shape == (C, height, width, 1)
    elif render_mode == "RGB":
        assert renders.shape == (C, height, width, 3)
    elif render_mode == "RGB+D":
        assert renders.shape == (C, height, width, 4)

    _renders, _alphas, _meta = _rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode=render_mode,
    )
    torch.testing.assert_close(renders, _renders, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(alphas, _alphas, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_compact_box():
    """Test Compact Box tile intersection."""
    from gsplat import points_isect_tiles_compact_box
    from gsplat.rendering import rasterization

    torch.manual_seed(42)

    # Create simple Gaussians
    N = 100
    C = 1
    means = torch.rand(N, 3, device=device) * 2 - 1  # [-1, 1]
    quats = torch.randn(N, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.rand(N, 3, device=device) * 0.1
    opacities = torch.rand(N, device=device) * 0.5 + 0.5  # [0.5, 1.0]
    colors = torch.rand(N, 3, device=device)

    width, height = 256, 256
    focal = 256.0
    Ks = torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        device=device,
    ).expand(C, -1, -1)
    viewmats = torch.eye(4, device=device).unsqueeze(0)  # [C, 4, 4]

    # Run rasterization to get projection info
    renders, alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        packed=False,
    )

    # Test Compact Box tile intersection
    means2d = info["means2d"]  # [C, N, 2]
    conics = info["conics"]    # [C, N, 3]
    depths = info["depths"]    # [C, N]
    radii = info["radii"]      # [C, N, 2]

    # Only test with visible Gaussians
    valid_mask = (radii > 0).all(-1).any(0)  # [N]
    if valid_mask.sum() == 0:
        pytest.skip("No visible Gaussians in test")

    # Get opacities per view
    opacities_2d = opacities.unsqueeze(0).expand(C, -1)  # [C, N]

    tile_size = 16
    tile_width = (width + tile_size - 1) // tile_size
    tile_height = (height + tile_size - 1) // tile_size

    # Test with different compact_box_mult values
    for mult in [0.5, 0.7, 1.0]:
        isect_ids, flatten_ids = points_isect_tiles_compact_box(
            means2d=means2d,
            conics=conics,
            opacities=opacities_2d,
            depths=depths,
            tile_size=tile_size,
            tile_width=tile_width,
            tile_height=tile_height,
            compact_box_mult=mult,
            sort=True,
            packed=False,
        )

        # Basic sanity checks
        assert isect_ids.dtype == torch.int64
        assert flatten_ids.dtype == torch.int32
        assert isect_ids.shape == flatten_ids.shape
        assert len(isect_ids.shape) == 1

        # Check that all flatten_ids are valid indices
        assert flatten_ids.min() >= 0
        assert flatten_ids.max() < C * N

        # Check that isect_ids are sorted (required for rasterization)
        assert torch.all(isect_ids[1:] >= isect_ids[:-1])

    print(f"✓ Compact Box test passed with {valid_mask.sum()}/{N} visible Gaussians")
