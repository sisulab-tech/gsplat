# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

gsplat is a CUDA-accelerated library for differentiable Gaussian splatting with Python bindings. It provides optimized rasterization of 3D/2D Gaussians for real-time radiance field rendering.

## Build & Development

**Environment**: Managed via [pixi.toml](pixi.toml) (Python 3.10, CUDA 12.4.1, PyTorch 2.4.1).

**Install for development (recommended — uses JIT compilation for CUDA)**:
```bash
BUILD_NO_CUDA=1 pip install -e .[dev]
```
JIT-compiled CUDA code is cached at `~/.cache/torch_extensions/py*-cu*/` and recompiles incrementally when CUDA sources change.

**Install with pre-compiled CUDA**:
```bash
pip install -e .[dev]
```

**Environment variables**:
- `BUILD_NO_CUDA=1` — skip CUDA compilation at install time, use JIT instead
- `MAX_JOBS` — parallel build jobs (defaults to 10)
- `WITH_SYMBOLS=1` — include debug symbols in binary
- `LINE_INFO=1` — include line info for CUDA debugging

## Testing

All tests require a CUDA GPU. CI runs on CPU so CUDA tests must pass locally before committing.

```bash
pytest tests/                          # all tests
pytest tests/test_basic.py             # single file
pytest tests/test_basic.py::test_proj  # single test
```

Test data is loaded from `assets/test_garden.npz` via `gsplat._helper.load_test_data()`.

## Formatting

**Python** — Black (checked in CI):
```bash
black . gsplat/ tests/ examples/ profiling/
```

**C++/CUDA** — clang-format (LLVM style, indent 4):
```bash
bash formatter.sh
```

## Architecture

### Rendering Pipeline

The main entry point is `rasterization()` in [rendering.py](gsplat/rendering.py). The pipeline:

1. **Projection** — `fully_fused_projection()` projects 3D Gaussians to 2D screen space (means2d, conics, depths)
2. **Tile intersection** — `isect_tiles()` determines which Gaussians overlap each 16×16 pixel tile
3. **Rasterization** — `rasterize_to_pixels()` alpha-composites Gaussians per tile, front-to-back
4. Both forward and backward passes are implemented as fused CUDA kernels

Variants: `rasterization_2dgs()` for 2D Gaussian splatting, `fully_fused_projection_with_ut()` for unscented transform distortion modeling.

### CUDA Backend

- [gsplat/cuda/csrc/](gsplat/cuda/csrc/) — CUDA kernels (~40 files): projection, rasterization, spherical harmonics, Adam optimizer
- [gsplat/cuda/_wrapper.py](gsplat/cuda/_wrapper.py) — Python bindings to compiled CUDA functions
- [gsplat/cuda/_backend.py](gsplat/cuda/_backend.py) — JIT compilation and lazy loading of the `gsplat.csrc` module
- [gsplat/cuda/_torch_impl.py](gsplat/cuda/_torch_impl.py) — Pure PyTorch fallback implementations (useful for debugging)
- [gsplat/cuda/ext.cpp](gsplat/cuda/ext.cpp) — PyTorch C++ extension entry point
- Third-party GLM library at `gsplat/cuda/csrc/third_party/glm` for GPU math

### Training Strategies

[gsplat/strategy/](gsplat/strategy/) controls Gaussian densification during training:
- `DefaultStrategy` — clone/split/prune from the original 3DGS paper
- `MCMCStrategy` — probabilistic MCMC-based densification
- Operations in [ops.py](gsplat/strategy/ops.py): `duplicate()`, `split()`, `remove()`, `reset_opa()`
- Interface: `step_pre_backward()` and `step_post_backward()` called around `loss.backward()`

### Key Modules

- [gsplat/optimizers/selective_adam.py](gsplat/optimizers/selective_adam.py) — fused CUDA Adam with visibility masking (only updates active Gaussians)
- [gsplat/distributed.py](gsplat/distributed.py) — multi-GPU distributed rendering
- [gsplat/utils.py](gsplat/utils.py) — quaternion/rotation utilities, depth operations
- [gsplat/exporter.py](gsplat/exporter.py) — export Gaussians to PLY and other formats

### Examples

Training scripts are in [examples/](examples/). Main entry point is [simple_trainer.py](examples/simple_trainer.py) using `tyro` for CLI args. Benchmarks in [examples/benchmarks/](examples/benchmarks/).

## Documentation

```bash
pip install -r docs/requirements.txt
sphinx-build docs/source _build
```
