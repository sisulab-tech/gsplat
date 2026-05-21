// NOTE: Taken from https://github.com/jkulhanek/tetra-nerf
#pragma once

#include <cuda_runtime.h>
#include <vector>

namespace at {
class Tensor;
}

namespace gsplat {

std::vector<uint4> triangulate(size_t num_points, float3 *points);
at::Tensor py_triangulate(const at::Tensor &points);

// float find_average_spacing(size_t num_points, float3 *points);

} // namespace gsplat
