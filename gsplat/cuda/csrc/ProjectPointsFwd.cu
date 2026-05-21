#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "Common.h"
#include "Projection.h"
#include "Utils.cuh"

namespace gsplat {
namespace cg = cooperative_groups;

/****************************************************************************
 * Projection of Points (Single Batch) Forward Pass
 ****************************************************************************/

template <typename scalar_t>
__global__ void project_points_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ means,    // [N, 3]
    const scalar_t *__restrict__ viewmats, // [C, 4, 4]
    const scalar_t *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const scalar_t near_plane,
    const scalar_t far_plane,
    // outputs
    int32_t *__restrict__ radii,    // [C, N]
    scalar_t *__restrict__ means2d, // [C, N, 2]
    scalar_t *__restrict__ depths   // [C, N]
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    viewmats += cid * 16;
    Ks += cid * 9;

    // glm is column-major but input is row-major
    mat3 R = mat3(
        viewmats[0],
        viewmats[4],
        viewmats[8], // 1st column
        viewmats[1],
        viewmats[5],
        viewmats[9], // 2nd column
        viewmats[2],
        viewmats[6],
        viewmats[10] // 3rd column
    );
    vec3 t = vec3(viewmats[3], viewmats[7], viewmats[11]);

    // transform Gaussian center to camera space
    vec3 mean_c;
    posW2C(R, t, glm::make_vec3(means), mean_c);
    if (mean_c.z < near_plane || mean_c.z > far_plane) {
        radii[idx] = 0;
        return;
    }

    // project the point to image plane
    vec2 mean2d;

    const scalar_t fx = Ks[0];
    const scalar_t fy = Ks[4];
    const scalar_t cx = Ks[2];
    const scalar_t cy = Ks[5];

    scalar_t x = mean_c[0], y = mean_c[1], z = mean_c[2];
    scalar_t rz = 1.f / z;
    mean2d = vec2({fx * x * rz + cx, fy * y * rz + cy});

    // mask out gaussians outside the image region
    if (mean2d.x <= 0 || mean2d.x >= image_width || mean2d.y <= 0 ||
        mean2d.y >= image_height) {
        radii[idx] = 0;
        return;
    }

    // write to outputs
    radii[idx] = 1;
    means2d[idx * 2] = mean2d.x;
    means2d[idx * 2 + 1] = mean2d.y;
    depths[idx] = mean_c.z;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> project_points_fwd_tensor(
    const at::Tensor means,    // [N, 3]
    const at::Tensor viewmats, // [C, 4, 4]
    const at::Tensor Ks,       // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float near_plane,
    const float far_plane
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    at::Tensor radii = at::empty({C, N}, means.options().dtype(at::kInt));
    at::Tensor means2d = at::empty({C, N, 2}, means.options());
    at::Tensor depths = at::empty({C, N}, means.options());

    if (C && N) {
        project_points_fwd_kernel<float>
            <<<(C * N + N_THREADS_PACKED - 1) / N_THREADS_PACKED,
               N_THREADS_PACKED,
               0,
               stream>>>(
                C,
                N,
                means.data_ptr<float>(),
                viewmats.data_ptr<float>(),
                Ks.data_ptr<float>(),
                image_width,
                image_height,
                near_plane,
                far_plane,
                radii.data_ptr<int32_t>(),
                means2d.data_ptr<float>(),
                depths.data_ptr<float>()
            );
    }
    return std::make_tuple(radii, means2d, depths);
}
}
