#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "Common.h"
#include "Projection.h"
#include "Utils.cuh"

namespace gsplat {
namespace cg = cooperative_groups;

/****************************************************************************
 * Projection of Gaussians (Single Batch) Forward Pass
 ****************************************************************************/

template <typename scalar_t>
__global__ void view_to_gaussians_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ means,       // [N, 3]
    const scalar_t *__restrict__ quats,       // [N, 4]
    const scalar_t *__restrict__ scales,      // [N, 3]
    const scalar_t *__restrict__ camtoworlds, // [C, 4, 4]
    const int32_t *__restrict__ radii, // [C, N, 2]
    // outputs
    scalar_t *__restrict__ view2gaussians // [C, N, 12]
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N || radii[idx * 2] <= 0 || radii[idx * 2 + 1] <= 0) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    camtoworlds += cid * 16;
    quats += gid * 4;
    scales += gid * 3;
    view2gaussians += idx * 12;

    // glm is column-major but input is row-major
    mat3 camtoworlds_R = mat3(
        camtoworlds[0],
        camtoworlds[4],
        camtoworlds[8], // 1st column
        camtoworlds[1],
        camtoworlds[5],
        camtoworlds[9], // 2nd column
        camtoworlds[2],
        camtoworlds[6],
        camtoworlds[10] // 3rd column
    );
    vec3 camtoworlds_t =
        vec3(camtoworlds[3], camtoworlds[7], camtoworlds[11]);

    vec3 mean = glm::make_vec3(means);
    vec4 quat = glm::make_vec4(quats);
    vec3 scale = glm::make_vec3(scales);
    mat3 rotmat = quat_to_rotmat(quat);

    mat3 worldtogaussian_R = glm::transpose(rotmat);
    vec3 worldtogaussian_t = -worldtogaussian_R * mean;

    mat3 view2gaussian_R = worldtogaussian_R * camtoworlds_R;
    vec3 view2gaussian_t =
        worldtogaussian_R * camtoworlds_t + worldtogaussian_t;

    // Whitened ray in the gaussian frame: for a camera ray r (z = 1) at depth
    // t, the Mahalanobis distance is |o + t * M r|^2 with M = S^-1 R and
    // o = S^-1 t_v2g. Storing M and o instead of the expanded quadric
    // (r^T R^T S^-2 R r, ...) avoids float32 cancellation between ~1/s^2
    // terms for very thin gaussians. Layout: M row-major (9), then o (3).
    vec3 inv_s = {
        rsqrtf(scale.x * scale.x + 1e-10f),
        rsqrtf(scale.y * scale.y + 1e-10f),
        rsqrtf(scale.z * scale.z + 1e-10f)
    };
#pragma unroll
    for (uint32_t r = 0; r < 3; ++r) {
#pragma unroll
        for (uint32_t c = 0; c < 3; ++c) {
            view2gaussians[r * 3 + c] = inv_s[r] * view2gaussian_R[c][r];
        }
        view2gaussians[9 + r] = inv_s[r] * view2gaussian_t[r];
    }
}

at::Tensor view_to_gaussians_fwd_tensor(
    const at::Tensor means,    // [N, 3]
    const at::Tensor quats,    // [N, 4]
    const at::Tensor scales,   // [N, 3]
    const at::Tensor viewmats, // [C, 4, 4]
    const at::Tensor radii     // [C, N]
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    CHECK_INPUT(viewmats);
    CHECK_INPUT(radii);

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    at::Tensor view2gaussians = at::empty({C, N, 12}, means.options());

    if (C && N) {
        view_to_gaussians_fwd_kernel<float>
            <<<(C * N + N_THREADS_PACKED - 1) / N_THREADS_PACKED, N_THREADS_PACKED, 0, stream>>>(
                C,
                N,
                means.data_ptr<float>(),
                quats.data_ptr<float>(),
                scales.data_ptr<float>(),
                viewmats.data_ptr<float>(),
                radii.data_ptr<int32_t>(),
                view2gaussians.data_ptr<float>()
            );
        // view_to_gaussians_fwd_kernel<double><<<(C * N + N_THREADS_PACKED - 1) /
        // N_THREADS_PACKED, N_THREADS_PACKED, 0, stream>>>(
        //     C, N, means.data_ptr<double>(),
        //     quats.data_ptr<double>(),
        //     scales.data_ptr<double>(),
        //     viewmats.data_ptr<double>(),
        //     radii.data_ptr<int32_t>(),
        //     view2gaussians.data_ptr<double>());
    }
    return view2gaussians;
}
}
