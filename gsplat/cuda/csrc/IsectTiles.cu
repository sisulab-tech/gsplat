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
 * Gaussian Tile Intersection
 ****************************************************************************/

template <typename scalar_t>
__global__ void points_isect_tiles(
    // if the data is [C, N, ...] or [nnz, ...] (packed)
    const bool packed,
    // parallelize over C * N, only used if packed is False
    const uint32_t C,
    const uint32_t N,
    // parallelize over nnz, only used if packed is True
    const uint32_t nnz,
    const int64_t *__restrict__ camera_ids,   // [nnz] optional
    const int64_t *__restrict__ gaussian_ids, // [nnz] optional
    // data
    const scalar_t *__restrict__ means2d,            // [C, N, 2] or [nnz, 2]
    const int32_t *__restrict__ radii,               // [C, N] or [nnz]
    const scalar_t *__restrict__ depths,             // [C, N] or [nnz]
    const int64_t *__restrict__ cum_tiles_per_gauss, // [C, N] or [nnz]
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_n_bits,
    int64_t *__restrict__ isect_ids,  // [n_isects]
    int32_t *__restrict__ flatten_ids // [n_isects]
) {
    // For now we'll upcast float16 and bfloat16 to float32
    // using OpT = typename OpType<scalar_t>::type;

    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    // bool first_pass = cum_tiles_per_gauss == nullptr;
    if (idx >= (packed ? nnz : C * N)) {
        return;
    }

    const float radius = radii[idx];
    if (radius <= 0) {
        return;
    }

    vec2 mean2d = glm::make_vec2(means2d + 2 * idx);

    float tile_x = mean2d.x / static_cast<float>(tile_size);
    float tile_y = mean2d.y / static_cast<float>(tile_size);

    uint32_t x = min(max(0, (uint32_t)floor(tile_x)), tile_width);
    uint32_t y = min(max(0, (uint32_t)floor(tile_y)), tile_height);

    int64_t cid; // camera id
    if (packed) {
        // parallelize over nnz
        cid = camera_ids[idx];
        // gid = gaussian_ids[idx];
    } else {
        // parallelize over C * N
        cid = idx / N;
        // gid = idx % N;
    }
    const int64_t cid_enc = cid << (32 + tile_n_bits);

    int64_t depth_id_enc = (int64_t)*(int32_t *)&(depths[idx]);
    int64_t cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
    int64_t tile_id = y * tile_width + x;
    // e.g. tile_n_bits = 22:
    // camera id (10 bits) | tile id (22 bits) | depth (32 bits)
    isect_ids[cur_idx] = cid_enc | (tile_id << 32) | depth_id_enc;
    // the flatten index in [C * N] or [nnz]
    flatten_ids[cur_idx] = static_cast<int32_t>(idx);
}

std::tuple<at::Tensor, at::Tensor> points_isect_tiles_tensor(
    const at::Tensor means2d,                    // [C, N, 2] or [nnz, 2]
    const at::Tensor radii,                      // [C, N] or [nnz]
    const at::Tensor depths,                     // [C, N] or [nnz]
    const at::optional<at::Tensor> camera_ids,   // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t C,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const bool sort,
    const bool double_buffer
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(radii);
    CHECK_INPUT(depths);
    if (camera_ids.has_value()) {
        CHECK_INPUT(camera_ids.value());
    }
    if (gaussian_ids.has_value()) {
        CHECK_INPUT(gaussian_ids.value());
    }
    bool packed = means2d.dim() == 2;

    uint32_t N = 0, nnz = 0, total_elems = 0;
    int64_t *camera_ids_ptr = nullptr;
    int64_t *gaussian_ids_ptr = nullptr;
    if (packed) {
        nnz = means2d.size(0);
        total_elems = nnz;
        TORCH_CHECK(
            camera_ids.has_value() && gaussian_ids.has_value(),
            "When packed is set, camera_ids and gaussian_ids must be provided."
        );
        camera_ids_ptr = camera_ids.value().data_ptr<int64_t>();
        gaussian_ids_ptr = gaussian_ids.value().data_ptr<int64_t>();
    } else {
        N = means2d.size(1); // number of gaussians
        total_elems = C * N;
    }

    uint32_t n_tiles = tile_width * tile_height;
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // the number of bits needed to encode the camera id and tile id
    // Note: std::bit_width requires C++20
    // uint32_t tile_n_bits = std::bit_width(n_tiles);
    // uint32_t cam_n_bits = std::bit_width(C);
    uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
    uint32_t cam_n_bits = (uint32_t)floor(log2(C)) + 1;
    // the first 32 bits are used for the camera id and tile id altogether, so
    // check if we have enough bits for them.
    assert(tile_n_bits + cam_n_bits <= 32);

    at::Tensor cum_tiles_per_gauss = at::cumsum(radii.view({-1}), 0);

    int64_t n_isects = cum_tiles_per_gauss[-1].item<int64_t>();

    at::Tensor isect_ids =
        at::empty({n_isects}, depths.options().dtype(at::kLong));
    at::Tensor flatten_ids =
        at::empty({n_isects}, depths.options().dtype(at::kInt));

    if (n_isects) {
        AT_DISPATCH_FLOATING_TYPES(
            means2d.scalar_type(),
            "points_isect_tiles",
            [&]() {
                points_isect_tiles<scalar_t>
                    <<<(total_elems + N_THREADS_PACKED - 1) / N_THREADS_PACKED,
                       N_THREADS_PACKED,
                       0,
                       stream>>>(
                        packed,
                        C,
                        N,
                        nnz,
                        camera_ids_ptr,
                        gaussian_ids_ptr,
                        means2d.data_ptr<scalar_t>(),
                        radii.data_ptr<int32_t>(),
                        depths.data_ptr<scalar_t>(),
                        cum_tiles_per_gauss.data_ptr<int64_t>(),
                        tile_size,
                        tile_width,
                        tile_height,
                        tile_n_bits,
                        isect_ids.data_ptr<int64_t>(),
                        flatten_ids.data_ptr<int32_t>()
                    );
            }
        );
    }

    // optionally sort the Gaussians by isect_ids
    if (n_isects && sort) {
        at::Tensor isect_ids_sorted = at::empty_like(isect_ids);
        at::Tensor flatten_ids_sorted = at::empty_like(flatten_ids);

        // https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
        // DoubleBuffer reduce the auxiliary memory usage from O(N+P) to O(P)
        if (double_buffer) {
            // Create a set of DoubleBuffers to wrap pairs of device pointers
            cub::DoubleBuffer<int64_t> d_keys(
                isect_ids.data_ptr<int64_t>(),
                isect_ids_sorted.data_ptr<int64_t>()
            );
            cub::DoubleBuffer<int32_t> d_values(
                flatten_ids.data_ptr<int32_t>(),
                flatten_ids_sorted.data_ptr<int32_t>()
            );
            CUB_WRAPPER(
                cub::DeviceRadixSort::SortPairs,
                d_keys,
                d_values,
                n_isects,
                0,
                32 + tile_n_bits + cam_n_bits,
                stream
            );
            switch (d_keys.selector) {
            case 0: // sorted items are stored in isect_ids
                isect_ids_sorted = isect_ids;
                break;
            case 1: // sorted items are stored in isect_ids_sorted
                break;
            }
            switch (d_values.selector) {
            case 0: // sorted items are stored in flatten_ids
                flatten_ids_sorted = flatten_ids;
                break;
            case 1: // sorted items are stored in flatten_ids_sorted
                break;
            }
            // printf("DoubleBuffer d_keys selector: %d\n", d_keys.selector);
            // printf("DoubleBuffer d_values selector: %d\n",
            // d_values.selector);
        } else {
            CUB_WRAPPER(
                cub::DeviceRadixSort::SortPairs,
                isect_ids.data_ptr<int64_t>(),
                isect_ids_sorted.data_ptr<int64_t>(),
                flatten_ids.data_ptr<int32_t>(),
                flatten_ids_sorted.data_ptr<int32_t>(),
                n_isects,
                0,
                32 + tile_n_bits + cam_n_bits,
                stream
            );
        }
        return std::make_tuple(isect_ids_sorted, flatten_ids_sorted);
    } else {
        return std::make_tuple(isect_ids, flatten_ids);
    }
}

} // namespace gsplat
