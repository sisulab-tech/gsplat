#include <ATen/DeviceGuard.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Ops.h"

namespace gsplat {

namespace cg = cooperative_groups;

// Per Gaussian, the number of masked pixels it contributes to: the same
// alpha >= ALPHA_THRESHOLD and transmittance tests as the 3DGS forward pass.
// Pixels outside the mask are skipped. Counts are kept per batch slot in
// shared memory and flushed once per batch.
__global__ void rasterize_to_pixels_3dgs_count_kernel(
    const uint32_t C,
    const uint32_t n_isects,
    const vec2 *__restrict__ means2d,       // [C, N, 2] or [nnz, 2]
    const vec3 *__restrict__ conics,        // [C, N, 3] or [nnz, 3]
    const float *__restrict__ opacities,    // [C, N] or [nnz]
    const bool *__restrict__ pixel_masks,   // [C, image_height, image_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    int32_t *__restrict__ counts              // [C * N] or [nnz]
) {
    auto block = cg::this_thread_block();
    int32_t camera_id = block.group_index().x;
    int32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += camera_id * tile_height * tile_width;
    pixel_masks += camera_id * image_height * image_width;

    float px = (float)j + 0.5f;
    float py = (float)i + 0.5f;
    bool inside = (i < image_height && j < image_width);
    bool done = !inside || !pixel_masks[i * image_width + j];

    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;                     // [block_size]
    int32_t *cnt_batch = &id_batch[block_size];           // [block_size]
    vec3 *xy_opacity_batch =
        reinterpret_cast<vec3 *>(&cnt_batch[block_size]); // [block_size]
    vec3 *conic_batch =
        reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]); // [block_size]

    float T = 1.0f;
    uint32_t tr = block.thread_rank();

    for (uint32_t b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        cnt_batch[tr] = 0;
        if (idx < range_end) {
            int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            const vec2 xy = means2d[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opacities[g]};
            conic_batch[tr] = conics[g];
        }
        block.sync();

        uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const vec3 conic = conic_batch[t];
            const vec3 xy_opac = xy_opacity_batch[t];
            const vec2 delta = {xy_opac.x - px, xy_opac.y - py};
            const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                conic.y * delta.x * delta.y;
            float alpha = min(0.999f, xy_opac.z * __expf(-sigma));
            if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                continue;
            }
            const float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) {
                done = true;
                break;
            }
            atomicAdd(&cnt_batch[t], 1);
            T = next_T;
        }
        block.sync();

        if (idx < range_end && cnt_batch[tr] > 0) {
            atomicAdd(&counts[id_batch[tr]], cnt_batch[tr]);
        }
    }
}

at::Tensor rasterize_to_pixels_3dgs_count(
    const at::Tensor means2d,     // [C, N, 2] or [nnz, 2]
    const at::Tensor conics,      // [C, N, 3] or [nnz, 3]
    const at::Tensor opacities,   // [C, N] or [nnz]
    const at::Tensor pixel_masks, // [C, image_height, image_width] bool
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const at::Tensor tile_offsets, // [C, tile_height, tile_width]
    const at::Tensor flatten_ids   // [n_isects]
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(opacities);
    CHECK_INPUT(pixel_masks);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    TORCH_CHECK(pixel_masks.scalar_type() == at::kBool, "pixel_masks must be bool");

    uint32_t C = tile_offsets.size(0);
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);
    uint32_t n_isects = flatten_ids.size(0);
    TORCH_CHECK(
        pixel_masks.size(0) == C && pixel_masks.size(1) == image_height &&
            pixel_masks.size(2) == image_width,
        "pixel_masks must be [C, image_height, image_width]"
    );

    at::Tensor counts =
        at::zeros({means2d.numel() / 2}, means2d.options().dtype(at::kInt));
    if (n_isects == 0) {
        return counts;
    }

    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {C, tile_height, tile_width};
    int64_t shmem_size = tile_size * tile_size *
                         (2 * sizeof(int32_t) + sizeof(vec3) + sizeof(vec3));
    if (cudaFuncSetAttribute(
            rasterize_to_pixels_3dgs_count_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }
    rasterize_to_pixels_3dgs_count_kernel<<<
        grid,
        threads,
        shmem_size,
        at::cuda::getCurrentCUDAStream()>>>(
        C,
        n_isects,
        reinterpret_cast<vec2 *>(means2d.data_ptr<float>()),
        reinterpret_cast<vec3 *>(conics.data_ptr<float>()),
        opacities.data_ptr<float>(),
        pixel_masks.data_ptr<bool>(),
        image_width,
        image_height,
        tile_size,
        tile_width,
        tile_height,
        tile_offsets.data_ptr<int32_t>(),
        flatten_ids.data_ptr<int32_t>(),
        counts.data_ptr<int32_t>()
    );
    return counts;
}

} // namespace gsplat
