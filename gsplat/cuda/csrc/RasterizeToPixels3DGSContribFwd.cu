// Forward-only 3DGS rasterizer that also records, per Gaussian and per view,
// the maximum contribution (alpha * T) it made to any pixel, the id of that
// pixel, and optionally its minimum distance to the accumulated depth.
//
// Port of the `renderCUDA` additions in 3DGS-to-PC's
// `gaussian-pointcloud-rasterization` (commit 6792c6f). The blending
// constants of that fork are reproduced here and differ from the ones the
// training kernels use: alpha is capped at 0.99 (not 0.999) and a pixel stops
// when T * (1 - alpha) < 1e-4 (not <=).
//
// Four deliberate differences from upstream:
//   1. Race-free reductions. Upstream writes the per-Gaussian pixel id and the
//      global per-Gaussian buffers non-atomically, so its output depends on
//      tile scheduling. The same max/argmax/min are computed here with atomic
//      compare-and-swap (contribution and pixel id packed into one 64-bit
//      word), which makes the result deterministic. Ties on the contribution
//      resolve to the smallest pixel id.
//   2. Principal point. Upstream assumes the image centre; this kernel
//      consumes gsplat's projection, which uses the full K (and gsplat's
//      pixel-centre convention, px = j + 0.5).
//   3. Tile bounds. Upstream bins Gaussians with a 3-sigma square; gsplat uses
//      exact ellipse-vs-tile culling, so a few pairs beyond 3 sigma that
//      upstream never evaluated are absent here too.
//   4. Surface distances only come from pixels inside the image. Upstream lets
//      the padding threads of an edge tile vote too, with an expected depth of
//      zero, which makes the result depend on the image size.
//
// Upstream also breaks masked threads out of the batch loop entirely, which
// leaves stale shared-memory slots for the remaining threads once a tile holds
// more than one batch. Here masked threads keep fetching but neither render
// nor vote, which is what that code intends.

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cfloat>

#include "Common.h"
#include "Rasterization.h"

namespace gsplat {

namespace cg = cooperative_groups;

// Upstream's blending constants, which differ from the training kernels'.
#define CONTRIB_ALPHA_CAP 0.99f
#define CONTRIB_T_EPS 1e-4f

// Pack a non-negative contribution and the pixel that produced it into one
// word. Float bit patterns order like their values for non-negative floats,
// and the complemented pixel id makes ties resolve to the smallest id.
__device__ inline unsigned long long
pack_contrib(const float contrib, const uint32_t pix_id) {
    return (static_cast<unsigned long long>(__float_as_uint(contrib)) << 32) |
           static_cast<unsigned long long>(~pix_id);
}

template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_3dgs_contrib_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const vec2 *__restrict__ means2d,         // [C, N, 2]
    const vec3 *__restrict__ conics,          // [C, N, 3]
    const scalar_t *__restrict__ colors,      // [C, N, CDIM]
    const scalar_t *__restrict__ opacities,   // [C, N]
    const scalar_t *__restrict__ depths,      // [C, N]
    const scalar_t *__restrict__ backgrounds, // [C, CDIM]
    const bool *__restrict__ pixel_masks,     // [C, image_height, image_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    scalar_t *__restrict__ render_colors,     // [C, image_height, image_width, CDIM]
    scalar_t *__restrict__ render_depths,     // [C, image_height, image_width]
    unsigned long long *__restrict__ max_contrib_packed, // [C * N]
    scalar_t *__restrict__ min_surface_dist,             // [C * N]
    const bool calculate_surface_distance
) {
    auto block = cg::this_thread_block();
    int32_t camera_id = block.group_index().x;
    int32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += camera_id * tile_height * tile_width;
    render_colors += camera_id * image_height * image_width * CDIM;
    render_depths += camera_id * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += camera_id * CDIM;
    }
    if (pixel_masks != nullptr) {
        pixel_masks += camera_id * image_height * image_width;
    }

    float px = (float)j + 0.5f;
    float py = (float)i + 0.5f;
    int32_t pix_id = i * image_width + j;

    bool inside = (i < image_height && j < image_width);
    bool masked = inside && pixel_masks != nullptr && !pixel_masks[pix_id];
    // masked pixels neither render nor vote, but still help fetch the tile
    bool done = !inside || masked;

    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    // the 64-bit array goes first so that it stays 8-byte aligned
    extern __shared__ int s[];
    unsigned long long *contrib_batch =
        reinterpret_cast<unsigned long long *>(s); // [block_size]
    int32_t *id_batch =
        reinterpret_cast<int32_t *>(&contrib_batch[block_size]); // [block_size]
    vec3 *xy_opacity_batch =
        reinterpret_cast<vec3 *>(&id_batch[block_size]); // [block_size]
    vec3 *conic_batch =
        reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]); // [block_size]
    float *depth_batch =
        reinterpret_cast<float *>(&conic_batch[block_size]); // [block_size]
    unsigned int *surface_dist_batch =
        reinterpret_cast<unsigned int *>(&depth_batch[block_size]); // [block_size]

    const unsigned int dist_init = __float_as_uint(FLT_MAX);

    float T = 1.0f;
    float expected_depth = 0.0f;
    float pix_out[CDIM] = {0.f};

    uint32_t tr = block.thread_rank();

    for (uint32_t b = 0; b < num_batches; ++b) {
        // this is also the barrier that protects the previous batch's shared
        // arrays from being overwritten
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < range_end) {
            int32_t g = flatten_ids[idx]; // flatten index in [C * N]
            id_batch[tr] = g;
            const vec2 xy = means2d[g];
            const float opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
            depth_batch[tr] = depths[g];
            contrib_batch[tr] = 0ull;
            surface_dist_batch[tr] = dist_init;
        }

        block.sync();

        uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const vec3 conic = conic_batch[t];
            const vec3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;
            const vec2 delta = {xy_opac.x - px, xy_opac.y - py};
            const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                conic.y * delta.x * delta.y;
            float alpha = min(CONTRIB_ALPHA_CAP, opac * __expf(-sigma));
            if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                continue;
            }

            const float next_T = T * (1.0f - alpha);
            if (next_T < CONTRIB_T_EPS) {
                done = true;
                break;
            }

            int32_t g = id_batch[t];
            const float vis = alpha * T;
            const float *c_ptr = colors + g * CDIM;
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                pix_out[k] += c_ptr[k] * vis;
            }
            expected_depth += depth_batch[t] * vis;

            atomicMax(&contrib_batch[t], pack_contrib(vis, (uint32_t)pix_id));

            T = next_T;
        }

        block.sync();

        if (idx < range_end) {
            const unsigned long long word = contrib_batch[tr];
            if (word != 0ull) {
                atomicMax(&max_contrib_packed[id_batch[tr]], word);
            }
        }

        if (calculate_surface_distance) {
            // every Gaussian in the batch is scored against the depth this
            // pixel has accumulated so far, whether or not it contributed
            if (inside && !masked) {
                for (uint32_t t = 0; t < batch_size; ++t) {
                    const float dist =
                        fabsf(depth_batch[t] - expected_depth);
                    atomicMin(
                        &surface_dist_batch[t], __float_as_uint(dist)
                    );
                }
            }

            block.sync();

            if (idx < range_end) {
                const unsigned int dist = surface_dist_batch[tr];
                if (dist != dist_init) {
                    atomicMin(
                        reinterpret_cast<unsigned int *>(min_surface_dist) +
                            id_batch[tr],
                        dist
                    );
                }
            }
        }
    }

    if (inside) {
        if (masked) {
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                render_colors[pix_id * CDIM + k] = 0.f;
            }
            render_depths[pix_id] = 0.f;
        } else {
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                render_colors[pix_id * CDIM + k] =
                    backgrounds == nullptr ? pix_out[k]
                                           : (pix_out[k] + T * backgrounds[k]);
            }
            render_depths[pix_id] = expected_depth;
        }
    }
}

__global__ void unpack_contrib_kernel(
    const uint32_t n,
    const unsigned long long *__restrict__ packed,
    float *__restrict__ max_contrib,
    int32_t *__restrict__ max_pixel
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    const unsigned long long word = packed[idx];
    if (word == 0ull) {
        // never voted for; upstream leaves both buffers at their zero init
        max_contrib[idx] = 0.f;
        max_pixel[idx] = 0;
        return;
    }
    max_contrib[idx] = __uint_as_float((uint32_t)(word >> 32));
    max_pixel[idx] =
        (int32_t)(~(uint32_t)(word & 0xFFFFFFFFull));
}

template <uint32_t CDIM>
void launch_rasterize_to_pixels_3dgs_contrib_fwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,   // [C, N, 2]
    const at::Tensor conics,    // [C, N, 3]
    const at::Tensor colors,    // [C, N, channels]
    const at::Tensor opacities, // [C, N]
    const at::Tensor depths,    // [C, N]
    const at::optional<at::Tensor> backgrounds,  // [C, channels]
    const at::optional<at::Tensor> pixel_masks,  // [C, image_height, image_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [C, tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    const bool calculate_surface_distance,
    // scratch: zero-initialised [C, N] int64, contribution + pixel id packed
    at::Tensor max_contrib_packed,
    // outputs
    at::Tensor renders,          // [C, image_height, image_width, channels]
    at::Tensor expected_depths,  // [C, image_height, image_width]
    at::Tensor max_contrib,      // [C, N]
    at::Tensor max_pixel,        // [C, N]
    at::Tensor min_surface_dist  // [C, N]
) {
    uint32_t C = tile_offsets.size(0);
    uint32_t N = means2d.size(1);
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);
    uint32_t n_isects = flatten_ids.size(0);

    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {C, tile_height, tile_width};

    int64_t block_size = tile_size * tile_size;
    int64_t shmem_size =
        block_size * (sizeof(unsigned long long) + sizeof(int32_t) +
                      2 * sizeof(vec3) + sizeof(float) + sizeof(unsigned int));

    if (cudaFuncSetAttribute(
            rasterize_to_pixels_3dgs_contrib_fwd_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }

    auto stream = at::cuda::getCurrentCUDAStream();

    rasterize_to_pixels_3dgs_contrib_fwd_kernel<CDIM, float>
        <<<grid, threads, shmem_size, stream>>>(
            C,
            N,
            n_isects,
            reinterpret_cast<vec2 *>(means2d.data_ptr<float>()),
            reinterpret_cast<vec3 *>(conics.data_ptr<float>()),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            depths.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                    : nullptr,
            pixel_masks.has_value() ? pixel_masks.value().data_ptr<bool>()
                                    : nullptr,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            expected_depths.data_ptr<float>(),
            reinterpret_cast<unsigned long long *>(max_contrib_packed.data_ptr<int64_t>()),
            min_surface_dist.data_ptr<float>(),
            calculate_surface_distance
        );

    uint32_t n_gauss = C * N;
    if (n_gauss > 0) {
        unpack_contrib_kernel<<<(n_gauss + 255) / 256, 256, 0, stream>>>(
            n_gauss,
            reinterpret_cast<unsigned long long *>(max_contrib_packed.data_ptr<int64_t>()),
            max_contrib.data_ptr<float>(),
            max_pixel.data_ptr<int32_t>()
        );
    }
}

// Explicit Instantiation: this should match how it is being called in .cpp
// file.
#define __INS__(CDIM)                                                          \
    template void launch_rasterize_to_pixels_3dgs_contrib_fwd_kernel<CDIM>(    \
        const at::Tensor means2d,                                              \
        const at::Tensor conics,                                               \
        const at::Tensor colors,                                               \
        const at::Tensor opacities,                                            \
        const at::Tensor depths,                                               \
        const at::optional<at::Tensor> backgrounds,                            \
        const at::optional<at::Tensor> pixel_masks,                            \
        uint32_t image_width,                                                  \
        uint32_t image_height,                                                 \
        uint32_t tile_size,                                                    \
        const at::Tensor tile_offsets,                                         \
        const at::Tensor flatten_ids,                                          \
        bool calculate_surface_distance,                                       \
        at::Tensor max_contrib_packed,                                         \
        at::Tensor renders,                                                    \
        at::Tensor expected_depths,                                            \
        at::Tensor max_contrib,                                                \
        at::Tensor max_pixel,                                                  \
        at::Tensor min_surface_dist                                            \
    );

__INS__(1)
__INS__(2)
__INS__(3)
__INS__(4)
__INS__(5)
__INS__(8)
__INS__(9)
__INS__(16)
__INS__(17)
__INS__(32)
__INS__(33)
#undef __INS__

} // namespace gsplat
