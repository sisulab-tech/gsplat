#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Rasterization.h"

namespace gsplat {

namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////
// Forward with RaDe-GS geometry outputs (arXiv:2406.01467).
//
// In addition to the standard color/alpha compositing this kernel emits,
// per pixel:
//  - expected plane depth: alpha-blended closed-form ray-Gaussian
//    intersection z-depth, accumulated raw (sum t_z * w, NOT normalized
//    by alpha). The per-Gaussian linearization t = rp.x * dx + rp.y * dy
//    + rp.z (an intersection *distance*) is precomputed on the Python
//    side (compute_ray_planes); the per-pixel factor rln converts
//    distance to z-depth.
//  - median plane depth: t_z of the last composited Gaussian whose
//    pre-composite transmittance exceeds 0.5 (RaDe-GS), plus its
//    intersection index for the backward pass.
//  - distortion: GOF's squared-pairwise accumulator on the 2DGS NDC
//    depth mapping m = far * (t_z - near) / ((far - near) * t_z)
//    (gaussian-opacity-fields forward.cu:543-557), emitted raw (before
//    the /(1-T)^2 normalization, done in Python). dist1/dist2 running
//    totals are saved for the backward pass.
//
// Normals are NOT handled here: they blend exactly like color channels,
// so the Python wrapper appends them to `colors`.
////////////////////////////////////////////////////////////////

template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_3dgs_geom_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    const vec2 *__restrict__ means2d,         // [C, N, 2] or [nnz, 2]
    const vec3 *__restrict__ conics,          // [C, N, 3] or [nnz, 3]
    const scalar_t *__restrict__ colors,      // [C, N, CDIM] or [nnz, CDIM]
    const scalar_t *__restrict__ opacities,   // [C, N] or [nnz]
    const vec3 *__restrict__ ray_planes,      // [C, N, 3] or [nnz, 3]
    const scalar_t *__restrict__ Ks,          // [C, 9] row-major intrinsics
    const float distort_near,
    const float distort_far,
    const scalar_t *__restrict__ backgrounds, // [C, CDIM]
    const bool *__restrict__ masks,           // [C, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    scalar_t
        *__restrict__ render_colors, // [C, image_height, image_width, CDIM]
    scalar_t *__restrict__ render_alphas,  // [C, image_height, image_width, 1]
    int32_t *__restrict__ last_ids,        // [C, image_height, image_width]
    scalar_t *__restrict__ render_edepths, // [C, image_height, image_width, 1]
    scalar_t *__restrict__ render_mdepths, // [C, image_height, image_width, 1]
    int32_t *__restrict__ median_ids,      // [C, image_height, image_width]
    scalar_t
        *__restrict__ render_distorts, // [C, image_height, image_width, 1]
    scalar_t *__restrict__ dist_accums // [C, image_height, image_width, 2]
) {
    auto block = cg::this_thread_block();
    int32_t camera_id = block.group_index().x;
    int32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += camera_id * tile_height * tile_width;
    render_colors += camera_id * image_height * image_width * CDIM;
    render_alphas += camera_id * image_height * image_width;
    last_ids += camera_id * image_height * image_width;
    render_edepths += camera_id * image_height * image_width;
    render_mdepths += camera_id * image_height * image_width;
    median_ids += camera_id * image_height * image_width;
    render_distorts += camera_id * image_height * image_width;
    dist_accums += camera_id * image_height * image_width * 2;
    if (backgrounds != nullptr) {
        backgrounds += camera_id * CDIM;
    }
    if (masks != nullptr) {
        masks += camera_id * tile_height * tile_width;
    }

    float px = (float)j + 0.5f;
    float py = (float)i + 0.5f;
    int32_t pix_id = i * image_width + j;

    // ray-distance -> z-depth factor for this pixel
    const float fx = Ks[camera_id * 9 + 0];
    const float fy = Ks[camera_id * 9 + 4];
    const float cx = Ks[camera_id * 9 + 2];
    const float cy = Ks[camera_id * 9 + 5];
    const float rln = rnorm3df((px - cx) / fx, (py - cy) / fy, 1.f);

    bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    // when the mask is provided, render the background color and return
    // if this tile is labeled as False
    if (masks != nullptr && inside && !masks[tile_id]) {
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            render_colors[pix_id * CDIM + k] =
                backgrounds == nullptr ? 0.0f : backgrounds[k];
        }
        render_edepths[pix_id] = 0.f;
        render_mdepths[pix_id] = 0.f;
        median_ids[pix_id] = -1;
        render_distorts[pix_id] = 0.f;
        dist_accums[pix_id * 2] = 0.f;
        dist_accums[pix_id * 2 + 1] = 0.f;
        return;
    }

    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]
    vec3 *xy_opacity_batch =
        reinterpret_cast<vec3 *>(&id_batch[block_size]); // [block_size]
    vec3 *conic_batch =
        reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]); // [block_size]
    vec3 *ray_plane_batch =
        reinterpret_cast<vec3 *>(&conic_batch[block_size]); // [block_size]

    float T = 1.0f;
    uint32_t cur_idx = 0;

    // geometry accumulators
    float edepth = 0.f;
    float mdepth = 0.f;
    int32_t median_idx = -1;
    float distort = 0.f;
    float dist1 = 0.f;
    float dist2 = 0.f;

    uint32_t tr = block.thread_rank();

    float pix_out[CDIM] = {0.f};
    for (uint32_t b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < range_end) {
            int32_t g = flatten_ids[idx]; // flatten index in [C * N] or [nnz]
            id_batch[tr] = g;
            const vec2 xy = means2d[g];
            const float opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
            ray_plane_batch[tr] = ray_planes[g];
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
            float alpha = min(0.999f, opac * __expf(-sigma));
            if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                continue;
            }

            // RaDe-GS plane depth: intersection distance from the
            // per-Gaussian linearization, converted to z-depth.
            const vec3 rp = ray_plane_batch[t];
            const float t_plane = rp.x * delta.x + rp.y * delta.y + rp.z;
            const float tz = t_plane * rln;
            // GOF skips Gaussians whose intersection depth is at or below
            // the near plane (forward.cu:518) -- they contribute to no
            // channel and leave T untouched. Without this, the NDC mapping
            // below diverges as tz -> 0.
            if (tz <= distort_near) {
                continue;
            }

            const float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) { // this pixel is done: exclusive
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

            edepth += tz * vis;
            if (T > 0.5f) {
                mdepth = tz;
                median_idx = static_cast<int32_t>(batch_start + t);
            }

            // GOF distortion on the NDC-mapped depth (tz > distort_near here)
            const float m = distort_far * (tz - distort_near) /
                            ((distort_far - distort_near) * tz);
            const float A_acc = 1.0f - T;
            distort += (m * m * A_acc + dist2 - 2.0f * m * dist1) * vis;
            dist1 += m * vis;
            dist2 += m * m * vis;

            cur_idx = batch_start + t;

            T = next_T;
        }
    }

    if (inside) {
        render_alphas[pix_id] = 1.0f - T;
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            render_colors[pix_id * CDIM + k] =
                backgrounds == nullptr ? pix_out[k]
                                       : (pix_out[k] + T * backgrounds[k]);
        }
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);
        render_edepths[pix_id] = edepth;
        render_mdepths[pix_id] = mdepth;
        median_ids[pix_id] = median_idx;
        render_distorts[pix_id] = distort;
        dist_accums[pix_id * 2] = dist1;
        dist_accums[pix_id * 2 + 1] = dist2;
    }
}

template <uint32_t CDIM>
void launch_rasterize_to_pixels_3dgs_geom_fwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,    // [C, N, 2] or [nnz, 2]
    const at::Tensor conics,     // [C, N, 3] or [nnz, 3]
    const at::Tensor colors,     // [C, N, channels] or [nnz, channels]
    const at::Tensor opacities,  // [C, N]  or [nnz]
    const at::Tensor ray_planes, // [C, N, 3] or [nnz, 3]
    const at::Tensor Ks,         // [C, 3, 3]
    const float distort_near,
    const float distort_far,
    const at::optional<at::Tensor> backgrounds, // [C, channels]
    const at::optional<at::Tensor> masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [C, tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // outputs
    at::Tensor renders, // [C, image_height, image_width, channels]
    at::Tensor alphas,  // [C, image_height, image_width]
    at::Tensor last_ids,
    at::Tensor render_edepths, // [C, image_height, image_width, 1]
    at::Tensor render_mdepths, // [C, image_height, image_width, 1]
    at::Tensor median_ids,     // [C, image_height, image_width]
    at::Tensor render_distorts, // [C, image_height, image_width, 1]
    at::Tensor dist_accums      // [C, image_height, image_width, 2]
) {
    bool packed = means2d.dim() == 2;

    uint32_t C = tile_offsets.size(0);
    uint32_t N = packed ? 0 : means2d.size(1);
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);
    uint32_t n_isects = flatten_ids.size(0);

    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {C, tile_height, tile_width};

    int64_t shmem_size =
        tile_size * tile_size *
        (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(vec3));

    if (cudaFuncSetAttribute(
            rasterize_to_pixels_3dgs_geom_fwd_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }

    rasterize_to_pixels_3dgs_geom_fwd_kernel<CDIM, float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            C,
            N,
            n_isects,
            packed,
            reinterpret_cast<vec2 *>(means2d.data_ptr<float>()),
            reinterpret_cast<vec3 *>(conics.data_ptr<float>()),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            reinterpret_cast<vec3 *>(ray_planes.data_ptr<float>()),
            Ks.data_ptr<float>(),
            distort_near,
            distort_far,
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                    : nullptr,
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            render_edepths.data_ptr<float>(),
            render_mdepths.data_ptr<float>(),
            median_ids.data_ptr<int32_t>(),
            render_distorts.data_ptr<float>(),
            dist_accums.data_ptr<float>()
        );
}

// Explicit Instantiation: this should match how it is being called in .cpp
// file. Geometry rendering carries RGB (3) + normals (3), padded by the
// Python side to a supported channel count; larger feature dims are
// supported up to 33.
#define __INS__(CDIM)                                                         \
    template void launch_rasterize_to_pixels_3dgs_geom_fwd_kernel<CDIM>(     \
        const at::Tensor means2d,                                             \
        const at::Tensor conics,                                              \
        const at::Tensor colors,                                              \
        const at::Tensor opacities,                                           \
        const at::Tensor ray_planes,                                          \
        const at::Tensor Ks,                                                  \
        const float distort_near,                                             \
        const float distort_far,                                              \
        const at::optional<at::Tensor> backgrounds,                           \
        const at::optional<at::Tensor> masks,                                 \
        uint32_t image_width,                                                 \
        uint32_t image_height,                                                \
        uint32_t tile_size,                                                   \
        const at::Tensor tile_offsets,                                        \
        const at::Tensor flatten_ids,                                         \
        at::Tensor renders,                                                   \
        at::Tensor alphas,                                                    \
        at::Tensor last_ids,                                                  \
        at::Tensor render_edepths,                                            \
        at::Tensor render_mdepths,                                            \
        at::Tensor median_ids,                                                \
        at::Tensor render_distorts,                                           \
        at::Tensor dist_accums                                                \
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
