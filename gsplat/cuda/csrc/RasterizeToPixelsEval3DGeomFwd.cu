#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Rasterization.h"

namespace gsplat {

namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////
// Forward with GOF eval3d geometry outputs (arXiv:2404.10772).
//
// Port of GOF's training rasterizer (gaussian-opacity-fields
// forward.cu:409 renderCUDA): color AND geometry are composited with
// the exact 3D Gaussian response along the camera-space pixel ray
// p(t) = t * r, r = ((px - cx) / fx, (py - cy) / fy, 1). Per Gaussian
// the squared Mahalanobis distance along the ray is the quadric
//
//   f(t) = AA t^2 + BB t + CC
//
// built from the precomputed 10-float view2gaussian packing
// [M00, M01, M02, M11, M12, M22, b0, b1, b2, c] (see
// compute_view2gaussians): with u = M r,
// AA = r^T u, BB = 2 b . r, CC = c. Then per (gaussian, pixel):
//  - alpha = min(0.999, opac * exp(-0.5 * max(0, CC - BB^2 / (4 AA))))
//    (the peak 3D response; no 2D projection involved);
//  - depth t* = -BB / (2 AA), a z-depth directly (the ray has unit z);
//  - normal = -u / sqrt(|u|^2 + 1e-7), per PIXEL (GOF evaluates M at
//    the ray direction, not at the optimum), composited like a color;
//  - GOF near-plane skip: t* <= distort_near contributes nothing and
//    leaves T untouched (forward.cu:518);
//  - expected/median depth and the GOF NDC distortion accumulator are
//    identical to the RaDe-GS 2D path (RasterizeToPixels3DGSGeomFwd).
//
// AA / BB / min_value use double precision like GOF (the CC -
// BB^2/(4AA) cancellation is catastrophic in float32 for distant
// Gaussians). gsplat conventions where GOF differs: true principal
// point (GOF hardcodes W/2, H/2), alpha clamp 0.999 (GOF: 0.99),
// exclusive termination at next_T <= 1e-4.
//
// Unlike the 2D geometry kernel, normals canNOT ride the color
// channels (they are per-pixel), so this kernel has a dedicated
// normal accumulator.
////////////////////////////////////////////////////////////////

template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_eval3d_geom_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const scalar_t *__restrict__ view2gaussians, // [C, N, 10]
    const scalar_t *__restrict__ colors,         // [C, N, CDIM]
    const scalar_t *__restrict__ opacities,      // [C, N]
    const scalar_t *__restrict__ Ks,             // [C, 9] row-major intrinsics
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
    scalar_t *__restrict__ render_normals, // [C, image_height, image_width, 3]
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
    render_normals += camera_id * image_height * image_width * 3;
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

    // camera-space pixel ray direction (unit z)
    const float fx = Ks[camera_id * 9 + 0];
    const float fy = Ks[camera_id * 9 + 4];
    const float cx = Ks[camera_id * 9 + 2];
    const float cy = Ks[camera_id * 9 + 5];
    const float rx = (px - cx) / fx;
    const float ry = (py - cy) / fy;

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
#pragma unroll
        for (uint32_t k = 0; k < 3; ++k) {
            render_normals[pix_id * 3 + k] = 0.f;
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
    int32_t *id_batch = (int32_t *)s;               // [block_size]
    float *opac_batch = (float *)&id_batch[block_size]; // [block_size]
    float *v2g_batch = (float *)&opac_batch[block_size]; // [block_size * 10]

    float T = 1.0f;
    uint32_t cur_idx = 0;

    // geometry accumulators
    float nrm_out[3] = {0.f};
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
            int32_t g = flatten_ids[idx]; // flatten index in [C * N]
            id_batch[tr] = g;
            opac_batch[tr] = opacities[g];
#pragma unroll
            for (uint32_t k = 0; k < 10; ++k) {
                v2g_batch[tr * 10 + k] = view2gaussians[g * 10 + k];
            }
        }

        block.sync();

        uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const float *v2g = v2g_batch + t * 10;
            const float opac = opac_batch[t];

            // u = M r; AA = r^T M r, BB = 2 b . r, CC = c
            const float u0 = v2g[0] * rx + v2g[1] * ry + v2g[2];
            const float u1 = v2g[1] * rx + v2g[3] * ry + v2g[4];
            const float u2 = v2g[2] * rx + v2g[4] * ry + v2g[5];
            const double AA = (double)rx * u0 + (double)ry * u1 + u2;
            const double BB =
                2.0 * ((double)v2g[6] * rx + (double)v2g[7] * ry + v2g[8]);
            const float CC = v2g[9];

            // exact intersection depth: z-depth directly (unit-z ray).
            // GOF skips at or below the near plane before the alpha test
            // (forward.cu:518) -- no contribution, T untouched.
            const float tz = (float)(-BB / (2.0 * AA));
            if (tz <= distort_near) {
                continue;
            }

            // peak 3D response; GOF clamps power at 0 (numerics only)
            const double min_value = -(BB / AA) * (BB / 4.0) + CC;
            float power = -0.5f * (float)min_value;
            if (power > 0.0f) {
                power = 0.0f;
            }
            const float vis = __expf(power);
            float alpha = min(0.999f, opac * vis);
            if (alpha < ALPHA_THRESHOLD) {
                continue;
            }

            const float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) { // this pixel is done: exclusive
                done = true;
                break;
            }

            int32_t g = id_batch[t];
            const float w = alpha * T;
            const float *c_ptr = colors + g * CDIM;
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                pix_out[k] += c_ptr[k] * w;
            }

            // per-pixel normal (GOF backward.cu:813 epsilon convention)
            const float len =
                sqrtf(u0 * u0 + u1 * u1 + u2 * u2 + 1e-7f);
            nrm_out[0] += -u0 / len * w;
            nrm_out[1] += -u1 / len * w;
            nrm_out[2] += -u2 / len * w;

            edepth += tz * w;
            if (T > 0.5f) {
                mdepth = tz;
                median_idx = static_cast<int32_t>(batch_start + t);
            }

            // GOF distortion on the NDC-mapped depth (tz > distort_near here)
            const float m = distort_far * (tz - distort_near) /
                            ((distort_far - distort_near) * tz);
            const float A_acc = 1.0f - T;
            distort += (m * m * A_acc + dist2 - 2.0f * m * dist1) * w;
            dist1 += m * w;
            dist2 += m * m * w;

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
#pragma unroll
        for (uint32_t k = 0; k < 3; ++k) {
            render_normals[pix_id * 3 + k] = nrm_out[k];
        }
        render_edepths[pix_id] = edepth;
        render_mdepths[pix_id] = mdepth;
        median_ids[pix_id] = median_idx;
        render_distorts[pix_id] = distort;
        dist_accums[pix_id * 2] = dist1;
        dist_accums[pix_id * 2 + 1] = dist2;
    }
}

template <uint32_t CDIM>
void launch_rasterize_to_pixels_eval3d_geom_fwd_kernel(
    // Gaussian parameters
    const at::Tensor view2gaussians, // [C, N, 10]
    const at::Tensor colors,         // [C, N, channels]
    const at::Tensor opacities,      // [C, N]
    const at::Tensor Ks,             // [C, 3, 3]
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
    at::Tensor render_normals, // [C, image_height, image_width, 3]
    at::Tensor render_edepths, // [C, image_height, image_width, 1]
    at::Tensor render_mdepths, // [C, image_height, image_width, 1]
    at::Tensor median_ids,     // [C, image_height, image_width]
    at::Tensor render_distorts, // [C, image_height, image_width, 1]
    at::Tensor dist_accums      // [C, image_height, image_width, 2]
) {
    uint32_t C = tile_offsets.size(0);
    uint32_t N = view2gaussians.size(1);
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);
    uint32_t n_isects = flatten_ids.size(0);

    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {C, tile_height, tile_width};

    int64_t shmem_size =
        tile_size * tile_size *
        (sizeof(int32_t) + sizeof(float) + sizeof(float) * 10);

    if (cudaFuncSetAttribute(
            rasterize_to_pixels_eval3d_geom_fwd_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }

    rasterize_to_pixels_eval3d_geom_fwd_kernel<CDIM, float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            C,
            N,
            n_isects,
            view2gaussians.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
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
            render_normals.data_ptr<float>(),
            render_edepths.data_ptr<float>(),
            render_mdepths.data_ptr<float>(),
            median_ids.data_ptr<int32_t>(),
            render_distorts.data_ptr<float>(),
            dist_accums.data_ptr<float>()
        );
}

// Explicit Instantiation: this should match how it is being called in .cpp
// file. Normals have a dedicated accumulator here, so CDIM is just the
// color/feature channel count.
#define __INS__(CDIM)                                                         \
    template void launch_rasterize_to_pixels_eval3d_geom_fwd_kernel<CDIM>(   \
        const at::Tensor view2gaussians,                                      \
        const at::Tensor colors,                                              \
        const at::Tensor opacities,                                           \
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
        at::Tensor render_normals,                                            \
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
