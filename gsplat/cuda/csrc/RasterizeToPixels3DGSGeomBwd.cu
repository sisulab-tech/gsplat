#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Rasterization.h"
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////
// Backward for rasterize_to_pixels_3dgs_geom_fwd.
//
// On top of the standard color/alpha chains this adds, walking
// back-to-front with T recovered progressively (T_k pre-composite,
// w_k = alpha_k * T_k):
//
// Expected depth E = sum_k w_k tz_k:
//   dE/dalpha_k = tz_k T_k - buffer_e / (1 - alpha_k)
//   dE/dtz_k    = w_k
//
// Median depth M = tz at the saved median intersection index; the
// selection carries no gradient, so only dM/dtz_median = 1 (RaDe-GS
// render_backward.cu semantics: no alpha weighting, no alpha chain).
//
// Distortion D = sum_k w_k e_k with e_k = m_k^2 A_k + S2_k - 2 m_k S1_k
// (A/S1/S2 prefix sums of w, w*m, w*m^2 over j < k) is the squared
// pairwise form sum_{j<k} w_k w_j (m_k - m_j)^2. With suffix sums
// P0/P1/P2 (over i > k) and the saved totals W = 1 - T_final,
// D1 = dist1, D2 = dist2:
//   A_k  = W  - w_k       - P0
//   S1_k = D1 - w_k m_k   - P1
//   S2_k = D2 - w_k m_k^2 - P2
//   dD/dw_k (m fixed) = e_k + (P2 - 2 m_k P1 + m_k^2 P0)
//   dD/dalpha_k = dl_dw_k T_k - buffer_d / (1 - alpha_k)
//   dD/dm_k = 2 w_k [(m_k A_k - S1_k) + (m_k P0 - P1)]
//
// tz = rln * (rp . (delta, 1)) chains into v_ray_planes and an extra
// v_means2d term (via delta = xy - pix); these do NOT pass through the
// alpha saturation gate (they are independent of alpha).
////////////////////////////////////////////////////////////////

template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_3dgs_geom_bwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    // fwd inputs
    const vec2 *__restrict__ means2d,         // [C, N, 2] or [nnz, 2]
    const vec3 *__restrict__ conics,          // [C, N, 3] or [nnz, 3]
    const scalar_t *__restrict__ colors,      // [C, N, CDIM] or [nnz, CDIM]
    const scalar_t *__restrict__ opacities,   // [C, N] or [nnz]
    const vec3 *__restrict__ ray_planes,      // [C, N, 3] or [nnz, 3]
    const scalar_t *__restrict__ Ks,          // [C, 9]
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
    // fwd outputs
    const scalar_t
        *__restrict__ render_alphas,      // [C, image_height, image_width, 1]
    const int32_t *__restrict__ last_ids, // [C, image_height, image_width]
    const int32_t *__restrict__ median_ids, // [C, image_height, image_width]
    const scalar_t *__restrict__ dist_accums, // [C, image_height,
                                              // image_width, 2]
    // grad outputs
    const scalar_t *__restrict__ v_render_colors, // [C, image_height,
                                                  // image_width, CDIM]
    const scalar_t
        *__restrict__ v_render_alphas, // [C, image_height, image_width, 1]
    const scalar_t
        *__restrict__ v_render_edepths, // [C, image_height, image_width, 1]
    const scalar_t
        *__restrict__ v_render_mdepths, // [C, image_height, image_width, 1]
    const scalar_t
        *__restrict__ v_render_distorts, // [C, image_height, image_width, 1]
    // grad inputs
    vec2 *__restrict__ v_means2d_abs,  // [C, N, 2] or [nnz, 2]
    vec2 *__restrict__ v_means2d,      // [C, N, 2] or [nnz, 2]
    vec3 *__restrict__ v_conics,       // [C, N, 3] or [nnz, 3]
    scalar_t *__restrict__ v_colors,   // [C, N, CDIM] or [nnz, CDIM]
    scalar_t *__restrict__ v_opacities, // [C, N] or [nnz]
    vec3 *__restrict__ v_ray_planes    // [C, N, 3] or [nnz, 3]
) {
    auto block = cg::this_thread_block();
    uint32_t camera_id = block.group_index().x;
    uint32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += camera_id * tile_height * tile_width;
    render_alphas += camera_id * image_height * image_width;
    last_ids += camera_id * image_height * image_width;
    median_ids += camera_id * image_height * image_width;
    dist_accums += camera_id * image_height * image_width * 2;
    v_render_colors += camera_id * image_height * image_width * CDIM;
    v_render_alphas += camera_id * image_height * image_width;
    v_render_edepths += camera_id * image_height * image_width;
    v_render_mdepths += camera_id * image_height * image_width;
    v_render_distorts += camera_id * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += camera_id * CDIM;
    }
    if (masks != nullptr) {
        masks += camera_id * tile_height * tile_width;
    }

    if (masks != nullptr && !masks[tile_id]) {
        return;
    }

    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    const int32_t pix_id =
        min(i * image_width + j, image_width * image_height - 1);

    const float fx = Ks[camera_id * 9 + 0];
    const float fy = Ks[camera_id * 9 + 4];
    const float cx = Ks[camera_id * 9 + 2];
    const float cy = Ks[camera_id * 9 + 5];
    const float rln = rnorm3df((px - cx) / fx, (py - cy) / fy, 1.f);

    bool inside = (i < image_height && j < image_width);

    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]
    vec3 *xy_opacity_batch =
        reinterpret_cast<vec3 *>(&id_batch[block_size]); // [block_size]
    vec3 *conic_batch =
        reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]); // [block_size]
    vec3 *ray_plane_batch =
        reinterpret_cast<vec3 *>(&conic_batch[block_size]); // [block_size]
    float *rgbs_batch =
        (float *)&ray_plane_batch[block_size]; // [block_size * CDIM]

    // this is the T AFTER the last gaussian in this pixel
    float T_final = 1.0f - render_alphas[pix_id];
    float T = T_final;
    // the contribution from gaussians behind the current one
    float buffer[CDIM] = {0.f};
    float buffer_e = 0.f; // sum_{i>k} w_i tz_i
    float buffer_d = 0.f; // sum_{i>k} dl_dw_i w_i
    // suffix sums for the distortion gradient
    float P0 = 0.f, P1 = 0.f, P2 = 0.f;
    const float W_tot = render_alphas[pix_id];
    const float D1_tot = dist_accums[pix_id * 2];
    const float D2_tot = dist_accums[pix_id * 2 + 1];

    const int32_t bin_final = inside ? last_ids[pix_id] : 0;
    const int32_t median_idx = inside ? median_ids[pix_id] : -1;

    float v_render_c[CDIM];
#pragma unroll
    for (uint32_t k = 0; k < CDIM; ++k) {
        v_render_c[k] = v_render_colors[pix_id * CDIM + k];
    }
    const float v_render_a = v_render_alphas[pix_id];
    const float v_edepth = v_render_edepths[pix_id];
    const float v_mdepth = v_render_mdepths[pix_id];
    const float v_distort = v_render_distorts[pix_id];

    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final =
        cg::reduce(warp, bin_final, cg::greater<int>());
    for (uint32_t b = 0; b < num_batches; ++b) {
        block.sync();

        const int32_t batch_end = range_end - 1 - block_size * b;
        const int32_t batch_size = min(block_size, batch_end + 1 - range_start);
        const int32_t idx = batch_end - tr;
        if (idx >= range_start) {
            int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            const vec2 xy = means2d[g];
            const float opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
            ray_plane_batch[tr] = ray_planes[g];
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                rgbs_batch[tr * CDIM + k] = colors[g * CDIM + k];
            }
        }
        block.sync();
        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size;
             ++t) {
            bool valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            float alpha;
            float opac;
            vec2 delta;
            vec3 conic;
            vec3 rp;
            float vis;
            float tz;

            if (valid) {
                conic = conic_batch[t];
                vec3 xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;
                delta = {xy_opac.x - px, xy_opac.y - py};
                float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                      conic.z * delta.y * delta.y) +
                              conic.y * delta.x * delta.y;
                vis = __expf(-sigma);
                alpha = min(0.999f, opac * vis);
                if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                    valid = false;
                }
                rp = ray_plane_batch[t];
                // mirror the forward's GOF near-plane skip
                // (backward.cu:790): such Gaussians contributed nothing
                tz = (rp.x * delta.x + rp.y * delta.y + rp.z) * rln;
                if (tz <= distort_near) {
                    valid = false;
                }
            }

            if (!warp.any(valid)) {
                continue;
            }
            float v_rgb_local[CDIM] = {0.f};
            vec3 v_conic_local = {0.f, 0.f, 0.f};
            vec2 v_xy_local = {0.f, 0.f};
            vec2 v_xy_abs_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            vec3 v_ray_plane_local = {0.f, 0.f, 0.f};
            if (valid) {
                float ra = 1.0f / (1.0f - alpha);
                T *= ra; // now T_k, the pre-composite transmittance
                const float fac = alpha * T; // w_k
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    v_rgb_local[k] = fac * v_render_c[k];
                }
                float v_alpha = 0.f;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    v_alpha += (rgbs_batch[t * CDIM + k] * T - buffer[k] * ra) *
                               v_render_c[k];
                }

                v_alpha += T_final * ra * v_render_a;
                if (backgrounds != nullptr) {
                    float accum = 0.f;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        accum += backgrounds[k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }

                // ---- geometry chains ----
                // expected depth
                v_alpha += (tz * T - buffer_e * ra) * v_edepth;
                float v_tz = fac * v_edepth;

                // median depth: bare value gradient, no alpha chain
                if (batch_end - t == median_idx) {
                    v_tz += v_mdepth;
                }

                // distortion (tz > distort_near past the validity gate)
                const float m = distort_far * (tz - distort_near) /
                                ((distort_far - distort_near) * tz);
                const float A_k = W_tot - fac - P0;
                const float S1_k = D1_tot - fac * m - P1;
                const float S2_k = D2_tot - fac * m * m - P2;
                const float e_k = m * m * A_k + S2_k - 2.0f * m * S1_k;
                const float dl_dw = e_k + (P2 - 2.0f * m * P1 + m * m * P0);
                v_alpha += (dl_dw * T - buffer_d * ra) * v_distort;
                const float v_m =
                    2.0f * fac * ((m * A_k - S1_k) + (m * P0 - P1)) *
                    v_distort;
                // dm/dtz = far * near / ((far - near) * tz^2)
                v_tz += v_m * distort_far * distort_near /
                        ((distort_far - distort_near) * tz * tz);

                // tz -> ray plane / means2d (independent of the alpha
                // saturation gate)
                const float v_t = v_tz * rln;
                v_ray_plane_local = {v_t * delta.x, v_t * delta.y, v_t};
                v_xy_local.x += v_t * rp.x;
                v_xy_local.y += v_t * rp.y;

                if (opac * vis <= 0.999f) {
                    const float v_sigma = -opac * vis * v_alpha;
                    v_conic_local = {
                        0.5f * v_sigma * delta.x * delta.x,
                        v_sigma * delta.x * delta.y,
                        0.5f * v_sigma * delta.y * delta.y
                    };
                    v_xy_local.x +=
                        v_sigma * (conic.x * delta.x + conic.y * delta.y);
                    v_xy_local.y +=
                        v_sigma * (conic.y * delta.x + conic.z * delta.y);
                    v_opacity_local = vis * v_alpha;
                }
                if (v_means2d_abs != nullptr) {
                    v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
                }

                // suffix updates
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    buffer[k] += rgbs_batch[t * CDIM + k] * fac;
                }
                buffer_e += tz * fac;
                buffer_d += dl_dw * fac;
                P0 += fac;
                P1 += fac * m;
                P2 += fac * m * m;
            }
            warpSum<CDIM>(v_rgb_local, warp);
            warpSum(v_conic_local, warp);
            warpSum(v_xy_local, warp);
            warpSum(v_ray_plane_local, warp);
            if (v_means2d_abs != nullptr) {
                warpSum(v_xy_abs_local, warp);
            }
            warpSum(v_opacity_local, warp);
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t];
                float *v_rgb_ptr = (float *)(v_colors) + CDIM * g;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                float *v_conic_ptr = (float *)(v_conics) + 3 * g;
                gpuAtomicAdd(v_conic_ptr, v_conic_local.x);
                gpuAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
                gpuAtomicAdd(v_conic_ptr + 2, v_conic_local.z);

                float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);

                if (v_means2d_abs != nullptr) {
                    float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                    gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                    gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                }

                gpuAtomicAdd(v_opacities + g, v_opacity_local);

                float *v_rp_ptr = (float *)(v_ray_planes) + 3 * g;
                gpuAtomicAdd(v_rp_ptr, v_ray_plane_local.x);
                gpuAtomicAdd(v_rp_ptr + 1, v_ray_plane_local.y);
                gpuAtomicAdd(v_rp_ptr + 2, v_ray_plane_local.z);
            }
        }
    }
}

template <uint32_t CDIM>
void launch_rasterize_to_pixels_3dgs_geom_bwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,    // [C, N, 2] or [nnz, 2]
    const at::Tensor conics,     // [C, N, 3] or [nnz, 3]
    const at::Tensor colors,     // [C, N, CDIM] or [nnz, CDIM]
    const at::Tensor opacities,  // [C, N] or [nnz]
    const at::Tensor ray_planes, // [C, N, 3] or [nnz, 3]
    const at::Tensor Ks,         // [C, 3, 3]
    const float distort_near,
    const float distort_far,
    const at::optional<at::Tensor> backgrounds, // [C, CDIM]
    const at::optional<at::Tensor> masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [C, tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor render_alphas, // [C, image_height, image_width, 1]
    const at::Tensor last_ids,      // [C, image_height, image_width]
    const at::Tensor median_ids,    // [C, image_height, image_width]
    const at::Tensor dist_accums,   // [C, image_height, image_width, 2]
    // gradients of outputs
    const at::Tensor v_render_colors,  // [C, image_height, image_width, CDIM]
    const at::Tensor v_render_alphas,  // [C, image_height, image_width, 1]
    const at::Tensor v_render_edepths, // [C, image_height, image_width, 1]
    const at::Tensor v_render_mdepths, // [C, image_height, image_width, 1]
    const at::Tensor v_render_distorts, // [C, image_height, image_width, 1]
    // outputs
    at::optional<at::Tensor> v_means2d_abs, // [C, N, 2] or [nnz, 2]
    at::Tensor v_means2d,                   // [C, N, 2] or [nnz, 2]
    at::Tensor v_conics,                    // [C, N, 3] or [nnz, 3]
    at::Tensor v_colors,                    // [C, N, CDIM] or [nnz, CDIM]
    at::Tensor v_opacities,                 // [C, N] or [nnz]
    at::Tensor v_ray_planes                 // [C, N, 3] or [nnz, 3]
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
        (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(vec3) +
         sizeof(float) * CDIM);

    if (n_isects == 0) {
        return;
    }

    if (cudaFuncSetAttribute(
            rasterize_to_pixels_3dgs_geom_bwd_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }

    rasterize_to_pixels_3dgs_geom_bwd_kernel<CDIM, float>
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
            render_alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            median_ids.data_ptr<int32_t>(),
            dist_accums.data_ptr<float>(),
            v_render_colors.data_ptr<float>(),
            v_render_alphas.data_ptr<float>(),
            v_render_edepths.data_ptr<float>(),
            v_render_mdepths.data_ptr<float>(),
            v_render_distorts.data_ptr<float>(),
            v_means2d_abs.has_value()
                ? reinterpret_cast<vec2 *>(
                      v_means2d_abs.value().data_ptr<float>()
                  )
                : nullptr,
            reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),
            reinterpret_cast<vec3 *>(v_conics.data_ptr<float>()),
            v_colors.data_ptr<float>(),
            v_opacities.data_ptr<float>(),
            reinterpret_cast<vec3 *>(v_ray_planes.data_ptr<float>())
        );
}

#define __INS__(CDIM)                                                         \
    template void launch_rasterize_to_pixels_3dgs_geom_bwd_kernel<CDIM>(     \
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
        const at::Tensor render_alphas,                                       \
        const at::Tensor last_ids,                                            \
        const at::Tensor median_ids,                                          \
        const at::Tensor dist_accums,                                         \
        const at::Tensor v_render_colors,                                     \
        const at::Tensor v_render_alphas,                                     \
        const at::Tensor v_render_edepths,                                    \
        const at::Tensor v_render_mdepths,                                    \
        const at::Tensor v_render_distorts,                                   \
        at::optional<at::Tensor> v_means2d_abs,                               \
        at::Tensor v_means2d,                                                 \
        at::Tensor v_conics,                                                  \
        at::Tensor v_colors,                                                  \
        at::Tensor v_opacities,                                               \
        at::Tensor v_ray_planes                                               \
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
