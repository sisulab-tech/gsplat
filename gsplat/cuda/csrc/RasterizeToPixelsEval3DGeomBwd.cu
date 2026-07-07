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
// Backward for rasterize_to_pixels_eval3d_geom_fwd.
//
// Gradient reference: GOF backward.cu:636 renderCUDA. All parameter
// gradients flow through the 10-float view2gaussian packing (this op
// emits dL/dview2gaussians; the chain to means/quats/scales is torch
// autograd through compute_view2gaussians, exactly GOF's split into
// render backward + computeView2Gaussian_backward). With u = M r:
//
//   AA = r^T u, BB = 2 b . r, CC = c
//   tz = -BB / (2 AA):      dtz/dAA = BB / (2 AA^2), dtz/dBB = -1/(2 AA)
//   mv = CC - BB^2 / (4 AA) (zero grad when the forward clamped mv < 0):
//     dmv/dAA = (BB/AA)^2 / 4, dmv/dBB = -BB/(2 AA), dmv/dCC = 1
//   alpha = opac * exp(-0.5 mv), gated like the fork on
//     opac * vis <= 0.999 (the oracle's clamp_max);
//   normal n = -u / len, len = sqrt(|u|^2 + 1e-7) (GOF backward.cu:813):
//     dL/dlen = (dL/dn . u) / len^2, dL/du = (-dL/dn + dL/dlen u) / len
//   then AA = r . u folds into dL/du (GOF backward.cu:938), and
//   dL/dview2gaussians accumulates dL/du (x) r symmetrized plus the
//   b / c terms (GOF backward.cu:943-952).
//
// The normal compositing adds a color-like alpha chain (buffer_n);
// expected/median depth and the distortion suffix-sum chains are
// identical to RasterizeToPixels3DGSGeomBwd (note: the distortion
// keeps the FULL gradient through the compositing weights, where GOF
// detaches them -- backward.cu:852 "detach weight"; deviation
// validated by the Phase-2 exit gate and matching the torch oracle).
//
// means2d / conics are consumed ONLY for GOF's densification signal
// (backward.cu:896-909: dL_dmean2D from the 2D conic with the 3D
// response G -- "we don't need this for back propagation but it is
// useful for gaussian density mechanism"; GOF's preprocess backward
// never consumes it into parameters). Here it is emitted in gsplat
// conventions: pixel units (no NDC 0.5*W factor), component-wise
// v_means2d / |v_means2d| into v_means2d_abs; the Python wrapper
// stashes them without returning autograd gradients, so like GOF the
// signal never leaks into parameter gradients.
////////////////////////////////////////////////////////////////

template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_eval3d_geom_bwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    // fwd inputs
    const scalar_t *__restrict__ view2gaussians, // [C, N, 10]
    const vec2 *__restrict__ means2d,            // [C, N, 2]
    const vec3 *__restrict__ conics,             // [C, N, 3]
    const scalar_t *__restrict__ colors,         // [C, N, CDIM]
    const scalar_t *__restrict__ opacities,      // [C, N]
    const scalar_t *__restrict__ Ks,             // [C, 9]
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
        *__restrict__ v_render_normals, // [C, image_height, image_width, 3]
    const scalar_t
        *__restrict__ v_render_edepths, // [C, image_height, image_width, 1]
    const scalar_t
        *__restrict__ v_render_mdepths, // [C, image_height, image_width, 1]
    const scalar_t
        *__restrict__ v_render_distorts, // [C, image_height, image_width, 1]
    // grad inputs
    scalar_t *__restrict__ v_view2gaussians, // [C, N, 10]
    scalar_t *__restrict__ v_colors,         // [C, N, CDIM]
    scalar_t *__restrict__ v_opacities,      // [C, N]
    vec2 *__restrict__ v_means2d,            // [C, N, 2] densify signal
    vec2 *__restrict__ v_means2d_abs         // [C, N, 2] densify signal
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
    v_render_normals += camera_id * image_height * image_width * 3;
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
    const float rx = (px - cx) / fx;
    const float ry = (py - cy) / fy;

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
    float *v2g_batch = (float *)&conic_batch[block_size]; // [block_size * 10]
    float *rgbs_batch =
        (float *)&v2g_batch[block_size * 10]; // [block_size * CDIM]

    // this is the T AFTER the last gaussian in this pixel
    float T_final = 1.0f - render_alphas[pix_id];
    float T = T_final;
    // the contribution from gaussians behind the current one
    float buffer[CDIM] = {0.f};
    float buffer_n[3] = {0.f}; // sum_{i>k} w_i n_i
    float buffer_e = 0.f;      // sum_{i>k} w_i tz_i
    float buffer_d = 0.f;      // sum_{i>k} dl_dw_i w_i
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
    float v_render_n[3];
#pragma unroll
    for (uint32_t k = 0; k < 3; ++k) {
        v_render_n[k] = v_render_normals[pix_id * 3 + k];
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
#pragma unroll
            for (uint32_t k = 0; k < 10; ++k) {
                v2g_batch[tr * 10 + k] = view2gaussians[g * 10 + k];
            }
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
            float vis;
            float tz;
            float u0, u1, u2;
            double AA, BB;
            float CC;
            double min_value;

            if (valid) {
                const float *v2g = v2g_batch + t * 10;
                opac = xy_opacity_batch[t].z;
                u0 = v2g[0] * rx + v2g[1] * ry + v2g[2];
                u1 = v2g[1] * rx + v2g[3] * ry + v2g[4];
                u2 = v2g[2] * rx + v2g[4] * ry + v2g[5];
                AA = (double)rx * u0 + (double)ry * u1 + u2;
                BB = 2.0 *
                     ((double)v2g[6] * rx + (double)v2g[7] * ry + v2g[8]);
                CC = v2g[9];
                // mirror the forward's GOF near-plane skip: such
                // Gaussians contributed nothing
                tz = (float)(-BB / (2.0 * AA));
                if (tz <= distort_near) {
                    valid = false;
                }
                min_value = -(BB / AA) * (BB / 4.0) + CC;
                float power = -0.5f * (float)min_value;
                if (power > 0.0f) {
                    power = 0.0f;
                }
                vis = __expf(power);
                alpha = min(0.999f, opac * vis);
                if (alpha < ALPHA_THRESHOLD) {
                    valid = false;
                }
            }

            if (!warp.any(valid)) {
                continue;
            }
            float v_rgb_local[CDIM] = {0.f};
            float v_v2g_local[10] = {0.f};
            vec2 v_xy_local = {0.f, 0.f};
            vec2 v_xy_abs_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
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
                // per-pixel normal composites like a color channel
                const float len =
                    sqrtf(u0 * u0 + u1 * u1 + u2 * u2 + 1e-7f);
                const float nn[3] = {-u0 / len, -u1 / len, -u2 / len};
                float v_nn[3];
#pragma unroll
                for (uint32_t k = 0; k < 3; ++k) {
                    v_alpha += (nn[k] * T - buffer_n[k] * ra) * v_render_n[k];
                    v_nn[k] = fac * v_render_n[k];
                }

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

                // normal -> u = M r (GOF backward.cu:871-877)
                float v_len =
                    (v_nn[0] * u0 + v_nn[1] * u1 + v_nn[2] * u2) /
                    (len * len);
                float v_u[3] = {
                    (-v_nn[0] + v_len * u0) / len,
                    (-v_nn[1] + v_len * u1) / len,
                    (-v_nn[2] + v_len * u2) / len
                };

                // tz -> AA, BB (independent of the alpha saturation gate)
                double v_AA = v_tz * BB / (2.0 * AA * AA);
                double v_BB = v_tz * -1.0 / (2.0 * AA);
                float v_CC = 0.f;

                if (opac * vis <= 0.999f) {
                    v_opacity_local = vis * v_alpha;
                    // alpha -> min_value; zero when the forward clamped
                    // the power at 0 (min_value < 0, numerics only)
                    if (min_value >= 0.0) {
                        const double v_minv =
                            -0.5 * (double)(opac * vis * v_alpha);
                        v_AA += v_minv * (BB / AA) * (BB / AA) / 4.0;
                        v_BB += v_minv * -BB / (2.0 * AA);
                        v_CC += (float)v_minv;
                    }

                    // GOF densification signal (backward.cu:896-909):
                    // 2D-conic footprint with the 3D response G, in
                    // gsplat pixel units. Never returned as an autograd
                    // gradient (GOF's preprocess ignores dL_dmean2D).
                    const vec3 conic = conic_batch[t];
                    const vec2 xy = {
                        xy_opacity_batch[t].x, xy_opacity_batch[t].y
                    };
                    const vec2 delta = {xy.x - px, xy.y - py};
                    const float v_G = opac * v_alpha;
                    const float gdx = vis * delta.x;
                    const float gdy = vis * delta.y;
                    v_xy_local = {
                        v_G * (-gdx * conic.x - gdy * conic.y),
                        v_G * (-gdy * conic.z - gdx * conic.y)
                    };
                    v_xy_abs_local = {
                        abs(v_xy_local.x), abs(v_xy_local.y)
                    };
                }

                // AA = r . u folds into dL/du (GOF backward.cu:938-940)
                v_u[0] += (float)v_AA * rx;
                v_u[1] += (float)v_AA * ry;
                v_u[2] += (float)v_AA;

                // dL/dview2gaussians (GOF backward.cu:943-952)
                v_v2g_local[0] = v_u[0] * rx;
                v_v2g_local[1] = v_u[0] * ry + v_u[1] * rx;
                v_v2g_local[2] = v_u[0] + v_u[2] * rx;
                v_v2g_local[3] = v_u[1] * ry;
                v_v2g_local[4] = v_u[1] + v_u[2] * ry;
                v_v2g_local[5] = v_u[2];
                v_v2g_local[6] = (float)(v_BB * 2.0) * rx;
                v_v2g_local[7] = (float)(v_BB * 2.0) * ry;
                v_v2g_local[8] = (float)(v_BB * 2.0);
                v_v2g_local[9] = v_CC;

                // suffix updates
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    buffer[k] += rgbs_batch[t * CDIM + k] * fac;
                }
#pragma unroll
                for (uint32_t k = 0; k < 3; ++k) {
                    buffer_n[k] += nn[k] * fac;
                }
                buffer_e += tz * fac;
                buffer_d += dl_dw * fac;
                P0 += fac;
                P1 += fac * m;
                P2 += fac * m * m;
            }
            warpSum<CDIM>(v_rgb_local, warp);
            warpSum<10>(v_v2g_local, warp);
            warpSum(v_xy_local, warp);
            warpSum(v_xy_abs_local, warp);
            warpSum(v_opacity_local, warp);
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t];
                float *v_rgb_ptr = (float *)(v_colors) + CDIM * g;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                float *v_v2g_ptr = (float *)(v_view2gaussians) + 10 * g;
#pragma unroll
                for (uint32_t k = 0; k < 10; ++k) {
                    gpuAtomicAdd(v_v2g_ptr + k, v_v2g_local[k]);
                }

                gpuAtomicAdd(v_opacities + g, v_opacity_local);

                float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);

                float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
            }
        }
    }
}

template <uint32_t CDIM>
void launch_rasterize_to_pixels_eval3d_geom_bwd_kernel(
    // Gaussian parameters
    const at::Tensor view2gaussians, // [C, N, 10]
    const at::Tensor means2d,        // [C, N, 2]
    const at::Tensor conics,         // [C, N, 3]
    const at::Tensor colors,         // [C, N, CDIM]
    const at::Tensor opacities,      // [C, N]
    const at::Tensor Ks,             // [C, 3, 3]
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
    const at::Tensor v_render_normals, // [C, image_height, image_width, 3]
    const at::Tensor v_render_edepths, // [C, image_height, image_width, 1]
    const at::Tensor v_render_mdepths, // [C, image_height, image_width, 1]
    const at::Tensor v_render_distorts, // [C, image_height, image_width, 1]
    // outputs
    at::Tensor v_view2gaussians, // [C, N, 10]
    at::Tensor v_colors,         // [C, N, CDIM]
    at::Tensor v_opacities,      // [C, N]
    at::Tensor v_means2d,        // [C, N, 2] densify signal
    at::Tensor v_means2d_abs     // [C, N, 2] densify signal
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
        (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * 10 +
         sizeof(float) * CDIM);

    if (n_isects == 0) {
        return;
    }

    if (cudaFuncSetAttribute(
            rasterize_to_pixels_eval3d_geom_bwd_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }

    rasterize_to_pixels_eval3d_geom_bwd_kernel<CDIM, float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            C,
            N,
            n_isects,
            view2gaussians.data_ptr<float>(),
            reinterpret_cast<vec2 *>(means2d.data_ptr<float>()),
            reinterpret_cast<vec3 *>(conics.data_ptr<float>()),
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
            render_alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            median_ids.data_ptr<int32_t>(),
            dist_accums.data_ptr<float>(),
            v_render_colors.data_ptr<float>(),
            v_render_alphas.data_ptr<float>(),
            v_render_normals.data_ptr<float>(),
            v_render_edepths.data_ptr<float>(),
            v_render_mdepths.data_ptr<float>(),
            v_render_distorts.data_ptr<float>(),
            v_view2gaussians.data_ptr<float>(),
            v_colors.data_ptr<float>(),
            v_opacities.data_ptr<float>(),
            reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),
            reinterpret_cast<vec2 *>(v_means2d_abs.data_ptr<float>())
        );
}

#define __INS__(CDIM)                                                         \
    template void launch_rasterize_to_pixels_eval3d_geom_bwd_kernel<CDIM>(   \
        const at::Tensor view2gaussians,                                      \
        const at::Tensor means2d,                                             \
        const at::Tensor conics,                                              \
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
        const at::Tensor render_alphas,                                       \
        const at::Tensor last_ids,                                            \
        const at::Tensor median_ids,                                          \
        const at::Tensor dist_accums,                                         \
        const at::Tensor v_render_colors,                                     \
        const at::Tensor v_render_alphas,                                     \
        const at::Tensor v_render_normals,                                    \
        const at::Tensor v_render_edepths,                                    \
        const at::Tensor v_render_mdepths,                                    \
        const at::Tensor v_render_distorts,                                   \
        at::Tensor v_view2gaussians,                                          \
        at::Tensor v_colors,                                                  \
        at::Tensor v_opacities,                                               \
        at::Tensor v_means2d,                                                 \
        at::Tensor v_means2d_abs                                              \
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
