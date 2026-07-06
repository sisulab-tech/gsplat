// Backward rasterizer for the opaque-triangle (MeshSplatting) primitive.
//
// Clean-room (Apache-2.0) VJP of `RasterizeToPixelsTriangleFwd.cu`, validated by
// gradient parity against the pure-PyTorch reference
// `sisu_040/gsplat/triangle_ref.py::render_triangles` (autograd).
//
// Structure mirrors RasterizeToPixels2DGSBwd.cu: per tile, walk the tile's
// triangles *back-to-front* (front-to-back reversed), recover the per-primitive
// transmittance T by dividing out (1 - alpha), and accumulate the standard
// front-to-back compositing gradients. The per-primitive geometry gradients are
// the triangle-window math (edge half-planes + incenter window + barycentric
// color), not the Gaussian kernel.
//
// Gradient targets (all per-triangle, scattered to shared triangles via atomics):
//   - v_colors        via barycentric interpolation weights
//   - v_proj_verts    via barycentric weights' dependence on the projected verts
//   - v_edge_normals  } the nearest edge (argmax signed distance) feeds phi_x,
//   - v_edge_offsets  }   hence the window; only that edge gets gradient
//   - v_phi_center    via window = (phi_x * phi_center)^sigma
//   - v_opacities     via alpha = opacity * window
//   - v_vertex_depths via barycentric per-vertex depth: the expected-depth map
//                     (vis-weighted) plus the median-depth map (bare bary at the
//                     one triangle that set the pixel's median)
//   - v_normals       via premultiplied normal accumulation
// sigma carries no gradient (scheduled scalar).

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Triangle.h"
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

template <uint32_t CDIM>
__global__ void rasterize_to_pixels_triangle_bwd_kernel(
    const uint32_t C,        // number of cameras
    const uint32_t T,        // number of triangles
    const uint32_t n_isects, // number of tile-triangle intersections
    // fwd inputs
    const float *__restrict__ proj_verts,   // [C, T, 3, 2]
    const float *__restrict__ edge_normals, // [C, T, 3, 2]
    const float *__restrict__ edge_offsets, // [C, T, 3]
    const float *__restrict__ phi_center,   // [C, T]
    const float *__restrict__ opacities,    // [C, T]
    const float *__restrict__ colors,        // [C, T, 3, CDIM] (3 vertex colors)
    const float *__restrict__ normals,       // [C, T, 3]
    const float *__restrict__ vertex_depths, // [C, T, 3]  per-vertex view-z
    const float sigma,
    const float eps,
    const float *__restrict__ backgrounds, // [C, CDIM]
    const bool *__restrict__ masks,        // [C, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // fwd outputs
    const float *__restrict__ render_alphas, // [C, H, W, 1]
    const int32_t *__restrict__ last_ids,    // [C, H, W]
    const int32_t *__restrict__ median_ids,  // [C, H, W]  isect idx of median tri
    // grad of outputs
    const float *__restrict__ v_render_colors,  // [C, H, W, CDIM]
    const float *__restrict__ v_render_alphas,  // [C, H, W, 1]
    const float *__restrict__ v_render_normals, // [C, H, W, 3]
    const float *__restrict__ v_render_depths,  // [C, H, W, 1]  expected depth
    const float *__restrict__ v_render_median,  // [C, H, W, 1]  median depth
    // grad of inputs (accumulated via atomics)
    float *__restrict__ v_proj_verts,     // [C, T, 3, 2]
    float *__restrict__ v_edge_normals,   // [C, T, 3, 2]
    float *__restrict__ v_edge_offsets,   // [C, T, 3]
    float *__restrict__ v_phi_center,     // [C, T]
    float *__restrict__ v_opacities,      // [C, T]
    float *__restrict__ v_colors,         // [C, T, 3, CDIM]
    float *__restrict__ v_normals,        // [C, T, 3]
    float *__restrict__ v_vertex_depths   // [C, T, 3]  per-vertex view-z grad
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
    v_render_colors += camera_id * image_height * image_width * CDIM;
    v_render_alphas += camera_id * image_height * image_width;
    v_render_normals += camera_id * image_height * image_width * 3;
    v_render_depths += camera_id * image_height * image_width;
    v_render_median += camera_id * image_height * image_width;
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
    const bool inside = (i < image_height && j < image_width);

    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    // Shared-memory layout (per cached triangle), same as the forward:
    //   id | 3 edge (nx, ny, off) | 3 proj-verts (x, y) | (phi, opac, depth) |
    //   normal. Colors are read from global (as in the forward) to keep the
    //   shared footprint independent of CDIM.
    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;
    vec3 *e_batch = reinterpret_cast<vec3 *>(&id_batch[block_size]);
    vec2 *pv_batch = reinterpret_cast<vec2 *>(&e_batch[block_size * 3]);
    vec3 *misc_batch = reinterpret_cast<vec3 *>(&pv_batch[block_size * 3]);
    vec3 *nrm_batch = reinterpret_cast<vec3 *>(&misc_batch[block_size]);
    vec3 *vd_batch = reinterpret_cast<vec3 *>(&nrm_batch[block_size]);

    // Transmittance after all contributing triangles (front-to-back product).
    const float T_final = inside ? 1.0f - render_alphas[pix_id] : 0.0f;
    float T_trans = T_final;
    // "later triangle" accumulators (contributions from triangles behind the
    // current one), used for d(out)/d(alpha).
    float buffer_c[CDIM] = {0.f};
    float buffer_n[3] = {0.f};
    float buffer_d = 0.f;

    const int32_t bin_final = inside ? last_ids[pix_id] : 0;

    // Upstream gradients for this pixel.
    float v_render_c[CDIM];
    float v_render_n[3];
    float v_render_d = 0.f;
    float v_render_a = 0.f;
    float v_render_med = 0.f;
#pragma unroll
    for (uint32_t k = 0; k < CDIM; ++k) {
        v_render_c[k] = inside ? v_render_colors[pix_id * CDIM + k] : 0.f;
    }
#pragma unroll
    for (uint32_t k = 0; k < 3; ++k) {
        v_render_n[k] = inside ? v_render_normals[pix_id * 3 + k] : 0.f;
    }
    if (inside) {
        v_render_d = v_render_depths[pix_id];
        v_render_a = v_render_alphas[pix_id];
        v_render_med = v_render_median[pix_id];
    }
    // Isect index of the triangle that set the median (surf) depth for this
    // pixel (-1 if the pixel never reached a triangle). Its per-vertex depth
    // gradient is applied when we reprocess exactly that triangle below.
    const int32_t median_id = inside ? median_ids[pix_id] : -1;

    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final =
        cg::reduce(warp, bin_final, cg::greater<int>());

    for (uint32_t b = 0; b < num_batches; ++b) {
        block.sync();

        // Load one batch of triangles, in reverse (0 index = furthest back).
        const int32_t batch_end = range_end - 1 - block_size * b;
        const int32_t batch_size = min(block_size, batch_end + 1 - range_start);
        const int32_t idx = batch_end - tr;
        if (idx >= range_start) {
            int32_t g = flatten_ids[idx]; // in [C * T]
            id_batch[tr] = g;
            e_batch[tr * 3 + 0] = vec3(
                edge_normals[g * 6 + 0],
                edge_normals[g * 6 + 1],
                edge_offsets[g * 3 + 0]
            );
            e_batch[tr * 3 + 1] = vec3(
                edge_normals[g * 6 + 2],
                edge_normals[g * 6 + 3],
                edge_offsets[g * 3 + 1]
            );
            e_batch[tr * 3 + 2] = vec3(
                edge_normals[g * 6 + 4],
                edge_normals[g * 6 + 5],
                edge_offsets[g * 3 + 2]
            );
            pv_batch[tr * 3 + 0] =
                vec2(proj_verts[g * 6 + 0], proj_verts[g * 6 + 1]);
            pv_batch[tr * 3 + 1] =
                vec2(proj_verts[g * 6 + 2], proj_verts[g * 6 + 3]);
            pv_batch[tr * 3 + 2] =
                vec2(proj_verts[g * 6 + 4], proj_verts[g * 6 + 5]);
            misc_batch[tr] = vec3(phi_center[g], opacities[g], 0.0f);
            nrm_batch[tr] =
                vec3(normals[g * 3], normals[g * 3 + 1], normals[g * 3 + 2]);
            vd_batch[tr] = vec3(
                vertex_depths[g * 3 + 0],
                vertex_depths[g * 3 + 1],
                vertex_depths[g * 3 + 2]
            );
        }
        block.sync();

        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size;
             ++t) {
            bool valid = inside;
            if (batch_end - t > bin_final) {
                valid = false;
            }

            // --- forward recompute for this (pixel, triangle) ---
            float alpha = 0.f, opac = 0.f, window = 0.f, base = 0.f,
                  phi_x = 0.f, phi_c = 0.f;
            int max_edge = 0;
            if (valid) {
                const vec3 e0 = e_batch[t * 3 + 0];
                const vec3 e1 = e_batch[t * 3 + 1];
                const vec3 e2 = e_batch[t * 3 + 2];
                const float d0 = e0.x * px + e0.y * py + e0.z;
                const float d1 = e1.x * px + e1.y * py + e1.z;
                const float d2 = e2.x * px + e2.y * py + e2.z;
                if (d0 > 0.0f || d1 > 0.0f || d2 > 0.0f) {
                    valid = false;
                } else {
                    const vec3 misc = misc_batch[t];
                    phi_c = misc.x;
                    opac = misc.y;
                    phi_x = fmaxf(d0, fmaxf(d1, d2));
                    // argmax = nearest edge (first occurrence, matching torch.max)
                    max_edge = (d0 >= d1) ? (d0 >= d2 ? 0 : 2)
                                          : (d1 >= d2 ? 1 : 2);
                    base = phi_x * phi_c;
                    if (base < 0.0f) {
                        base = 0.0f;
                    }
                    window = powf(base, sigma);
                    // Reference backward clamps at 0.99 (vs 0.999 in its own
                    // forward — an upstream inconsistency we reproduce) and
                    // skips sub-1/255 alphas: they got no gradient forward.
                    alpha = fminf(0.99f, opac * window);
                    if (alpha < 1.0f / 255.0f) {
                        valid = false;
                    }
                }
            }

            if (!warp.any(valid)) {
                continue;
            }

            // Per-primitive gradient accumulators for this pixel. The edge
            // gradient is kept per-edge (not just the nearest) because the
            // nearest edge varies per pixel/lane: warp-reducing a single
            // "nearest edge" slot and writing it to lane 0's edge index would
            // scatter every lane's gradient onto one edge. Each lane writes its
            // gradient into its own edge slot; the warp reduce then sums each of
            // the 3 slots independently.
            float v_col_local[3 * CDIM] = {0.f};
            vec2 v_pv_local[3] = {vec2(0.f), vec2(0.f), vec2(0.f)};
            vec2 v_en_local[3] = {vec2(0.f), vec2(0.f), vec2(0.f)};
            float v_eo_local[3] = {0.f, 0.f, 0.f};
            float v_phic_local = 0.f;
            float v_opac_local = 0.f;
            float v_vd_local[3] = {0.f}; // per-vertex view-z gradient
            float v_nrm_local[3] = {0.f};

            if (valid) {
                const float ra = 1.0f / (1.0f - alpha);
                T_trans *= ra; // transmittance in front of this triangle
                const float fac = alpha * T_trans; // visibility (blend weight)

                // Barycentric weights from the projected vertices (sign-
                // preserving magnitude clamp, matching the forward/reference).
                const vec2 pv0 = pv_batch[t * 3 + 0];
                const vec2 pv1 = pv_batch[t * 3 + 1];
                const vec2 pv2 = pv_batch[t * 3 + 2];
                const float v0x = pv1.x - pv0.x, v0y = pv1.y - pv0.y;
                const float v1x = pv2.x - pv0.x, v1y = pv2.y - pv0.y;
                const float v2x = px - pv0.x, v2y = py - pv0.y;
                const float bden_raw = v0x * v1y - v1x * v0y;
                const bool saturated = fabsf(bden_raw) < eps;
                const float bden =
                    (bden_raw < 0.0f ? -1.0f : 1.0f) * fmaxf(fabsf(bden_raw), eps);
                const float b0 = (v2x * v1y - v1x * v2y) / bden;
                const float b1 = (-v2x * v0y + v0x * v2y) / bden;
                const float b2 = 1.0f - b0 - b1;

                const float *col = colors + id_batch[t] * 3 * CDIM;

                // Interpolated color, plus color/barycentric gradients.
                float cpix[CDIM];
                float gb0 = 0.f, gb1 = 0.f;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    const float c0 = col[0 * CDIM + k];
                    const float c1 = col[1 * CDIM + k];
                    const float c2 = col[2 * CDIM + k];
                    cpix[k] = b2 * c0 + b0 * c1 + b1 * c2;
                    const float vcp = fac * v_render_c[k]; // dL/d(color pixel)
                    v_col_local[0 * CDIM + k] += vcp * b2;
                    v_col_local[1 * CDIM + k] += vcp * b0;
                    v_col_local[2 * CDIM + k] += vcp * b1;
                    gb0 += vcp * (c1 - c0);
                    gb1 += vcp * (c2 - c0);
                }

                // Barycentric weights -> projected vertices.
                const float inv = 1.0f / bden;
                const float dnum0 = gb0 * inv;
                const float dnum1 = gb1 * inv;
                const float dbden =
                    saturated ? 0.f : -(gb0 * b0 + gb1 * b1) * inv;
                const float gv0x = dnum1 * v2y + dbden * v1y;
                const float gv0y = -dnum1 * v2x - dbden * v1x;
                const float gv1x = -dnum0 * v2y - dbden * v0y;
                const float gv1y = dnum0 * v2x + dbden * v0x;
                const float gv2x = dnum0 * v1y - dnum1 * v0y;
                const float gv2y = -dnum0 * v1x + dnum1 * v0x;
                v_pv_local[0].x = -(gv0x + gv1x + gv2x);
                v_pv_local[0].y = -(gv0y + gv1y + gv2y);
                v_pv_local[1].x = gv0x;
                v_pv_local[1].y = gv0y;
                v_pv_local[2].x = gv1x;
                v_pv_local[2].y = gv1y;

                // Barycentric per-vertex depth (matches the forward):
                // depth_interp = b2*vd0 + b0*vd1 + b1*vd2.
                const vec3 vd = vd_batch[t];
                const float depth_interp = b2 * vd.x + b0 * vd.y + b1 * vd.z;
                // Expected-depth gradient to the 3 per-vertex depths, weighted by
                // vis*bary (reference dL_dvertice_depth[v] += dL_ddepth*fac*w).
                v_vd_local[0] = fac * b2 * v_render_d;
                v_vd_local[1] = fac * b0 * v_render_d;
                v_vd_local[2] = fac * b1 * v_render_d;
                // Median-depth gradient: only the triangle that set this pixel's
                // median contributes, weighted by the bare bary weight (no vis,
                // no alpha — reference commented out dL_dz; the bary->proj_verts
                // path is the dead dL_dpoints2D and is dropped, like color).
                if ((batch_end - (int32_t)t) == median_id) {
                    v_vd_local[0] += b2 * v_render_med;
                    v_vd_local[1] += b0 * v_render_med;
                    v_vd_local[2] += b1 * v_render_med;
                }
                const vec3 nrm = nrm_batch[t];
                v_nrm_local[0] = fac * v_render_n[0];
                v_nrm_local[1] = fac * v_render_n[1];
                v_nrm_local[2] = fac * v_render_n[2];

                // d(out)/d(alpha): current-triangle term minus the "later"
                // buffer term, over color / normal / depth / alpha outputs.
                float v_alpha = 0.f;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    v_alpha +=
                        (cpix[k] * T_trans - buffer_c[k] * ra) * v_render_c[k];
                }
                v_alpha += (nrm.x * T_trans - buffer_n[0] * ra) * v_render_n[0];
                v_alpha += (nrm.y * T_trans - buffer_n[1] * ra) * v_render_n[1];
                v_alpha += (nrm.z * T_trans - buffer_n[2] * ra) * v_render_n[2];
                v_alpha += (depth_interp * T_trans - buffer_d * ra) * v_render_d;
                v_alpha += T_final * ra * v_render_a;
                if (backgrounds != nullptr) {
                    float accum = 0.f;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        accum += backgrounds[k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }

                // alpha = min(0.99, opacity * window). The reference propagates
                // the gradient straight through the clamp (dL_dopacity +=
                // dL_dalpha * Cx unconditionally), so no saturation gate here.
                v_opac_local = v_alpha * window;
                const float v_window = v_alpha * opac;
                // window = base^sigma  (base > 0 here)
                const float v_base = v_window * sigma * powf(base, sigma - 1.0f);
                // base = phi_x * phi_center
                const float v_phi_x = v_base * phi_c;
                v_phic_local = v_base * phi_x;
                // phi_x = max_k(edge_k . p): route to the nearest edge slot.
                v_en_local[max_edge] = vec2(v_phi_x * px, v_phi_x * py);
                v_eo_local[max_edge] = v_phi_x;

                // Update the "later triangle" buffers for the next (front) one.
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    buffer_c[k] += cpix[k] * fac;
                }
                buffer_n[0] += nrm.x * fac;
                buffer_n[1] += nrm.y * fac;
                buffer_n[2] += nrm.z * fac;
                buffer_d += depth_interp * fac;
            }

            // Warp-reduce then scatter once per warp to cut atomic traffic.
            warpSum<3 * CDIM>(v_col_local, warp);
            warpSum<3>(v_nrm_local, warp);
            warpSum(v_pv_local[0], warp);
            warpSum(v_pv_local[1], warp);
            warpSum(v_pv_local[2], warp);
            warpSum(v_en_local[0], warp);
            warpSum(v_en_local[1], warp);
            warpSum(v_en_local[2], warp);
            warpSum<3>(v_eo_local, warp);
            warpSum(v_phic_local, warp);
            warpSum(v_opac_local, warp);
            warpSum<3>(v_vd_local, warp);

            if (warp.thread_rank() == 0) {
                const int32_t g = id_batch[t];
                float *vc = v_colors + g * 3 * CDIM;
#pragma unroll
                for (uint32_t k = 0; k < 3 * CDIM; ++k) {
                    gpuAtomicAdd(vc + k, v_col_local[k]);
                }
                float *vpv = v_proj_verts + g * 6;
                gpuAtomicAdd(vpv + 0, v_pv_local[0].x);
                gpuAtomicAdd(vpv + 1, v_pv_local[0].y);
                gpuAtomicAdd(vpv + 2, v_pv_local[1].x);
                gpuAtomicAdd(vpv + 3, v_pv_local[1].y);
                gpuAtomicAdd(vpv + 4, v_pv_local[2].x);
                gpuAtomicAdd(vpv + 5, v_pv_local[2].y);
                float *ven = v_edge_normals + g * 6;
                float *veo = v_edge_offsets + g * 3;
#pragma unroll
                for (int k = 0; k < 3; ++k) {
                    gpuAtomicAdd(ven + k * 2 + 0, v_en_local[k].x);
                    gpuAtomicAdd(ven + k * 2 + 1, v_en_local[k].y);
                    gpuAtomicAdd(veo + k, v_eo_local[k]);
                }
                gpuAtomicAdd(v_phi_center + g, v_phic_local);
                gpuAtomicAdd(v_opacities + g, v_opac_local);
                gpuAtomicAdd(v_vertex_depths + g * 3 + 0, v_vd_local[0]);
                gpuAtomicAdd(v_vertex_depths + g * 3 + 1, v_vd_local[1]);
                gpuAtomicAdd(v_vertex_depths + g * 3 + 2, v_vd_local[2]);
                gpuAtomicAdd(v_normals + g * 3 + 0, v_nrm_local[0]);
                gpuAtomicAdd(v_normals + g * 3 + 1, v_nrm_local[1]);
                gpuAtomicAdd(v_normals + g * 3 + 2, v_nrm_local[2]);
            }
        }
    }
}

template <uint32_t CDIM>
void launch_rasterize_to_pixels_triangle_bwd_kernel(
    // fwd inputs
    const at::Tensor proj_verts,   // [C, T, 3, 2]
    const at::Tensor edge_normals, // [C, T, 3, 2]
    const at::Tensor edge_offsets, // [C, T, 3]
    const at::Tensor phi_center,   // [C, T]
    const at::Tensor opacities,    // [C, T]
    const at::Tensor colors,        // [C, T, 3, CDIM]
    const at::Tensor normals,       // [C, T, 3]
    const at::Tensor vertex_depths, // [C, T, 3]
    const float sigma,
    const float eps,
    const at::optional<at::Tensor> backgrounds, // [C, CDIM]
    const at::optional<at::Tensor> masks,       // [C, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const at::Tensor tile_offsets, // [C, tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // fwd outputs
    const at::Tensor render_alphas, // [C, H, W, 1]
    const at::Tensor last_ids,      // [C, H, W]
    const at::Tensor median_ids,    // [C, H, W]
    // grad outputs
    const at::Tensor v_render_colors,  // [C, H, W, CDIM]
    const at::Tensor v_render_alphas,  // [C, H, W, 1]
    const at::Tensor v_render_normals, // [C, H, W, 3]
    const at::Tensor v_render_depths,  // [C, H, W, 1]
    const at::Tensor v_render_median,  // [C, H, W, 1]
    // grad inputs
    at::Tensor v_proj_verts,     // [C, T, 3, 2]
    at::Tensor v_edge_normals,   // [C, T, 3, 2]
    at::Tensor v_edge_offsets,   // [C, T, 3]
    at::Tensor v_phi_center,     // [C, T]
    at::Tensor v_opacities,      // [C, T]
    at::Tensor v_colors,         // [C, T, 3, CDIM]
    at::Tensor v_normals,        // [C, T, 3]
    at::Tensor v_vertex_depths   // [C, T, 3]
) {
    uint32_t C = tile_offsets.size(0);
    uint32_t T = proj_verts.size(1);
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);
    uint32_t n_isects = flatten_ids.size(0);

    if (n_isects == 0) {
        return;
    }

    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {C, tile_height, tile_width};

    // id | 3 edge vec3 | 3 proj-vert vec2 | misc vec3 | normal vec3 | vd vec3
    int64_t shmem_size =
        tile_size * tile_size *
        (sizeof(int32_t) + 3 * sizeof(vec3) + 3 * sizeof(vec2) + sizeof(vec3) +
         sizeof(vec3) + sizeof(vec3));

    if (cudaFuncSetAttribute(
            rasterize_to_pixels_triangle_bwd_kernel<CDIM>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }

    rasterize_to_pixels_triangle_bwd_kernel<CDIM>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            C,
            T,
            n_isects,
            proj_verts.data_ptr<float>(),
            edge_normals.data_ptr<float>(),
            edge_offsets.data_ptr<float>(),
            phi_center.data_ptr<float>(),
            opacities.data_ptr<float>(),
            colors.data_ptr<float>(),
            normals.data_ptr<float>(),
            vertex_depths.data_ptr<float>(),
            sigma,
            eps,
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
            v_render_colors.data_ptr<float>(),
            v_render_alphas.data_ptr<float>(),
            v_render_normals.data_ptr<float>(),
            v_render_depths.data_ptr<float>(),
            v_render_median.data_ptr<float>(),
            v_proj_verts.data_ptr<float>(),
            v_edge_normals.data_ptr<float>(),
            v_edge_offsets.data_ptr<float>(),
            v_phi_center.data_ptr<float>(),
            v_opacities.data_ptr<float>(),
            v_colors.data_ptr<float>(),
            v_normals.data_ptr<float>(),
            v_vertex_depths.data_ptr<float>()
        );
}

#define __INS__(CDIM)                                                          \
    template void launch_rasterize_to_pixels_triangle_bwd_kernel<CDIM>(        \
        const at::Tensor proj_verts,                                           \
        const at::Tensor edge_normals,                                         \
        const at::Tensor edge_offsets,                                         \
        const at::Tensor phi_center,                                           \
        const at::Tensor opacities,                                            \
        const at::Tensor colors,                                               \
        const at::Tensor normals,                                              \
        const at::Tensor vertex_depths,                                        \
        const float sigma,                                                     \
        const float eps,                                                       \
        const at::optional<at::Tensor> backgrounds,                            \
        const at::optional<at::Tensor> masks,                                  \
        const uint32_t image_width,                                            \
        const uint32_t image_height,                                           \
        const uint32_t tile_size,                                              \
        const at::Tensor tile_offsets,                                         \
        const at::Tensor flatten_ids,                                          \
        const at::Tensor render_alphas,                                        \
        const at::Tensor last_ids,                                             \
        const at::Tensor median_ids,                                           \
        const at::Tensor v_render_colors,                                      \
        const at::Tensor v_render_alphas,                                      \
        const at::Tensor v_render_normals,                                     \
        const at::Tensor v_render_depths,                                      \
        const at::Tensor v_render_median,                                      \
        at::Tensor v_proj_verts,                                               \
        at::Tensor v_edge_normals,                                             \
        at::Tensor v_edge_offsets,                                             \
        at::Tensor v_phi_center,                                               \
        at::Tensor v_opacities,                                                \
        at::Tensor v_colors,                                                   \
        at::Tensor v_normals,                                                  \
        at::Tensor v_vertex_depths                                             \
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
