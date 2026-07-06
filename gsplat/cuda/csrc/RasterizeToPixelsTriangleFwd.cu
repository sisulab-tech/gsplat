// Forward rasterizer for the opaque-triangle (MeshSplatting) primitive.
//
// Clean-room (Apache-2.0) implementation mirroring the pure-PyTorch reference
// `sisu_040/gsplat/triangle_ref.py::render_triangles`. Per tile, per pixel, we
// walk the tile's triangles front-to-back (depth order supplied by isect_tiles)
// and composite:
//
//   d_k    = n_k . p + off_k            (signed distance to edge k)
//   inside = all(d_k <= 0)
//   phi_x  = max_k d_k                  (nearest edge, <= 0 inside)
//   window = max(0, phi_x * phi_center)^sigma
//   alpha  = min(0.999, opacity * window)
//   color  = barycentric interp of the 3 projected-vertex colors
//
// Structure follows RasterizeToPixels2DGSFwd.cu (shared-memory batched loads),
// but the per-primitive weight and the per-pixel color interpolation are the
// triangle-window math, not the Gaussian kernel.

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Triangle.h"

namespace gsplat {

namespace cg = cooperative_groups;

template <uint32_t CDIM>
__global__ void rasterize_to_pixels_triangle_fwd_kernel(
    const uint32_t C,        // number of cameras
    const uint32_t T,        // number of triangles
    const uint32_t n_isects, // number of tile-triangle intersections
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
    // outputs
    float *__restrict__ render_colors,  // [C, H, W, CDIM]
    float *__restrict__ render_alphas,  // [C, H, W, 1]
    float *__restrict__ render_normals, // [C, H, W, 3]
    float *__restrict__ render_depths,  // [C, H, W, 1]  expected (alpha-wtd) depth
    float *__restrict__ render_median,  // [C, H, W, 1]  median (surf) depth
    int32_t *__restrict__ last_ids,     // [C, H, W]
    int32_t *__restrict__ median_ids,   // [C, H, W]  isect idx of median tri
    // detached per-(camera, triangle) stats for densification / pruning
    float *__restrict__ max_blending,   // [C, T]
    int32_t *__restrict__ pixel_count   // [C, T]
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
    render_normals += camera_id * image_height * image_width * 3;
    render_depths += camera_id * image_height * image_width;
    render_median += camera_id * image_height * image_width;
    last_ids += camera_id * image_height * image_width;
    median_ids += camera_id * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += camera_id * CDIM;
    }
    if (masks != nullptr) {
        masks += camera_id * tile_height * tile_width;
    }

    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    const int32_t pix_id = i * image_width + j;

    bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    if (masks != nullptr && inside && !masks[tile_id]) {
        for (uint32_t k = 0; k < CDIM; ++k) {
            render_colors[pix_id * CDIM + k] =
                backgrounds == nullptr ? 0.0f : backgrounds[k];
        }
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

    // Shared-memory layout (per cached triangle):
    //   id | 3 edge (nx, ny, off) | 3 proj-verts (x, y) | (phi, opac, -) |
    //   normal | (vd0, vd1, vd2)
    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;
    vec3 *e_batch = reinterpret_cast<vec3 *>(&id_batch[block_size]);
    vec2 *pv_batch = reinterpret_cast<vec2 *>(&e_batch[block_size * 3]);
    vec3 *misc_batch = reinterpret_cast<vec3 *>(&pv_batch[block_size * 3]);
    vec3 *nrm_batch = reinterpret_cast<vec3 *>(&misc_batch[block_size]);
    vec3 *vd_batch = reinterpret_cast<vec3 *>(&nrm_batch[block_size]);

    float T_trans = 1.0f;
    uint32_t cur_idx = 0;
    uint32_t tr = block.thread_rank();

    float pix_out[CDIM] = {0.f};
    float normal_out[3] = {0.f};
    float depth_out = 0.f;
    // Median (surf) depth: depth_interp of the last composited triangle whose
    // pre-composite transmittance is still > 0.5 (2DGS/reference convention).
    float median_depth = 0.f;
    int32_t median_idx = -1;

    for (uint32_t b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < range_end) {
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
            pv_batch[tr * 3 + 0] = vec2(proj_verts[g * 6 + 0], proj_verts[g * 6 + 1]);
            pv_batch[tr * 3 + 1] = vec2(proj_verts[g * 6 + 2], proj_verts[g * 6 + 3]);
            pv_batch[tr * 3 + 2] = vec2(proj_verts[g * 6 + 4], proj_verts[g * 6 + 5]);
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

        uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const vec3 e0 = e_batch[t * 3 + 0];
            const vec3 e1 = e_batch[t * 3 + 1];
            const vec3 e2 = e_batch[t * 3 + 2];
            const float d0 = e0.x * px + e0.y * py + e0.z;
            const float d1 = e1.x * px + e1.y * py + e1.z;
            const float d2 = e2.x * px + e2.y * py + e2.z;
            // outside the triangle -> window is 0, nothing to composite
            if (d0 > 0.0f || d1 > 0.0f || d2 > 0.0f) {
                continue;
            }
            const float phi_x = fmaxf(d0, fmaxf(d1, d2));

            const vec3 misc = misc_batch[t];
            const float phi_c = misc.x;
            const float opac = misc.y;

            float base = phi_x * phi_c; // in [0, 1]
            if (base < 0.0f) {
                base = 0.0f;
            }
            const float window = powf(base, sigma);
            const float alpha = fminf(0.999f, opac * window);
            // Reference cutoff: sub-1/255 alphas are skipped entirely — no
            // compositing, no stats, and (in the backward) no gradient.
            if (alpha < 1.0f / 255.0f) {
                continue;
            }

            int32_t g = id_batch[t];
            // The reference counts the pixel (was_rendered) before the
            // early-termination test, so a terminal triangle still counts.
            atomicAdd(&pixel_count[g], 1);

            const float next_T = T_trans * (1.0f - alpha);
            if (next_T < 1e-4f) {
                done = true;
                break;
            }

            const float vis = alpha * T_trans;

            // Barycentric weights from the projected vertices. Matches the
            // reference (including its lopsided clamp_min(eps) on the denom).
            const vec2 pv0 = pv_batch[t * 3 + 0];
            const vec2 pv1 = pv_batch[t * 3 + 1];
            const vec2 pv2 = pv_batch[t * 3 + 2];
            const float v0x = pv1.x - pv0.x, v0y = pv1.y - pv0.y;
            const float v1x = pv2.x - pv0.x, v1y = pv2.y - pv0.y;
            const float v2x = px - pv0.x, v2y = py - pv0.y;
            // Sign-preserving magnitude clamp (see triangle_ref.py): keep the
            // true winding sign so negative-winding triangles interpolate
            // correctly instead of blowing up by ~1/eps.
            float bden = v0x * v1y - v1x * v0y;
            bden = (bden < 0.0f ? -1.0f : 1.0f) * fmaxf(fabsf(bden), eps);
            const float b0 = (v2x * v1y - v1x * v2y) / bden;
            const float b1 = (-v2x * v0y + v0x * v2y) / bden;
            const float b2 = 1.0f - b0 - b1;

            // Detached per-triangle stat (no gradient path): importance = max
            // blend weight this triangle achieves over the image.
            // vis = alpha * T_trans >= 0, so the non-negative-float atomicMax
            // int-reinterpret trick is valid.
            atomicMax((int *)&max_blending[g], __float_as_int(vis));

            const float *col = colors + g * 3 * CDIM;
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                const float ck = b2 * col[0 * CDIM + k] +
                                 b0 * col[1 * CDIM + k] + b1 * col[2 * CDIM + k];
                pix_out[k] += ck * vis;
            }

            const vec3 nrm = nrm_batch[t];
            normal_out[0] += nrm.x * vis;
            normal_out[1] += nrm.y * vis;
            normal_out[2] += nrm.z * vis;

            // Barycentric per-vertex depth (reference forward.cu:621:
            // depth_interp = wA*d0 + wB*d1 + wC*d2, wA=b2, wB=b0, wC=b1). Both
            // the expected depth (alpha-weighted) and the median depth read it.
            const vec3 vd = vd_batch[t];
            const float depth_interp = b2 * vd.x + b0 * vd.y + b1 * vd.z;
            depth_out += depth_interp * vis;
            // Median: pre-composite transmittance still > 0.5. Overwrite so the
            // last such triangle wins (forward.cu:625).
            if (T_trans > 0.5f) {
                median_depth = depth_interp;
                median_idx = (int32_t)(batch_start + t);
            }

            cur_idx = batch_start + t;
            T_trans = next_T;
        }
    }

    if (inside) {
        render_alphas[pix_id] = 1.0f - T_trans;
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            render_colors[pix_id * CDIM + k] =
                backgrounds == nullptr
                    ? pix_out[k]
                    : (pix_out[k] + T_trans * backgrounds[k]);
        }
#pragma unroll
        for (uint32_t k = 0; k < 3; ++k) {
            render_normals[pix_id * 3 + k] = normal_out[k];
        }
        render_depths[pix_id] = depth_out;
        render_median[pix_id] = median_depth;
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);
        median_ids[pix_id] = median_idx;
    }
}

template <uint32_t CDIM>
void launch_rasterize_to_pixels_triangle_fwd_kernel(
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
    // outputs
    at::Tensor render_colors,  // [C, H, W, CDIM]
    at::Tensor render_alphas,  // [C, H, W, 1]
    at::Tensor render_normals, // [C, H, W, 3]
    at::Tensor render_depths,  // [C, H, W, 1]
    at::Tensor render_median,  // [C, H, W, 1]
    at::Tensor last_ids,       // [C, H, W]
    at::Tensor median_ids,     // [C, H, W]
    at::Tensor max_blending,   // [C, T]
    at::Tensor pixel_count     // [C, T]
) {
    uint32_t C = tile_offsets.size(0);
    uint32_t T = proj_verts.size(1);
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);
    uint32_t n_isects = flatten_ids.size(0);

    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {C, tile_height, tile_width};

    // id | 3 edge vec3 | 3 proj-vert vec2 | misc vec3 | normal vec3 | vd vec3
    int64_t shmem_size =
        tile_size * tile_size *
        (sizeof(int32_t) + 3 * sizeof(vec3) + 3 * sizeof(vec2) + sizeof(vec3) +
         sizeof(vec3) + sizeof(vec3));

    if (cudaFuncSetAttribute(
            rasterize_to_pixels_triangle_fwd_kernel<CDIM>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }

    rasterize_to_pixels_triangle_fwd_kernel<CDIM>
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
            render_colors.data_ptr<float>(),
            render_alphas.data_ptr<float>(),
            render_normals.data_ptr<float>(),
            render_depths.data_ptr<float>(),
            render_median.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            median_ids.data_ptr<int32_t>(),
            max_blending.data_ptr<float>(),
            pixel_count.data_ptr<int32_t>()
        );
}

// Explicit instantiation. Triangle colors are RGB (3) or RGB + depth-ish
// channels; keep the common small channel counts.
#define __INS__(CDIM)                                                          \
    template void launch_rasterize_to_pixels_triangle_fwd_kernel<CDIM>(        \
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
        at::Tensor render_colors,                                              \
        at::Tensor render_alphas,                                              \
        at::Tensor render_normals,                                             \
        at::Tensor render_depths,                                              \
        at::Tensor render_median,                                             \
        at::Tensor last_ids,                                                   \
        at::Tensor median_ids,                                                \
        at::Tensor max_blending,                                               \
        at::Tensor pixel_count                                                 \
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
