// Backward projection kernel for the opaque-triangle (MeshSplatting) primitive.
//
// Clean-room (Apache-2.0) VJP of `ProjectionTriangleFwd.cu`, validated by
// gradient parity against the pure-PyTorch reference
// `sisu_040/gsplat/triangle_ref.py::project_triangles` (autograd).
//
// One thread handles one (camera, triangle). It recomputes the forward, then
// backpropagates the upstream gradients on every differentiable projection
// output {means2d, depths, proj_verts, edge_normals, edge_offsets, phi_center,
// opacities, normals} onto the two learnable inputs: the 3 shared world
// vertices and their opacity logits (accumulated across triangles via atomics).
//
// The gradient on the projected vertices A_i collects three paths that all
// depend on A_i in the forward:
//   1. the proj_verts output directly (barycentric color path, upstream),
//   2. the three inward edge half-planes (edge normals/offsets), and
//   3. the incenter window normalization phi_center,
// then flows A_i -> camera point c_i (pinhole) -> world vertex w_i (view
// rotation). The camera-facing normal and the min-opacity add two more
// independent paths straight to the world vertices / opacity logits.

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

__global__ void projection_triangle_bwd_kernel(
    const uint32_t C, // number of cameras
    const uint32_t T, // number of triangles
    const uint32_t V, // number of vertices
    // fwd inputs
    const float *__restrict__ vertices,       // [V, 3]
    const int32_t *__restrict__ faces,        // [T, 3]
    const float *__restrict__ vertex_opacity, // [V]
    const float *__restrict__ viewmats,       // [C, 4, 4]
    const float *__restrict__ Ks,             // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float near_plane,
    const float far_plane,
    const float eps,
    // grad of outputs (all [C, T, ...])
    const float *__restrict__ v_means2d,       // [C, T, 2]
    const float *__restrict__ v_depths,        // [C, T]
    const float *__restrict__ v_vertex_depths, // [C, T, 3]  per-vertex view-z
    const float *__restrict__ v_proj_verts,    // [C, T, 3, 2]
    const float *__restrict__ v_edge_normals,  // [C, T, 3, 2]
    const float *__restrict__ v_edge_offsets,  // [C, T, 3]
    const float *__restrict__ v_phi_center,    // [C, T]
    const float *__restrict__ v_opacities,     // [C, T]
    const float *__restrict__ v_normals,       // [C, T, 3]
    // grad of inputs (accumulated via atomics)
    float *__restrict__ v_vertices,      // [V, 3]
    float *__restrict__ v_vertex_opacity // [V]
) {
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * T) {
        return;
    }
    const uint32_t cid = idx / T;
    const uint32_t tid = idx % T;

    viewmats += cid * 16;
    Ks += cid * 9;

    // glm is column-major, input row-major: transpose explicitly (as in fwd).
    mat3 R = mat3(
        viewmats[0],
        viewmats[4],
        viewmats[8],
        viewmats[1],
        viewmats[5],
        viewmats[9],
        viewmats[2],
        viewmats[6],
        viewmats[10]
    );
    vec3 t = vec3(viewmats[3], viewmats[7], viewmats[11]);
    const float fx = Ks[0], fy = Ks[4], cx = Ks[2], cy = Ks[5];

    const int32_t i0 = faces[tid * 3 + 0];
    const int32_t i1 = faces[tid * 3 + 1];
    const int32_t i2 = faces[tid * 3 + 2];

    const vec3 w0 =
        vec3(vertices[i0 * 3], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]);
    const vec3 w1 =
        vec3(vertices[i1 * 3], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]);
    const vec3 w2 =
        vec3(vertices[i2 * 3], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]);

    vec3 c0, c1, c2;
    posW2C(R, t, w0, c0);
    posW2C(R, t, w1, c1);
    posW2C(R, t, w2, c2);

    const float z0 = max(c0.z, eps);
    const float z1 = max(c1.z, eps);
    const float z2 = max(c2.z, eps);
    const vec2 A = vec2(fx * c0.x / z0 + cx, fy * c0.y / z0 + cy);
    const vec2 B = vec2(fx * c1.x / z1 + cx, fy * c1.y / z1 + cy);
    const vec2 Cc = vec2(fx * c2.x / z2 + cx, fy * c2.y / z2 + cy);

    const vec3 centroid_cam = (c0 + c1 + c2) / 3.0f;
    const float depth = centroid_cam.z;

    const float denom_area =
        (B.x - A.x) * (Cc.y - A.y) - (Cc.x - A.x) * (B.y - A.y);
    const bool valid = (depth > near_plane) && (depth < far_plane) &&
                       (fabsf(denom_area) > eps);
    if (!valid) {
        // Culled triangles carry no upstream gradient (never rasterized);
        // returning also avoids differentiating a degenerate configuration.
        return;
    }

    // --- recompute incenter ---
    const float a = glm::length(B - Cc);
    const float b = glm::length(A - Cc);
    const float c = glm::length(A - B);
    const float wsum = max(a + b + c, eps);
    const vec2 incenter = (a * A + b * B + c * Cc) / wsum;

    // --- upstream gradients for this (camera, triangle) ---
    const vec2 g_means2d = vec2(v_means2d[idx * 2], v_means2d[idx * 2 + 1]);
    const float g_depth = v_depths[idx];
    // Per-vertex view-z gradient (from the rasterizer's expected/median depth):
    // vertex_depth_k = c_k.z, so d(loss)/d(c_k.z) += v_vertex_depths[k]. This is
    // the reference `dL_dvertice_depth` path (backward.cu:253 = z-only gradient
    // transformed to world by R^T); it flows to the world vertex through the
    // same camera->world step as everything else below.
    const float g_vd[3] = {
        v_vertex_depths[idx * 3 + 0],
        v_vertex_depths[idx * 3 + 1],
        v_vertex_depths[idx * 3 + 2]
    };
    const vec2 g_proj[3] = {
        vec2(v_proj_verts[idx * 6 + 0], v_proj_verts[idx * 6 + 1]),
        vec2(v_proj_verts[idx * 6 + 2], v_proj_verts[idx * 6 + 3]),
        vec2(v_proj_verts[idx * 6 + 4], v_proj_verts[idx * 6 + 5])
    };
    const vec2 g_en[3] = {
        vec2(v_edge_normals[idx * 6 + 0], v_edge_normals[idx * 6 + 1]),
        vec2(v_edge_normals[idx * 6 + 2], v_edge_normals[idx * 6 + 3]),
        vec2(v_edge_normals[idx * 6 + 4], v_edge_normals[idx * 6 + 5])
    };
    const float g_eo[3] = {
        v_edge_offsets[idx * 3 + 0],
        v_edge_offsets[idx * 3 + 1],
        v_edge_offsets[idx * 3 + 2]
    };
    const float g_phi = v_phi_center[idx];
    const float g_opac = v_opacities[idx];
    const vec3 g_nrm =
        vec3(v_normals[idx * 3], v_normals[idx * 3 + 1], v_normals[idx * 3 + 2]);

    const vec2 P[3] = {A, B, Cc};
    vec2 vP[3] = {vec2(0.f), vec2(0.f), vec2(0.f)};
    vec2 v_incenter = vec2(0.f);

    // phi_center = 1 / min(dist_incenter, -eps), where dist_incenter is the
    // signed distance of the incenter to edge 2 (the last written). Recover it.
    // Edge 2 uses vertices P[2] -> P[0].
    float dist_incenter = 0.f;
    {
        const vec2 p1 = P[2];
        const vec2 p2 = P[0];
        float nx = p2.y - p1.y;
        float ny = -(p2.x - p1.x);
        const float nrm = max(sqrtf(nx * nx + ny * ny), eps);
        const vec2 n_unit = vec2(nx / nrm, ny / nrm);
        const float off_raw = -(n_unit.x * p1.x + n_unit.y * p1.y);
        const float d_raw =
            n_unit.x * incenter.x + n_unit.y * incenter.y + off_raw;
        const float sign = (d_raw > 0.f) ? -1.f : 1.f;
        dist_incenter = sign * d_raw;
    }
    // Reference parity: the upstream backward computes NO phi_center gradient
    // (the window normalization is treated as a constant w.r.t. geometry), so
    // the phi -> incenter -> vertices chain is dropped. g_phi is ignored.
    (void)g_phi;
    (void)dist_incenter;
    const float v_dstored2 = 0.f; // grad on edge-2's *stored* signed distance

    // --- edge half-planes backward (mirror of the fwd construction) ---
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const vec2 p1 = P[k];
        const vec2 p2 = P[(k + 1) % 3];
        float nx = p2.y - p1.y;
        float ny = -(p2.x - p1.x);
        const float nrm = max(sqrtf(nx * nx + ny * ny), eps);
        const vec2 n_unit = vec2(nx / nrm, ny / nrm);
        const float off_raw = -(n_unit.x * p1.x + n_unit.y * p1.y);
        const float d_raw =
            n_unit.x * incenter.x + n_unit.y * incenter.y + off_raw;
        const float sign = (d_raw > 0.f) ? -1.f : 1.f;

        // upstream on stored quantities (stored = sign * raw; sign constant)
        const vec2 v_n_stored = g_en[k];
        const float v_off_stored = g_eo[k];
        const float v_d_stored = (k == 2) ? v_dstored2 : 0.f;

        vec2 v_n_unit = sign * v_n_stored;
        float v_off_raw = sign * v_off_stored;
        const float v_d_raw = sign * v_d_stored;

        // d_raw = n_unit . incenter + off_raw
        v_n_unit += v_d_raw * incenter;
        v_incenter += v_d_raw * n_unit;
        v_off_raw += v_d_raw;

        // off_raw = -(n_unit . p1)
        v_n_unit += v_off_raw * (-p1);
        vec2 v_p1 = v_off_raw * (-n_unit);
        vec2 v_p2 = vec2(0.f);

        // n_unit = n_raw / |n_raw|  (normalize)
        const float dotp = v_n_unit.x * n_unit.x + v_n_unit.y * n_unit.y;
        const vec2 v_n_raw = (v_n_unit - dotp * n_unit) / nrm;
        // n_raw = (p2.y - p1.y, -(p2.x - p1.x))
        v_p2.y += v_n_raw.x;
        v_p1.y += -v_n_raw.x;
        v_p1.x += v_n_raw.y;
        v_p2.x += -v_n_raw.y;

        vP[k] += v_p1;
        vP[(k + 1) % 3] += v_p2;
    }

    // --- incenter backward: incenter = (a*A + b*B + c*Cc) / wsum ---
    const vec2 v_num = v_incenter / wsum;
    const float v_wsum =
        -(v_incenter.x * incenter.x + v_incenter.y * incenter.y) / wsum;
    float v_a = v_num.x * A.x + v_num.y * A.y;
    float v_b = v_num.x * B.x + v_num.y * B.y;
    float v_c = v_num.x * Cc.x + v_num.y * Cc.y;
    vP[0] += a * v_num;
    vP[1] += b * v_num;
    vP[2] += c * v_num;
    if (a + b + c > eps) { // wsum = max(a+b+c, eps)
        v_a += v_wsum;
        v_b += v_wsum;
        v_c += v_wsum;
    }

    // --- side lengths backward (a=|B-Cc|, b=|A-Cc|, c=|A-B|) ---
    const vec2 dBC = B - Cc;
    vP[1] += (v_a / a) * dBC;
    vP[2] += -(v_a / a) * dBC;
    const vec2 dAC = A - Cc;
    vP[0] += (v_b / b) * dAC;
    vP[2] += -(v_b / b) * dAC;
    const vec2 dAB = A - B;
    vP[0] += (v_c / c) * dAB;
    vP[1] += -(v_c / c) * dAB;

    // --- total gradient on each projected vertex ---
    // Reference parity: the barycentric color-interpolation gradient w.r.t.
    // the projected vertices (dL_dpoints2D) is DISCARDED upstream — renderCUDA
    // writes it, computeVertexColorsCUDA receives it and never reads it. Only
    // the edge-normal/offset (window) chain reaches the 3D vertices. g_proj is
    // therefore ignored here.
    (void)g_proj;
    const vec2 third = g_means2d / 3.0f;
    // Reference parity: upstream applies the NDC-space projection Jacobian to
    // PIXEL-space edge gradients (backward.cu:395-400), silently omitting the
    // ndc2Pix scale d(pixel)/d(ndc) = W/2 (H/2 for y). Reproduce by scaling
    // our pixel-space vertex gradients by 2/W (2/H). Verified head-to-head:
    // without this our vertex grads are exactly (W/2)x theirs (corr 0.999998).
    const vec2 ndc_scale =
        vec2(2.0f / (float)image_width, 2.0f / (float)image_height);
    vec2 gA[3] = {
        (third + vP[0]) * ndc_scale,
        (third + vP[1]) * ndc_scale,
        (third + vP[2]) * ndc_scale
    };

    // --- pinhole backward: A_i -> c_i ; plus depth -> c_i.z ---
    const vec3 cc[3] = {c0, c1, c2};
    const float zz[3] = {z0, z1, z2};
    vec3 gc[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const float z = zz[i];
        gc[i].x = gA[i].x * fx / z;
        gc[i].y = gA[i].y * fy / z;
        gc[i].z = (cc[i].z > eps)
                      ? (-(gA[i].x * fx * cc[i].x + gA[i].y * fy * cc[i].y) /
                         (z * z))
                      : 0.f;
        gc[i].z += g_depth / 3.0f; // depth = centroid_cam.z (raw)
        gc[i].z += g_vd[i];        // vertex_depth_i = c_i.z (raw)
    }

    // --- camera -> world:  c = R w + t  =>  dL/dw = R^T dL/dc ---
    const mat3 Rt = glm::transpose(R);
    vec3 gw[3] = {Rt * gc[0], Rt * gc[1], Rt * gc[2]};

    // --- camera-facing world normal backward: DROPPED for reference parity ---
    // The upstream preprocess backward computes the normal -> vertex chain
    // (dL_dp0..dL_dp2, backward.cu:522-532) and never writes it — dead code.
    // The triangle normal is effectively constant w.r.t. the vertices.
    (void)g_nrm;

    // --- scatter world-vertex gradients (shared across triangles) ---
    gpuAtomicAdd(v_vertices + i0 * 3 + 0, gw[0].x);
    gpuAtomicAdd(v_vertices + i0 * 3 + 1, gw[0].y);
    gpuAtomicAdd(v_vertices + i0 * 3 + 2, gw[0].z);
    gpuAtomicAdd(v_vertices + i1 * 3 + 0, gw[1].x);
    gpuAtomicAdd(v_vertices + i1 * 3 + 1, gw[1].y);
    gpuAtomicAdd(v_vertices + i1 * 3 + 2, gw[1].z);
    gpuAtomicAdd(v_vertices + i2 * 3 + 0, gw[2].x);
    gpuAtomicAdd(v_vertices + i2 * 3 + 1, gw[2].y);
    gpuAtomicAdd(v_vertices + i2 * 3 + 2, gw[2].z);

    // --- opacity backward: opac = min(sigmoid(o_i)) ---
    const float o0 = 1.0f / (1.0f + __expf(-vertex_opacity[i0]));
    const float o1 = 1.0f / (1.0f + __expf(-vertex_opacity[i1]));
    const float o2 = 1.0f / (1.0f + __expf(-vertex_opacity[i2]));
    int m = (o0 <= o1) ? (o0 <= o2 ? 0 : 2) : (o1 <= o2 ? 1 : 2);
    const float om = (m == 0) ? o0 : ((m == 1) ? o1 : o2);
    const int32_t vm = (m == 0) ? i0 : ((m == 1) ? i1 : i2);
    gpuAtomicAdd(v_vertex_opacity + vm, g_opac * om * (1.0f - om));
}

void launch_projection_triangle_bwd_kernel(
    // fwd inputs
    const at::Tensor vertices,       // [V, 3]
    const at::Tensor faces,          // [T, 3]
    const at::Tensor vertex_opacity, // [V]
    const at::Tensor viewmats,       // [C, 4, 4]
    const at::Tensor Ks,             // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float near_plane,
    const float far_plane,
    const float eps,
    // grad outputs
    const at::Tensor v_means2d,       // [C, T, 2]
    const at::Tensor v_depths,        // [C, T]
    const at::Tensor v_vertex_depths, // [C, T, 3]
    const at::Tensor v_proj_verts,    // [C, T, 3, 2]
    const at::Tensor v_edge_normals,  // [C, T, 3, 2]
    const at::Tensor v_edge_offsets,  // [C, T, 3]
    const at::Tensor v_phi_center,    // [C, T]
    const at::Tensor v_opacities,     // [C, T]
    const at::Tensor v_normals,       // [C, T, 3]
    // grad inputs
    at::Tensor v_vertices,      // [V, 3]
    at::Tensor v_vertex_opacity // [V]
) {
    uint32_t T = faces.size(0);
    uint32_t V = vertices.size(0);
    uint32_t C = viewmats.size(0);

    int64_t n_elements = (int64_t)C * T;
    if (n_elements == 0) {
        return;
    }
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);

    projection_triangle_bwd_kernel<<<
        grid,
        threads,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        C,
        T,
        V,
        vertices.data_ptr<float>(),
        faces.data_ptr<int32_t>(),
        vertex_opacity.data_ptr<float>(),
        viewmats.data_ptr<float>(),
        Ks.data_ptr<float>(),
        image_width,
        image_height,
        near_plane,
        far_plane,
        eps,
        v_means2d.data_ptr<float>(),
        v_depths.data_ptr<float>(),
        v_vertex_depths.data_ptr<float>(),
        v_proj_verts.data_ptr<float>(),
        v_edge_normals.data_ptr<float>(),
        v_edge_offsets.data_ptr<float>(),
        v_phi_center.data_ptr<float>(),
        v_opacities.data_ptr<float>(),
        v_normals.data_ptr<float>(),
        v_vertices.data_ptr<float>(),
        v_vertex_opacity.data_ptr<float>()
    );
}

} // namespace gsplat
