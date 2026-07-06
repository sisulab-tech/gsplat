// Forward projection kernel for the opaque-triangle (MeshSplatting) primitive.
//
// Clean-room re-derivation (Apache-2.0) of the screen-space triangle-window
// representation described in arXiv:2512.06818, mapped onto gsplat's own
// pinhole/view conventions (posW2C + K-based pixel projection). This is NOT a
// port of the INRIA-licensed `diff-triangle-mesh-rasterization` CUDA; it mirrors
// the pure-PyTorch reference `sisu_040/gsplat/triangle_ref.py::project_triangles`
// so that the two agree bit-for-bit (modulo float precision).
//
// One thread handles one (camera, triangle). It gathers the triangle's 3 shared
// vertices, projects them, and emits the per-triangle quantities the rasterizer
// consumes: projected vertices, three inward edge half-planes, the incenter
// window normalization, min-opacity, camera-facing normal, centroid depth, and a
// square pixel-radius bbox for tile binning.

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Triangle.h"
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

__global__ void projection_triangle_fwd_kernel(
    const uint32_t C, // number of cameras
    const uint32_t T, // number of triangles
    const uint32_t V, // number of vertices
    const float *__restrict__ vertices,       // [V, 3]
    const int32_t *__restrict__ faces,        // [T, 3]
    const float *__restrict__ vertex_opacity, // [V]  (pre-activation logits)
    const float *__restrict__ viewmats,       // [C, 4, 4]  world -> camera
    const float *__restrict__ Ks,             // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float near_plane,
    const float far_plane,
    const float eps,
    // outputs (all [C, T, ...])
    int32_t *__restrict__ radii,        // [C, T, 2]
    float *__restrict__ means2d,        // [C, T, 2]
    float *__restrict__ depths,         // [C, T]
    float *__restrict__ vertex_depths,  // [C, T, 3]  per-vertex view-z
    float *__restrict__ proj_verts,     // [C, T, 3, 2]
    float *__restrict__ edge_normals,   // [C, T, 3, 2]
    float *__restrict__ edge_offsets,   // [C, T, 3]
    float *__restrict__ phi_center,     // [C, T]
    float *__restrict__ opacities,      // [C, T]
    float *__restrict__ normals         // [C, T, 3]
) {
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * T) {
        return;
    }
    const uint32_t cid = idx / T; // camera id
    const uint32_t tid = idx % T; // triangle id

    // By default this triangle is culled (0 radius => skipped by isect_tiles).
    radii[idx * 2] = 0;
    radii[idx * 2 + 1] = 0;

    viewmats += cid * 16;
    Ks += cid * 9;

    // glm is column-major but input is row-major: transpose explicitly.
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

    const vec3 w0 = vec3(vertices[i0 * 3], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]);
    const vec3 w1 = vec3(vertices[i1 * 3], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]);
    const vec3 w2 = vec3(vertices[i2 * 3], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]);

    // World -> camera, then pinhole projection to pixels (z clamped to eps).
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

    // Centroid depth (in camera space) and projected-vertex centroid.
    const vec3 centroid_cam = (c0 + c1 + c2) / 3.0f;
    const float depth = centroid_cam.z;
    const vec2 mean2d = (A + B + Cc) / 3.0f;

    // Signed area determinant of the projected triangle (degeneracy test).
    const float denom_area =
        (B.x - A.x) * (Cc.y - A.y) - (Cc.x - A.x) * (B.y - A.y);
    const bool valid = (depth > near_plane) && (depth < far_plane) &&
                       (fabsf(denom_area) > eps);
    if (!valid) {
        return;
    }

    // 2D incenter, weighted by opposite side lengths (a opposite A, ...).
    const float a = glm::length(B - Cc);
    const float b = glm::length(A - Cc);
    const float c = glm::length(A - B);
    const float wsum = max(a + b + c, eps);
    const vec2 incenter = (a * A + b * B + c * Cc) / wsum;

    // Three edge half-planes (v0->v1, v1->v2, v2->v0), oriented inward via the
    // incenter sign so that the interior satisfies n . p + off <= 0.
    const vec2 P[3] = {A, B, Cc};
    float dist_incenter = 0.0f;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const vec2 p1 = P[k];
        const vec2 p2 = P[(k + 1) % 3];
        float nx = p2.y - p1.y;
        float ny = -(p2.x - p1.x);
        const float nrm = max(sqrtf(nx * nx + ny * ny), eps);
        nx /= nrm;
        ny /= nrm;
        float off = -(nx * p1.x + ny * p1.y);
        float d = nx * incenter.x + ny * incenter.y + off;
        const float sign = (d > 0.0f) ? -1.0f : 1.0f;
        nx *= sign;
        ny *= sign;
        off *= sign;
        d *= sign;
        edge_normals[idx * 6 + k * 2 + 0] = nx;
        edge_normals[idx * 6 + k * 2 + 1] = ny;
        edge_offsets[idx * 3 + k] = off;
        dist_incenter = d; // equidistant for all edges; last one is fine
    }
    // Window normalization so the window == 1 at the incenter (dist < 0).
    const float phi = 1.0f / min(dist_incenter, -eps);

    // Per-triangle opacity = min activated vertex weight.
    const float o0 = 1.0f / (1.0f + __expf(-vertex_opacity[i0]));
    const float o1 = 1.0f / (1.0f + __expf(-vertex_opacity[i1]));
    const float o2 = 1.0f / (1.0f + __expf(-vertex_opacity[i2]));
    const float opac = min(o0, min(o1, o2));
    // Reference preprocess cull (forward.cu:174,273): triangles whose min
    // activated opacity is below stopping_influence never rasterize — no
    // compositing, no stats, no gradient. They are frozen until revived by
    // a neighboring vertex or pruned.
    if (opac < 0.01f) {
        return;
    }

    // Camera-facing world-space normal (for the normal map / regularizers).
    vec3 nrmw = glm::cross(w1 - w0, w2 - w0);
    nrmw = nrmw / max(glm::length(nrmw), eps);
    const vec3 normal_cam = R * nrmw;
    const vec3 cdir = centroid_cam / max(glm::length(centroid_cam), eps);
    const float cos_theta = glm::dot(normal_cam, cdir);
    // Reference preprocess cull (forward.cu:268-271): near-edge-on triangles
    // (|cos| < 0.001 between view direction and normal) are culled outright.
    if (fabsf(cos_theta) < 0.001f) {
        return;
    }
    if (cos_theta > 0.0f) {
        nrmw = -nrmw;
    }

    // Reference size culls (forward.cu:356): triangles whose max
    // vertex-to-center pixel distance exceeds 1600 (giant), is below 1
    // (sub-pixel), or whose incircle radius is under 1 pixel (skinny —
    // dist > -1 with the negative-inside convention) never rasterize.
    const float dA = glm::length(A - mean2d);
    const float dB = glm::length(B - mean2d);
    const float dCc = glm::length(Cc - mean2d);
    const float max_distance = max(dA, max(dB, dCc));
    if (max_distance > 1600.0f || max_distance < 1.0f ||
        dist_incenter > -1.0f) {
        return;
    }

    // Square pixel-radius bbox around the projected-vertex centroid.
    const float rx = ceilf(max(
        max(fabsf(A.x - mean2d.x), fabsf(B.x - mean2d.x)), fabsf(Cc.x - mean2d.x)
    ));
    const float ry = ceilf(max(
        max(fabsf(A.y - mean2d.y), fabsf(B.y - mean2d.y)), fabsf(Cc.y - mean2d.y)
    ));
    if (mean2d.x + rx <= 0 || mean2d.x - rx >= image_width ||
        mean2d.y + ry <= 0 || mean2d.y - ry >= image_height) {
        return; // fully off-screen; radii already 0
    }

    // Write outputs.
    radii[idx * 2] = (int32_t)rx;
    radii[idx * 2 + 1] = (int32_t)ry;
    means2d[idx * 2] = mean2d.x;
    means2d[idx * 2 + 1] = mean2d.y;
    depths[idx] = depth;
    // Per-vertex camera-space z (raw, unclamped) for the rasterizer's
    // barycentric depth interpolation. Matches the reference `vertex_depth`
    // (forward.cu:121 `vertex_depth[idx] = p_view.z`), which the expected- and
    // median-depth maps interpolate from — NOT the centroid `depth` above,
    // which is only the front-to-back sort key.
    vertex_depths[idx * 3 + 0] = c0.z;
    vertex_depths[idx * 3 + 1] = c1.z;
    vertex_depths[idx * 3 + 2] = c2.z;
    proj_verts[idx * 6 + 0] = A.x;
    proj_verts[idx * 6 + 1] = A.y;
    proj_verts[idx * 6 + 2] = B.x;
    proj_verts[idx * 6 + 3] = B.y;
    proj_verts[idx * 6 + 4] = Cc.x;
    proj_verts[idx * 6 + 5] = Cc.y;
    phi_center[idx] = phi;
    opacities[idx] = opac;
    normals[idx * 3 + 0] = nrmw.x;
    normals[idx * 3 + 1] = nrmw.y;
    normals[idx * 3 + 2] = nrmw.z;
}

void launch_projection_triangle_fwd_kernel(
    // inputs
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
    // outputs
    at::Tensor radii,         // [C, T, 2]
    at::Tensor means2d,       // [C, T, 2]
    at::Tensor depths,        // [C, T]
    at::Tensor vertex_depths, // [C, T, 3]
    at::Tensor proj_verts,    // [C, T, 3, 2]
    at::Tensor edge_normals,  // [C, T, 3, 2]
    at::Tensor edge_offsets,  // [C, T, 3]
    at::Tensor phi_center,    // [C, T]
    at::Tensor opacities,     // [C, T]
    at::Tensor normals        // [C, T, 3]
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

    projection_triangle_fwd_kernel<<<
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
        radii.data_ptr<int32_t>(),
        means2d.data_ptr<float>(),
        depths.data_ptr<float>(),
        vertex_depths.data_ptr<float>(),
        proj_verts.data_ptr<float>(),
        edge_normals.data_ptr<float>(),
        edge_offsets.data_ptr<float>(),
        phi_center.data_ptr<float>(),
        opacities.data_ptr<float>(),
        normals.data_ptr<float>()
    );
}

} // namespace gsplat
