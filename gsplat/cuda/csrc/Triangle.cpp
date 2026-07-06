#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD
#include <tuple>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "Common.h"   // where all the macros are defined
#include "Ops.h"      // a collection of all gsplat operators
#include "Triangle.h" // where the launch functions are declared

namespace gsplat {

std::tuple<
    at::Tensor, // radii
    at::Tensor, // means2d
    at::Tensor, // depths
    at::Tensor, // vertex_depths
    at::Tensor, // proj_verts
    at::Tensor, // edge_normals
    at::Tensor, // edge_offsets
    at::Tensor, // phi_center
    at::Tensor, // opacities
    at::Tensor> // normals
projection_triangle_fwd(
    const at::Tensor vertices,       // [V, 3]
    const at::Tensor faces,          // [T, 3]
    const at::Tensor vertex_opacity, // [V]
    const at::Tensor viewmats,       // [C, 4, 4]
    const at::Tensor Ks,             // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float near_plane,
    const float far_plane,
    const float eps
) {
    DEVICE_GUARD(vertices);
    CHECK_INPUT(vertices);
    CHECK_INPUT(faces);
    CHECK_INPUT(vertex_opacity);
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);

    uint32_t T = faces.size(0);     // number of triangles
    uint32_t C = viewmats.size(0);  // number of cameras

    auto opt = vertices.options();
    at::Tensor radii = at::empty({C, T, 2}, opt.dtype(at::kInt));
    at::Tensor means2d = at::empty({C, T, 2}, opt);
    at::Tensor depths = at::empty({C, T}, opt);
    at::Tensor vertex_depths = at::zeros({C, T, 3}, opt);
    at::Tensor proj_verts = at::empty({C, T, 3, 2}, opt);
    at::Tensor edge_normals = at::empty({C, T, 3, 2}, opt);
    at::Tensor edge_offsets = at::empty({C, T, 3}, opt);
    at::Tensor phi_center = at::empty({C, T}, opt);
    at::Tensor opacities = at::empty({C, T}, opt);
    at::Tensor normals = at::zeros({C, T, 3}, opt);

    launch_projection_triangle_fwd_kernel(
        vertices,
        faces,
        vertex_opacity,
        viewmats,
        Ks,
        image_width,
        image_height,
        near_plane,
        far_plane,
        eps,
        radii,
        means2d,
        depths,
        vertex_depths,
        proj_verts,
        edge_normals,
        edge_offsets,
        phi_center,
        opacities,
        normals
    );
    return std::make_tuple(
        radii,
        means2d,
        depths,
        vertex_depths,
        proj_verts,
        edge_normals,
        edge_offsets,
        phi_center,
        opacities,
        normals
    );
}

std::tuple<
    at::Tensor, // render_colors
    at::Tensor, // render_alphas
    at::Tensor, // render_normals
    at::Tensor, // render_depths
    at::Tensor, // render_median
    at::Tensor, // last_ids
    at::Tensor, // median_ids
    at::Tensor, // max_blending  [C, T]
    at::Tensor> // pixel_count   [C, T]
rasterize_to_pixels_triangle_fwd(
    const at::Tensor proj_verts,    // [C, T, 3, 2]
    const at::Tensor edge_normals,  // [C, T, 3, 2]
    const at::Tensor edge_offsets,  // [C, T, 3]
    const at::Tensor phi_center,    // [C, T]
    const at::Tensor opacities,     // [C, T]
    const at::Tensor colors,        // [C, T, 3, CDIM]
    const at::Tensor normals,       // [C, T, 3]
    const at::Tensor vertex_depths, // [C, T, 3]
    const double sigma,
    const double eps,
    const at::optional<at::Tensor> backgrounds, // [C, CDIM]
    const at::optional<at::Tensor> masks,       // [C, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const at::Tensor tile_offsets, // [C, tile_height, tile_width]
    const at::Tensor flatten_ids   // [n_isects]
) {
    DEVICE_GUARD(proj_verts);
    CHECK_INPUT(proj_verts);
    CHECK_INPUT(edge_normals);
    CHECK_INPUT(edge_offsets);
    CHECK_INPUT(phi_center);
    CHECK_INPUT(opacities);
    CHECK_INPUT(colors);
    CHECK_INPUT(normals);
    CHECK_INPUT(vertex_depths);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    if (backgrounds.has_value()) {
        CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }
    auto opt = proj_verts.options();

    uint32_t C = tile_offsets.size(0);
    uint32_t channels = colors.size(-1);

    at::Tensor render_colors =
        at::empty({C, image_height, image_width, channels}, opt);
    at::Tensor render_alphas =
        at::zeros({C, image_height, image_width, 1}, opt);
    at::Tensor render_normals =
        at::zeros({C, image_height, image_width, 3}, opt);
    at::Tensor render_depths =
        at::zeros({C, image_height, image_width, 1}, opt);
    at::Tensor render_median =
        at::zeros({C, image_height, image_width, 1}, opt);
    at::Tensor last_ids =
        at::empty({C, image_height, image_width}, opt.dtype(at::kInt));
    at::Tensor median_ids =
        at::empty({C, image_height, image_width}, opt.dtype(at::kInt));

    // Detached per-triangle stats (accumulated via atomics in the kernel).
    uint32_t T = proj_verts.size(1);
    at::Tensor max_blending = at::zeros({C, T}, opt);
    at::Tensor pixel_count = at::zeros({C, T}, opt.dtype(at::kInt));

#define __LAUNCH_KERNEL__(N)                                                   \
    case N:                                                                    \
        launch_rasterize_to_pixels_triangle_fwd_kernel<N>(                     \
            proj_verts,                                                        \
            edge_normals,                                                      \
            edge_offsets,                                                      \
            phi_center,                                                        \
            opacities,                                                         \
            colors,                                                            \
            normals,                                                           \
            vertex_depths,                                                     \
            (float)sigma,                                                      \
            (float)eps,                                                        \
            backgrounds,                                                       \
            masks,                                                             \
            image_width,                                                       \
            image_height,                                                      \
            tile_size,                                                         \
            tile_offsets,                                                      \
            flatten_ids,                                                       \
            render_colors,                                                     \
            render_alphas,                                                     \
            render_normals,                                                    \
            render_depths,                                                     \
            render_median,                                                     \
            last_ids,                                                          \
            median_ids,                                                        \
            max_blending,                                                      \
            pixel_count                                                        \
        );                                                                     \
        break;

    switch (channels) {
        __LAUNCH_KERNEL__(1)
        __LAUNCH_KERNEL__(2)
        __LAUNCH_KERNEL__(3)
        __LAUNCH_KERNEL__(4)
        __LAUNCH_KERNEL__(5)
        __LAUNCH_KERNEL__(8)
        __LAUNCH_KERNEL__(9)
        __LAUNCH_KERNEL__(16)
        __LAUNCH_KERNEL__(17)
        __LAUNCH_KERNEL__(32)
        __LAUNCH_KERNEL__(33)
    default:
        AT_ERROR("Unsupported number of channels: ", channels);
    }
#undef __LAUNCH_KERNEL__

    return std::make_tuple(
        render_colors,
        render_alphas,
        render_normals,
        render_depths,
        render_median,
        last_ids,
        median_ids,
        max_blending,
        pixel_count
    );
}

std::tuple<
    at::Tensor, // v_vertices
    at::Tensor> // v_vertex_opacity
projection_triangle_bwd(
    const at::Tensor vertices,       // [V, 3]
    const at::Tensor faces,          // [T, 3]
    const at::Tensor vertex_opacity, // [V]
    const at::Tensor viewmats,       // [C, 4, 4]
    const at::Tensor Ks,             // [C, 3, 3]
    const int64_t image_width,
    const int64_t image_height,
    const double near_plane,
    const double far_plane,
    const double eps,
    // grad outputs
    const at::Tensor v_means2d,       // [C, T, 2]
    const at::Tensor v_depths,        // [C, T]
    const at::Tensor v_vertex_depths, // [C, T, 3]
    const at::Tensor v_proj_verts,    // [C, T, 3, 2]
    const at::Tensor v_edge_normals,  // [C, T, 3, 2]
    const at::Tensor v_edge_offsets,  // [C, T, 3]
    const at::Tensor v_phi_center,    // [C, T]
    const at::Tensor v_opacities,     // [C, T]
    const at::Tensor v_normals        // [C, T, 3]
) {
    DEVICE_GUARD(vertices);
    CHECK_INPUT(vertices);
    CHECK_INPUT(faces);
    CHECK_INPUT(vertex_opacity);
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);
    CHECK_INPUT(v_means2d);
    CHECK_INPUT(v_depths);
    CHECK_INPUT(v_vertex_depths);
    CHECK_INPUT(v_proj_verts);
    CHECK_INPUT(v_edge_normals);
    CHECK_INPUT(v_edge_offsets);
    CHECK_INPUT(v_phi_center);
    CHECK_INPUT(v_opacities);
    CHECK_INPUT(v_normals);

    auto opt = vertices.options();
    at::Tensor v_vertices = at::zeros_like(vertices);
    at::Tensor v_vertex_opacity = at::zeros_like(vertex_opacity);

    launch_projection_triangle_bwd_kernel(
        vertices,
        faces,
        vertex_opacity,
        viewmats,
        Ks,
        (uint32_t)image_width,
        (uint32_t)image_height,
        (float)near_plane,
        (float)far_plane,
        (float)eps,
        v_means2d,
        v_depths,
        v_vertex_depths,
        v_proj_verts,
        v_edge_normals,
        v_edge_offsets,
        v_phi_center,
        v_opacities,
        v_normals,
        v_vertices,
        v_vertex_opacity
    );
    return std::make_tuple(v_vertices, v_vertex_opacity);
}

std::tuple<
    at::Tensor, // v_proj_verts
    at::Tensor, // v_edge_normals
    at::Tensor, // v_edge_offsets
    at::Tensor, // v_phi_center
    at::Tensor, // v_opacities
    at::Tensor, // v_colors
    at::Tensor, // v_normals
    at::Tensor> // v_vertex_depths
rasterize_to_pixels_triangle_bwd(
    const at::Tensor proj_verts,    // [C, T, 3, 2]
    const at::Tensor edge_normals,  // [C, T, 3, 2]
    const at::Tensor edge_offsets,  // [C, T, 3]
    const at::Tensor phi_center,    // [C, T]
    const at::Tensor opacities,     // [C, T]
    const at::Tensor colors,        // [C, T, 3, CDIM]
    const at::Tensor normals,       // [C, T, 3]
    const at::Tensor vertex_depths, // [C, T, 3]
    const double sigma,
    const double eps,
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
    const at::Tensor v_render_median   // [C, H, W, 1]
) {
    DEVICE_GUARD(proj_verts);
    CHECK_INPUT(proj_verts);
    CHECK_INPUT(edge_normals);
    CHECK_INPUT(edge_offsets);
    CHECK_INPUT(phi_center);
    CHECK_INPUT(opacities);
    CHECK_INPUT(colors);
    CHECK_INPUT(normals);
    CHECK_INPUT(vertex_depths);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    CHECK_INPUT(render_alphas);
    CHECK_INPUT(last_ids);
    CHECK_INPUT(median_ids);
    CHECK_INPUT(v_render_colors);
    CHECK_INPUT(v_render_alphas);
    CHECK_INPUT(v_render_normals);
    CHECK_INPUT(v_render_depths);
    CHECK_INPUT(v_render_median);
    if (backgrounds.has_value()) {
        CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }

    uint32_t channels = colors.size(-1);

    at::Tensor v_proj_verts = at::zeros_like(proj_verts);
    at::Tensor v_edge_normals = at::zeros_like(edge_normals);
    at::Tensor v_edge_offsets = at::zeros_like(edge_offsets);
    at::Tensor v_phi_center = at::zeros_like(phi_center);
    at::Tensor v_opacities = at::zeros_like(opacities);
    at::Tensor v_colors = at::zeros_like(colors);
    at::Tensor v_normals = at::zeros_like(normals);
    at::Tensor v_vertex_depths = at::zeros_like(vertex_depths);

#define __LAUNCH_KERNEL__(N)                                                   \
    case N:                                                                    \
        launch_rasterize_to_pixels_triangle_bwd_kernel<N>(                     \
            proj_verts,                                                        \
            edge_normals,                                                      \
            edge_offsets,                                                      \
            phi_center,                                                        \
            opacities,                                                         \
            colors,                                                            \
            normals,                                                           \
            vertex_depths,                                                     \
            (float)sigma,                                                      \
            (float)eps,                                                        \
            backgrounds,                                                       \
            masks,                                                             \
            image_width,                                                       \
            image_height,                                                      \
            tile_size,                                                         \
            tile_offsets,                                                      \
            flatten_ids,                                                       \
            render_alphas,                                                     \
            last_ids,                                                          \
            median_ids,                                                        \
            v_render_colors,                                                   \
            v_render_alphas,                                                   \
            v_render_normals,                                                  \
            v_render_depths,                                                   \
            v_render_median,                                                   \
            v_proj_verts,                                                      \
            v_edge_normals,                                                    \
            v_edge_offsets,                                                    \
            v_phi_center,                                                      \
            v_opacities,                                                       \
            v_colors,                                                          \
            v_normals,                                                         \
            v_vertex_depths                                                    \
        );                                                                     \
        break;

    switch (channels) {
        __LAUNCH_KERNEL__(1)
        __LAUNCH_KERNEL__(2)
        __LAUNCH_KERNEL__(3)
        __LAUNCH_KERNEL__(4)
        __LAUNCH_KERNEL__(5)
        __LAUNCH_KERNEL__(8)
        __LAUNCH_KERNEL__(9)
        __LAUNCH_KERNEL__(16)
        __LAUNCH_KERNEL__(17)
        __LAUNCH_KERNEL__(32)
        __LAUNCH_KERNEL__(33)
    default:
        AT_ERROR("Unsupported number of channels: ", channels);
    }
#undef __LAUNCH_KERNEL__

    return std::make_tuple(
        v_proj_verts,
        v_edge_normals,
        v_edge_offsets,
        v_phi_center,
        v_opacities,
        v_colors,
        v_normals,
        v_vertex_depths
    );
}

} // namespace gsplat
