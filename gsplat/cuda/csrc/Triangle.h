#pragma once

#include <cstdint>

namespace at {
class Tensor;
} // namespace at

namespace gsplat {

/////////////////////////////////////////////////
// projection_triangle  (MeshSplatting primitive)
/////////////////////////////////////////////////

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
);

void launch_projection_triangle_bwd_kernel(
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
);

/////////////////////////////////////////////////
// rasterize_to_pixels_triangle
/////////////////////////////////////////////////

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
);

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
);

} // namespace gsplat
