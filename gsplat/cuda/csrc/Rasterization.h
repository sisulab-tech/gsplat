#pragma once

#include <cstdint>
#include "Cameras.h"

namespace at {
class Tensor;
}

namespace gsplat {

#define FILTER_INV_SQUARE_2DGS 2.0f

/////////////////////////////////////////////////
// rasterize_to_pixels_3dgs
/////////////////////////////////////////////////

template <uint32_t CDIM>
void launch_rasterize_to_pixels_3dgs_fwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,   // [C, N, 2] or [nnz, 2]
    const at::Tensor conics,    // [C, N, 3] or [nnz, 3]
    const at::Tensor colors,    // [C, N, channels] or [nnz, channels]
    const at::Tensor opacities, // [C, N]  or [nnz]
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
    at::Tensor last_ids // [C, image_height, image_width]
);

template <uint32_t CDIM>
void launch_rasterize_to_pixels_3dgs_bwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,                   // [C, N, 2] or [nnz, 2]
    const at::Tensor conics,                    // [C, N, 3] or [nnz, 3]
    const at::Tensor colors,                    // [C, N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [C, N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [C, 3]
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
    // gradients of outputs
    const at::Tensor v_render_colors, // [C, image_height, image_width, 3]
    const at::Tensor v_render_alphas, // [C, image_height, image_width, 1]
    // outputs
    at::optional<at::Tensor> v_means2d_abs, // [C, N, 2] or [nnz, 2]
    at::Tensor v_means2d,                   // [C, N, 2] or [nnz, 2]
    at::Tensor v_conics,                    // [C, N, 3] or [nnz, 3]
    at::Tensor v_colors,                    // [C, N, 3] or [nnz, 3]
    at::Tensor v_opacities                  // [C, N] or [nnz]
);

/////////////////////////////////////////////////
// rasterize_to_pixels_3dgs_geom (RaDe-GS geometry outputs)
/////////////////////////////////////////////////

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
    at::Tensor render_edepths,  // [C, image_height, image_width, 1]
    at::Tensor render_mdepths,  // [C, image_height, image_width, 1]
    at::Tensor median_ids,      // [C, image_height, image_width]
    at::Tensor render_distorts, // [C, image_height, image_width, 1]
    at::Tensor dist_accums      // [C, image_height, image_width, 2]
);

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
    const at::Tensor v_render_colors,   // [C, image_height, image_width, CDIM]
    const at::Tensor v_render_alphas,   // [C, image_height, image_width, 1]
    const at::Tensor v_render_edepths,  // [C, image_height, image_width, 1]
    const at::Tensor v_render_mdepths,  // [C, image_height, image_width, 1]
    const at::Tensor v_render_distorts, // [C, image_height, image_width, 1]
    // outputs
    at::optional<at::Tensor> v_means2d_abs, // [C, N, 2] or [nnz, 2]
    at::Tensor v_means2d,                   // [C, N, 2] or [nnz, 2]
    at::Tensor v_conics,                    // [C, N, 3] or [nnz, 3]
    at::Tensor v_colors,                    // [C, N, CDIM] or [nnz, CDIM]
    at::Tensor v_opacities,                 // [C, N] or [nnz]
    at::Tensor v_ray_planes                 // [C, N, 3] or [nnz, 3]
);

/////////////////////////////////////////////////
// rasterize_to_pixels_eval3d_geom (GOF eval3d geometry outputs)
/////////////////////////////////////////////////

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
);

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
);

/////////////////////////////////////////////////
// rasterize_to_indices_3dgs
/////////////////////////////////////////////////

void launch_rasterize_to_indices_3dgs_kernel(
    const uint32_t range_start,
    const uint32_t range_end,          // iteration steps
    const float visibility_threshold,  // observe gate: T>threshold (0 = off)
    const at::Tensor transmittances, // [C, image_height, image_width]
    // Gaussian parameters
    const at::Tensor means2d,   // [C, N, 2]
    const at::Tensor conics,    // [C, N, 3]
    const at::Tensor opacities, // [C, N]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [C, tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // helper for double pass
    const at::optional<at::Tensor>
        chunk_starts, // [C, image_height, image_width]
    // outputs
    at::optional<at::Tensor> chunk_cnts,   // [C, image_height, image_width]
    at::optional<at::Tensor> gaussian_ids, // [n_elems]
    at::optional<at::Tensor> pixel_ids     // [n_elems]
);

/////////////////////////////////////////////////
// rasterize_to_pixels_2dgs
/////////////////////////////////////////////////

template <uint32_t CDIM>
void launch_rasterize_to_pixels_2dgs_fwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,        // [C, N, 2] or [nnz, 2]
    const at::Tensor ray_transforms, // [C, N, 3, 3] or [nnz, 3, 3]
    const at::Tensor colors,         // [C, N, channels] or [nnz, channels]
    const at::Tensor opacities,      // [C, N]  or [nnz]
    const at::Tensor normals,        // [C, N, 3]
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
    at::Tensor renders,        // [C, image_height, image_width, channels]
    at::Tensor alphas,         // [C, image_height, image_width, 1]
    at::Tensor render_normals, // [C, image_height, image_width, 3]
    at::Tensor render_distort, // [C, image_height, image_width, 1]
    at::Tensor render_median,  // [C, image_height, image_width, 1]
    at::Tensor last_ids,       // [C, image_height, image_width]
    at::Tensor median_ids      // [C, image_height, image_width]
);
template <uint32_t CDIM>
void launch_rasterize_to_pixels_2dgs_bwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,                   // [C, N, 2] or [nnz, 2]
    const at::Tensor ray_transforms,            // [C, N, 3, 3] or [nnz, 3, 3]
    const at::Tensor colors,                    // [C, N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [C, N] or [nnz]
    const at::Tensor normals,                   // [C, N, 3] or [nnz, 3]
    const at::Tensor densify,                   // [C, N, 2] or [nnz, 2]
    const at::optional<at::Tensor> backgrounds, // [C, 3]
    const at::optional<at::Tensor> masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // ray_crossions
    const at::Tensor tile_offsets, // [C, tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor render_colors, // [C, image_height, image_width, CDIM]
    const at::Tensor render_alphas, // [C, image_height, image_width, 1]
    const at::Tensor last_ids,      // [C, image_height, image_width]
    const at::Tensor median_ids,    // [C, image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors,  // [C, image_height, image_width, 3]
    const at::Tensor v_render_alphas,  // [C, image_height, image_width, 1]
    const at::Tensor v_render_normals, // [C, image_height, image_width, 3]
    const at::Tensor v_render_distort, // [C, image_height, image_width, 1]
    const at::Tensor v_render_median,  // [C, image_height, image_width, 1]
    // outputs
    at::optional<at::Tensor> v_means2d_abs, // [C, N, 2] or [nnz, 2]
    at::Tensor v_means2d,                   // [C, N, 2] or [nnz, 2]
    at::Tensor v_ray_transforms,            // [C, N, 3, 3] or [nnz, 3, 3]
    at::Tensor v_colors,                    // [C, N, 3] or [nnz, 3]
    at::Tensor v_opacities,                 // [C, N] or [nnz]
    at::Tensor v_normals,                   // [C, N, 3] or [nnz, 3]
    at::Tensor v_densify                    // [C, N, 2] or [nnz, 2]
);

/////////////////////////////////////////////////
// rasterize_to_indices_2dgs
/////////////////////////////////////////////////

void launch_rasterize_to_indices_2dgs_kernel(
    const uint32_t range_start,
    const uint32_t range_end,        // iteration steps
    const at::Tensor transmittances, // [C, image_height, image_width]
    // Gaussian parameters
    const at::Tensor means2d,        // [C, N, 2]
    const at::Tensor ray_transforms, // [C, N, 3, 3]
    const at::Tensor opacities,      // [C, N]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [C, tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // helper for double pass
    const at::optional<at::Tensor>
        chunk_starts, // [C, image_height, image_width]
    // outputs
    at::optional<at::Tensor> chunk_cnts,   // [C, image_height, image_width]
    at::optional<at::Tensor> gaussian_ids, // [n_elems]
    at::optional<at::Tensor> pixel_ids     // [n_elems]
);

///////////////////////////////////////////////////
// rasterize_to_pixels_from_world_3dgs
///////////////////////////////////////////////////

template <uint32_t CDIM>
void launch_rasterize_to_pixels_from_world_3dgs_fwd_kernel(
    // Gaussian parameters
    const at::Tensor means, // [N, 3]
    const at::Tensor quats, // [N, 4]
    const at::Tensor scales, // [N, 3]
    const at::Tensor colors,    // [C, N, channels] or [nnz, channels]
    const at::Tensor opacities, // [C, N]  or [nnz]
    const at::optional<at::Tensor> backgrounds, // [C, channels]
    const at::optional<at::Tensor> masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // camera
    const at::Tensor viewmats0,             // [C, 4, 4]
    const at::optional<at::Tensor> viewmats1, // [C, 4, 4] optional for rolling shutter
    const at::Tensor Ks,                   // [C, 3, 3]
    const CameraModelType camera_model,
    // uncented transform
    const UnscentedTransformParameters ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> radial_coeffs, // [C, 6] or [C, 4] optional
    const at::optional<at::Tensor> tangential_coeffs, // [C, 2] optional
    const at::optional<at::Tensor> thin_prism_coeffs, // [C, 2] optional
    // intersections
    const at::Tensor tile_offsets, // [C, tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // outputs
    at::Tensor renders, // [C, image_height, image_width, channels]
    at::Tensor alphas,  // [C, image_height, image_width]
    at::Tensor last_ids // [C, image_height, image_width]
);

template <uint32_t CDIM>
void launch_rasterize_to_pixels_from_world_3dgs_bwd_kernel(
    // Gaussian parameters
    const at::Tensor means, // [N, 3]
    const at::Tensor quats, // [N, 4]
    const at::Tensor scales, // [N, 3]
    const at::Tensor colors,                    // [C, N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [C, N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [C, 3]
    const at::optional<at::Tensor> masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // camera
    const at::Tensor viewmats0,             // [C, 4, 4]
    const at::optional<at::Tensor> viewmats1, // [C, 4, 4] optional for rolling shutter
    const at::Tensor Ks,                   // [C, 3, 3]
    const CameraModelType camera_model,
    // uncented transform
    const UnscentedTransformParameters ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> radial_coeffs, // [C, 6] or [C, 4] optional
    const at::optional<at::Tensor> tangential_coeffs, // [C, 2] optional
    const at::optional<at::Tensor> thin_prism_coeffs, // [C, 2] optional
    // intersections
    const at::Tensor tile_offsets, // [C, tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor render_alphas, // [C, image_height, image_width, 1]
    const at::Tensor last_ids,      // [C, image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors, // [C, image_height, image_width, 3]
    const at::Tensor v_render_alphas, // [C, image_height, image_width, 1]
    // outputs
    at::Tensor v_means,      // [N, 3]
    at::Tensor v_quats,      // [N, 4]
    at::Tensor v_scales,     // [N, 3]
    at::Tensor v_colors,                    // [C, N, 3] or [nnz, 3]
    at::Tensor v_opacities                  // [C, N] or [nnz]
) ;

} // namespace gsplat