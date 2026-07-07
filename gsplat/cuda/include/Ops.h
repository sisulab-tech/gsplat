// A collection of operators for gsplat
#pragma once

#include <ATen/core/Tensor.h>

#include "Cameras.h"
#include "Common.h"

namespace gsplat {

// null operator for tutorial. Does nothing.
at::Tensor null(const at::Tensor input);

// Project 3D gaussians (in camera space) to 2D image planes with EWA splatting.
std::tuple<at::Tensor, at::Tensor> projection_ewa_simple_fwd(
    const at::Tensor means,  // [C, N, 3]
    const at::Tensor covars, // [C, N, 3, 3]
    const at::Tensor Ks,     // [C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model
);
std::tuple<at::Tensor, at::Tensor> projection_ewa_simple_bwd(
    const at::Tensor means,  // [C, N, 3]
    const at::Tensor covars, // [C, N, 3, 3]
    const at::Tensor Ks,     // [C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model,
    const at::Tensor v_means2d, // [C, N, 2]
    const at::Tensor v_covars2d // [C, N, 2, 2]
);

// Fuse the following operations:
// 1. compute covar from {quats, scales}
// 2. transform 3D gaussians from world space to camera space
//    - w/ near far plane check
// 3. projection camera space 3D gaussians to 2D image planes with EWA
// splatting.
//    - w/ minimum radius check
// 4. add a bit blurring to the 2D gaussians for anti-aliasing.
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
projection_ewa_3dgs_fused_fwd(
    const at::Tensor means,                   // [N, 3]
    const at::optional<at::Tensor> covars,    // [N, 6] optional
    const at::optional<at::Tensor> quats,     // [N, 4] optional
    const at::optional<at::Tensor> scales,    // [N, 3] optional
    const at::optional<at::Tensor> opacities, // [N] optional
    const at::Tensor viewmats,                // [C, 4, 4]
    const at::Tensor Ks,                      // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool calc_compensations,
    const CameraModelType camera_model
);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_ewa_3dgs_fused_bwd(
    // fwd inputs
    const at::Tensor means,                // [N, 3]
    const at::optional<at::Tensor> covars, // [N, 6] optional
    const at::optional<at::Tensor> quats,  // [N, 4] optional
    const at::optional<at::Tensor> scales, // [N, 3] optional
    const at::Tensor viewmats,             // [C, 4, 4]
    const at::Tensor Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const at::Tensor radii,                       // [C, N, 2]
    const at::Tensor conics,                      // [C, N, 3]
    const at::optional<at::Tensor> compensations, // [C, N] optional
    // grad outputs
    const at::Tensor v_means2d,                     // [C, N, 2]
    const at::Tensor v_depths,                      // [C, N]
    const at::Tensor v_conics,                      // [C, N, 3]
    const at::optional<at::Tensor> v_compensations, // [C, N] optional
    const bool viewmats_requires_grad
);

// On top of fusing the operations like `projection_ewa_3dgs_fused_{fwd, bwd}`,
// The packed version compresses the [C, N, D] tensors (both intermidiate and
// output) into a jagged format [nnz, D], leveraging the sparsity of these
// tensors.
//
// This could lead to less memory usage than `_fused_{fwd, bwd}` if the level of
// sparsity is high, i.e., most of the gaussians are not in the camera frustum.
// But at the cost of slightly slower speed.
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
projection_ewa_3dgs_packed_fwd(
    const at::Tensor means,                   // [N, 3]
    const at::optional<at::Tensor> covars,    // [N, 6] optional
    const at::optional<at::Tensor> quats,     // [N, 4] optional
    const at::optional<at::Tensor> scales,    // [N, 3] optional
    const at::optional<at::Tensor> opacities, // [N] optional
    const at::Tensor viewmats,                // [C, 4, 4]
    const at::Tensor Ks,                      // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool calc_compensations,
    const CameraModelType camera_model
);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_ewa_3dgs_packed_bwd(
    // fwd inputs
    const at::Tensor means,                // [N, 3]
    const at::optional<at::Tensor> covars, // [N, 6]
    const at::optional<at::Tensor> quats,  // [N, 4]
    const at::optional<at::Tensor> scales, // [N, 3]
    const at::Tensor viewmats,             // [C, 4, 4]
    const at::Tensor Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const at::Tensor camera_ids,                  // [nnz]
    const at::Tensor gaussian_ids,                // [nnz]
    const at::Tensor conics,                      // [nnz, 3]
    const at::optional<at::Tensor> compensations, // [nnz] optional
    // grad outputs
    const at::Tensor v_means2d,                     // [nnz, 2]
    const at::Tensor v_depths,                      // [nnz]
    const at::Tensor v_conics,                      // [nnz, 3]
    const at::optional<at::Tensor> v_compensations, // [nnz] optional
    const bool viewmats_requires_grad,
    const bool sparse_grad
);

// Sphereical harmonics
at::Tensor spherical_harmonics_fwd(
    const uint32_t degrees_to_use,
    const at::Tensor dirs,               // [..., 3]
    const at::Tensor coeffs,             // [..., K, 3]
    const at::optional<at::Tensor> masks // [...]
);
std::tuple<at::Tensor, at::Tensor> spherical_harmonics_bwd(
    const uint32_t K,
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., 3]
    const at::Tensor coeffs,              // [..., K, 3]
    const at::optional<at::Tensor> masks, // [...]
    const at::Tensor v_colors,            // [..., 3]
    bool compute_v_dirs
);

// Fused Adam that supports a valid mask to skip updating certain parameters.
// Note skipping is not equivalent with zeroing out the gradients, which will
// still update parameters with momentum.
void adam(
    at::Tensor &param,                    // [..., D]
    const at::Tensor &param_grad,         // [..., D]
    at::Tensor &exp_avg,                  // [..., D]
    at::Tensor &exp_avg_sq,               // [..., D]
    const at::optional<at::Tensor> valid, // [...]
    const float lr,
    const float b1,
    const float b2,
    const float eps
);

// GS Tile Intersection
std::tuple<at::Tensor, at::Tensor, at::Tensor> intersect_tile(
    const at::Tensor means2d,                    // [C, N, 2] or [nnz, 2]
    const at::Tensor radii,                      // [C, N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [C, N] or [nnz]
    const at::optional<at::Tensor> camera_ids,   // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t C,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const bool sort
);
at::Tensor intersect_offset(
    const at::Tensor isect_ids, // [n_isects]
    const uint32_t C,
    const uint32_t tile_width,
    const uint32_t tile_height
);

// Compute Covariance and Precision Matrices from Quaternion and Scale
std::tuple<at::Tensor, at::Tensor> quat_scale_to_covar_preci_fwd(
    const at::Tensor quats,  // [N, 4]
    const at::Tensor scales, // [N, 3]
    const bool compute_covar,
    const bool compute_preci,
    const bool triu
);
std::tuple<at::Tensor, at::Tensor> quat_scale_to_covar_preci_bwd(
    const at::Tensor quats,  // [N, 4]
    const at::Tensor scales, // [N, 3]
    const bool triu,
    const at::optional<at::Tensor> v_covars, // [N, 3, 3] or [N, 6]
    const at::optional<at::Tensor> v_precis  // [N, 3, 3] or [N, 6]
);

// Rasterize 3D Gaussian to pixels
std::tuple<at::Tensor, at::Tensor, at::Tensor> rasterize_to_pixels_3dgs_fwd(
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
    const at::Tensor flatten_ids   // [n_isects]
);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_3dgs_bwd(
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
    // options
    bool absgrad
);

// Rasterize 3D Gaussians to pixels with RaDe-GS geometry outputs
// (expected/median plane depth, GOF distortion). Returns
// (renders, alphas, last_ids, render_edepths, render_mdepths, median_ids,
//  render_distorts, dist_accums).
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
rasterize_to_pixels_3dgs_geom_fwd(
    // Gaussian parameters
    const at::Tensor means2d,    // [C, N, 2] or [nnz, 2]
    const at::Tensor conics,     // [C, N, 3] or [nnz, 3]
    const at::Tensor colors,     // [C, N, channels] or [nnz, channels]
    const at::Tensor opacities,  // [C, N]  or [nnz]
    const at::Tensor ray_planes, // [C, N, 3] or [nnz, 3]
    const at::Tensor Ks,         // [C, 3, 3]
    const double distort_near,
    const double distort_far,
    const at::optional<at::Tensor> backgrounds, // [C, channels]
    const at::optional<at::Tensor> masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [C, tile_height, tile_width]
    const at::Tensor flatten_ids   // [n_isects]
);
// Returns (v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities,
// v_ray_planes).
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_3dgs_geom_bwd(
    // Gaussian parameters
    const at::Tensor means2d,    // [C, N, 2] or [nnz, 2]
    const at::Tensor conics,     // [C, N, 3] or [nnz, 3]
    const at::Tensor colors,     // [C, N, CDIM] or [nnz, CDIM]
    const at::Tensor opacities,  // [C, N] or [nnz]
    const at::Tensor ray_planes, // [C, N, 3] or [nnz, 3]
    const at::Tensor Ks,         // [C, 3, 3]
    const double distort_near,
    const double distort_far,
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
    // options
    bool absgrad
);

// Rasterize 3D Gaussians to pixels with GOF eval3d geometry outputs:
// color/alpha/normal/expected/median depth/distortion composited with the
// peak 3D response along camera-space pixel rays (view2gaussians quadric).
// Returns (renders, alphas, last_ids, render_normals, render_edepths,
// render_mdepths, median_ids, render_distorts, dist_accums).
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
rasterize_to_pixels_eval3d_geom_fwd(
    // Gaussian parameters
    const at::Tensor view2gaussians, // [C, N, 10]
    const at::Tensor colors,         // [C, N, channels]
    const at::Tensor opacities,      // [C, N]
    const at::Tensor Ks,             // [C, 3, 3]
    const double distort_near,
    const double distort_far,
    const at::optional<at::Tensor> backgrounds, // [C, channels]
    const at::optional<at::Tensor> masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [C, tile_height, tile_width]
    const at::Tensor flatten_ids   // [n_isects]
);
// Returns (v_view2gaussians, v_colors, v_opacities, v_means2d,
// v_means2d_abs). v_means2d / v_means2d_abs are GOF's conic-based
// densification signal only, never chained into parameter gradients.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_eval3d_geom_bwd(
    // Gaussian parameters
    const at::Tensor view2gaussians, // [C, N, 10]
    const at::Tensor means2d,        // [C, N, 2]
    const at::Tensor conics,         // [C, N, 3]
    const at::Tensor colors,         // [C, N, CDIM]
    const at::Tensor opacities,      // [C, N]
    const at::Tensor Ks,             // [C, 3, 3]
    const double distort_near,
    const double distort_far,
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
    const at::Tensor v_render_distorts // [C, image_height, image_width, 1]
);

// Rasterize 3D Gaussian, but only return the indices of gaussians and pixels.
std::tuple<at::Tensor, at::Tensor> rasterize_to_indices_3dgs(
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
    const at::Tensor flatten_ids   // [n_isects]
);

// Relocate some Gaussians in the Densification Process.
// Equation (9) in "3D Gaussian Splatting as Markov Chain Monte Carlo"
std::tuple<at::Tensor, at::Tensor> relocation(
    at::Tensor opacities, // [N]
    at::Tensor scales,    // [N, 3]
    at::Tensor ratios,    // [N]
    at::Tensor binoms,    // [n_max, n_max]
    const int n_max
);

// Projection for 2DGS
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
projection_2dgs_fused_fwd(
    const at::Tensor means,    // [N, 3]
    const at::Tensor quats,    // [N, 4]
    const at::Tensor scales,   // [N, 3]
    const at::Tensor viewmats, // [C, 4, 4]
    const at::Tensor Ks,       // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip
);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_2dgs_fused_bwd(
    // fwd inputs
    const at::Tensor means,    // [N, 3]
    const at::Tensor quats,    // [N, 4]
    const at::Tensor scales,   // [N, 3]
    const at::Tensor viewmats, // [C, 4, 4]
    const at::Tensor Ks,       // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    // fwd outputs
    const at::Tensor radii,          // [C, N, 2]
    const at::Tensor ray_transforms, // [C, N, 3, 3]
    // grad outputs
    const at::Tensor v_means2d,        // [C, N, 2]
    const at::Tensor v_depths,         // [C, N]
    const at::Tensor v_normals,        // [C, N, 3]
    const at::Tensor v_ray_transforms, // [C, N, 3, 3]
    const bool viewmats_requires_grad
);

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
projection_2dgs_packed_fwd(
    const at::Tensor means,    // [N, 3]
    const at::Tensor quats,    // [N, 4]
    const at::Tensor scales,   // [N, 3]
    const at::Tensor viewmats, // [C, 4, 4]
    const at::Tensor Ks,       // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float near_plane,
    const float far_plane,
    const float radius_clip
);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_2dgs_packed_bwd(
    // fwd inputs
    const at::Tensor means,    // [N, 3]
    const at::Tensor quats,    // [N, 4]
    const at::Tensor scales,   // [N, 3]
    const at::Tensor viewmats, // [C, 4, 4]
    const at::Tensor Ks,       // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    // fwd outputs
    const at::Tensor camera_ids,     // [nnz]
    const at::Tensor gaussian_ids,   // [nnz]
    const at::Tensor ray_transforms, // [nnz, 3, 3]
    // grad outputs
    const at::Tensor v_means2d,        // [nnz, 2]
    const at::Tensor v_depths,         // [nnz]
    const at::Tensor v_ray_transforms, // [nnz, 3, 3]
    const at::Tensor v_normals,        // [nnz, 3]
    const bool viewmats_requires_grad,
    const bool sparse_grad
);

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
rasterize_to_pixels_2dgs_fwd(
    // Gaussian parameters
    const at::Tensor means2d,        // [C, N, 2] or [nnz, 2]
    const at::Tensor ray_transforms, // [C, N, 3] or [nnz, 3]
    const at::Tensor colors,         // [C, N, channels] or [nnz, channels]
    const at::Tensor opacities,      // [C, N]  or [nnz]
    const at::Tensor normals,        // [C, N, 3] or [nnz, 3]
    const at::optional<at::Tensor> backgrounds, // [C, channels]
    const at::optional<at::Tensor> masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [C, tile_height, tile_width]
    const at::Tensor flatten_ids   // [n_isects]
);
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
rasterize_to_pixels_2dgs_bwd(
    // Gaussian parameters
    const at::Tensor means2d,        // [C, N, 2] or [nnz, 2]
    const at::Tensor ray_transforms, // [C, N, 3, 3] or [nnz, 3, 3]
    const at::Tensor colors,         // [C, N, 3] or [nnz, 3]
    const at::Tensor opacities,      // [C, N] or [nnz]
    const at::Tensor normals,        // [C, N, 3] or [nnz, 3]
    const at::Tensor densify,
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
    const at::Tensor render_colors, // [C, image_height, image_width, COLOR_DIM]
    const at::Tensor render_alphas, // [C, image_height, image_width, 1]
    const at::Tensor last_ids,      // [C, image_height, image_width]
    const at::Tensor median_ids,    // [C, image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors,  // [C, image_height, image_width, 3]
    const at::Tensor v_render_alphas,  // [C, image_height, image_width, 1]
    const at::Tensor v_render_normals, // [C, image_height, image_width, 3]
    const at::Tensor v_render_distort, // [C, image_height, image_width, 1]
    const at::Tensor v_render_median,  // [C, image_height, image_width, 1]
    // options
    bool absgrad
);

std::tuple<at::Tensor, at::Tensor> rasterize_to_indices_2dgs(
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
    const at::Tensor flatten_ids   // [n_isects]
);

// Use uncented transform to project 3D gaussians to 2D. (none differentiable)
// https://arxiv.org/abs/2412.12507
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
projection_ut_3dgs_fused(
    const at::Tensor means,                   // [N, 3]
    const at::Tensor quats,                   // [N, 4]
    const at::Tensor scales,                  // [N, 3]
    const at::optional<at::Tensor> opacities, // [N] optional
    const at::Tensor viewmats0,               // [C, 4, 4]
    const at::optional<at::Tensor>
        viewmats1,       // [C, 4, 4] optional for rolling shutter
    const at::Tensor Ks, // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool calc_compensations,
    const CameraModelType camera_model,
    // uncented transform
    const UnscentedTransformParameters ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> radial_coeffs, // [C, 6] or [C, 4] optional
    const at::optional<at::Tensor> tangential_coeffs, // [C, 2] optional
    const at::optional<at::Tensor> thin_prism_coeffs  // [C, 2] optional
);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_from_world_3dgs_fwd(
    // Gaussian parameters
    const at::Tensor means,     // [N, 3]
    const at::Tensor quats,     // [N, 4]
    const at::Tensor scales,    // [N, 3]
    const at::Tensor colors,    // [C, N, channels] or [nnz, channels]
    const at::Tensor opacities, // [C, N]  or [nnz]
    const at::optional<at::Tensor> backgrounds, // [C, channels]
    const at::optional<at::Tensor> masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // camera
    const at::Tensor viewmats0, // [C, 4, 4]
    const at::optional<at::Tensor>
        viewmats1,       // [C, 4, 4] optional for rolling shutter
    const at::Tensor Ks, // [C, 3, 3]
    const CameraModelType camera_model,
    // uncented transform
    const UnscentedTransformParameters ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> radial_coeffs, // [C, 6] or [C, 4] optional
    const at::optional<at::Tensor> tangential_coeffs, // [C, 2] optional
    const at::optional<at::Tensor> thin_prism_coeffs, // [C, 2] optional
    // intersections
    const at::Tensor tile_offsets, // [C, tile_height, tile_width]
    const at::Tensor flatten_ids   // [n_isects]
);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_from_world_3dgs_bwd(
    // Gaussian parameters
    const at::Tensor means,                     // [N, 3]
    const at::Tensor quats,                     // [N, 4]
    const at::Tensor scales,                    // [N, 3]
    const at::Tensor colors,                    // [C, N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [C, N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [C, 3]
    const at::optional<at::Tensor> masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // camera
    const at::Tensor viewmats0, // [C, 4, 4]
    const at::optional<at::Tensor>
        viewmats1,       // [C, 4, 4] optional for rolling shutter
    const at::Tensor Ks, // [C, 3, 3]
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
    const at::Tensor v_render_alphas  // [C, image_height, image_width, 1]
);

at::Tensor compute_3D_smoothing_filter_fwd_tensor(
    const at::Tensor means,    // [N, 3]
    const at::Tensor viewmats, // [C, 4, 4]
    const at::Tensor Ks,       // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float near_plane
);

std::tuple<at::Tensor, at::Tensor, at::Tensor> project_points_fwd_tensor(
    const at::Tensor means,    // [N, 3]
    const at::Tensor viewmats, // [C, 4, 4]
    const at::Tensor Ks,       // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float near_plane,
    const float far_plane
);

std::tuple<at::Tensor, at::Tensor> points_isect_tiles_tensor(
    const at::Tensor means2d,                    // [C, N, 2] or [nnz, 2]
    const at::Tensor radii,                      // [C, N] or [nnz]
    const at::Tensor depths,                     // [C, N] or [nnz]
    const at::optional<at::Tensor> camera_ids,   // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t C,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const bool sort,
    const bool double_buffer
);

std::tuple<at::Tensor, at::Tensor> integrate_to_points_fwd_tensor(
    // Point parameters
    const at::Tensor points2d,     // [C, N, 2]
    const at::Tensor point_depths, // [C, N, 3]
    // Gaussian parameters
    const at::Tensor means2d,                   // [C, N, 2]
    const at::Tensor conics,                    // [C, N, 3]
    const at::Tensor colors,                    // [C, N, D]
    const at::Tensor opacities,                 // [N]
    const at::Tensor view2gaussians,            // [C, N, 10]
    const at::Tensor Ks,                        // [C, 3, 3]
    const at::optional<at::Tensor> backgrounds, // [C, D]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [C, tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // points intersections
    const at::Tensor point_tile_offsets, // [C, tile_height, tile_width]
    const at::Tensor point_flatten_ids   // [n_isects]
);

at::Tensor view_to_gaussians_fwd_tensor(
    const at::Tensor means,    // [N, 3]
    const at::Tensor quats,    // [N, 4]
    const at::Tensor scales,   // [N, 3]
    const at::Tensor viewmats, // [C, 4, 4]
    const at::Tensor radii     // [C, N]
);

at::Tensor py_triangulate(const at::Tensor &points);

// Projection + rasterization for the opaque-triangle (MeshSplatting) primitive.
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
);

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
);

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
    const at::Tensor v_means2d,       // [C, T, 2]
    const at::Tensor v_depths,        // [C, T]
    const at::Tensor v_vertex_depths, // [C, T, 3]
    const at::Tensor v_proj_verts,    // [C, T, 3, 2]
    const at::Tensor v_edge_normals,  // [C, T, 3, 2]
    const at::Tensor v_edge_offsets,  // [C, T, 3]
    const at::Tensor v_phi_center,    // [C, T]
    const at::Tensor v_opacities,     // [C, T]
    const at::Tensor v_normals        // [C, T, 3]
);

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
    const at::Tensor tile_offsets,     // [C, tile_height, tile_width]
    const at::Tensor flatten_ids,      // [n_isects]
    const at::Tensor render_alphas,    // [C, H, W, 1]
    const at::Tensor last_ids,         // [C, H, W]
    const at::Tensor median_ids,       // [C, H, W]
    const at::Tensor v_render_colors,  // [C, H, W, CDIM]
    const at::Tensor v_render_alphas,  // [C, H, W, 1]
    const at::Tensor v_render_normals, // [C, H, W, 3]
    const at::Tensor v_render_depths,  // [C, H, W, 1]
    const at::Tensor v_render_median   // [C, H, W, 1]
);

} // namespace gsplat
