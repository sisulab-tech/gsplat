#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "Common.h"
#include "Projection.h"
#include "Utils.cuh"

namespace gsplat {
namespace cg = cooperative_groups;

/****************************************************************************
 * Compact Box (CB) Implementation - Based on FastGS/Speedy-Splat
 *
 * This implements precise ellipse-tile intersection using Mahalanobis distance
 * filtering, with a configurable multiplier for the Compact Box technique.
 ****************************************************************************/

// Compute intersection of ellipse with a horizontal or vertical line
__device__ inline vec2 computeEllipseIntersection(
    const vec3 conic,     // [conic.x, conic.y, conic.z] = [a, b, c]
    const float opacity,
    const float disc,
    const float t,
    const vec2 p,
    const bool isY,
    const float coord
) {
    float p_u = isY ? p.y : p.x;
    float p_v = isY ? p.x : p.y;
    float coeff = isY ? conic.x : conic.z;

    float h = coord - p_u;  // h = y - p.y for y, x - p.x for x
    float sqrt_term = sqrt(disc * h * h + t * coeff);

    return vec2{
        (-conic.y * h - sqrt_term) / coeff + p_v,
        (-conic.y * h + sqrt_term) / coeff + p_v
    };
}

// Process tiles along a slice (either horizontal or vertical)
__device__ inline uint32_t processTiles(
    const vec3 conic,
    const float opacity,
    const float disc,
    const float t,
    const vec2 p,
    vec2 bbox_min,
    vec2 bbox_max,
    vec2 bbox_argmin,
    vec2 bbox_argmax,
    int2 rect_min,
    int2 rect_max,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_size,
    const bool isY,
    uint32_t idx,
    uint32_t off,
    float depth,
    int64_t cid_enc,
    int64_t *__restrict__ isect_ids,
    int32_t *__restrict__ flatten_ids
) {
    // Set variables based on the isY flag
    float BLOCK_U = isY ? (float)tile_size : (float)tile_size;
    float BLOCK_V = isY ? (float)tile_size : (float)tile_size;

    if (isY) {
        // Swap x and y
        rect_min = int2{rect_min.y, rect_min.x};
        rect_max = int2{rect_max.y, rect_max.x};
        bbox_min = vec2{bbox_min.y, bbox_min.x};
        bbox_max = vec2{bbox_max.y, bbox_max.x};
        bbox_argmin = vec2{bbox_argmin.y, bbox_argmin.x};
        bbox_argmax = vec2{bbox_argmax.y, bbox_argmax.x};
    }

    uint32_t tiles_count = 0;
    vec2 intersect_min_line, intersect_max_line;
    float ellipse_min, ellipse_max;
    float min_line, max_line;

    // Initialize max line
    intersect_max_line = vec2{bbox_max.y, bbox_min.y};

    min_line = rect_min.x * BLOCK_U;
    // Initialize min line intersections
    if (bbox_min.x <= min_line) {
        intersect_min_line = computeEllipseIntersection(
            conic, opacity, disc, t, p, isY, rect_min.x * BLOCK_U
        );
    } else {
        intersect_min_line = intersect_max_line;
    }

    // Loop over either y slices or x slices based on the isY flag
    for (int u = rect_min.x; u < rect_max.x; ++u) {
        max_line = min_line + BLOCK_U;
        if (max_line <= bbox_max.x) {
            intersect_max_line = computeEllipseIntersection(
                conic, opacity, disc, t, p, isY, max_line
            );
        }

        // Determine ellipse min/max in this slice
        if (min_line <= bbox_argmin.y && bbox_argmin.y < max_line) {
            ellipse_min = bbox_min.y;
        } else {
            ellipse_min = min(intersect_min_line.x, intersect_max_line.x);
        }

        if (min_line <= bbox_argmax.y && bbox_argmax.y < max_line) {
            ellipse_max = bbox_max.y;
        } else {
            ellipse_max = max(intersect_min_line.y, intersect_max_line.y);
        }

        // Convert ellipse_min/ellipse_max to tiles touched
        int min_tile_v = max(
            rect_min.y,
            min(rect_max.y, (int)(ellipse_min / BLOCK_V))
        );
        int max_tile_v = min(
            rect_max.y,
            max(rect_min.y, (int)(ellipse_max / BLOCK_V + 1))
        );

        tiles_count += max_tile_v - min_tile_v;

        // Write to arrays if provided
        if (isect_ids != nullptr && flatten_ids != nullptr) {
            for (int v = min_tile_v; v < max_tile_v; v++) {
                // Compute tile ID
                int64_t tile_id = isY ? (u * tile_width + v) : (v * tile_width + u);

                // Encode: camera id | tile id | depth
                int64_t depth_id_enc = (int64_t) * (int32_t *)&depth;
                isect_ids[off] = cid_enc | (tile_id << 32) | depth_id_enc;
                flatten_ids[off] = static_cast<int32_t>(idx);
                off++;
            }
        }

        intersect_min_line = intersect_max_line;
        min_line = max_line;
    }
    return tiles_count;
}

// Main function: duplicate Gaussian to tiles it touches using Compact Box
__device__ inline uint32_t duplicateToTilesTouched(
    const vec2 p,
    const vec3 conic,
    const float opacity,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_size,
    const float mult,
    uint32_t idx,
    uint32_t off,
    float depth,
    int64_t cid_enc,
    int64_t *__restrict__ isect_ids,
    int32_t *__restrict__ flatten_ids
) {
    // Calculate discriminant
    float disc = conic.y * conic.y - conic.x * conic.z;

    // If ill-formed ellipse, return 0
    if (conic.x <= 0 || conic.z <= 0 || disc >= 0) {
        return 0;
    }

    // Threshold: opacity * Gaussian = 1 / 255 (ALPHA_THRESHOLD = 1/255)
    float t = 2.0f * log(opacity * 255.0f);
    t = mult * t;  // beta in Compact Box

    // Compute bounding box of the ellipse
    float x_term = sqrt(-(conic.y * conic.y * t) / (disc * conic.x));
    x_term = (conic.y < 0) ? x_term : -x_term;
    float y_term = sqrt(-(conic.y * conic.y * t) / (disc * conic.z));
    y_term = (conic.y < 0) ? y_term : -y_term;

    vec2 bbox_argmin = vec2{p.y - y_term, p.x - x_term};
    vec2 bbox_argmax = vec2{p.y + y_term, p.x + x_term};

    vec2 bbox_min = vec2{
        computeEllipseIntersection(conic, opacity, disc, t, p, true, bbox_argmin.x).x,
        computeEllipseIntersection(conic, opacity, disc, t, p, false, bbox_argmin.y).x
    };
    vec2 bbox_max = vec2{
        computeEllipseIntersection(conic, opacity, disc, t, p, true, bbox_argmax.x).y,
        computeEllipseIntersection(conic, opacity, disc, t, p, false, bbox_argmax.y).y
    };

    // Rectangular tile extent of ellipse
    int2 rect_min = int2{
        max(0, min((int)tile_width, (int)(bbox_min.x / tile_size))),
        max(0, min((int)tile_height, (int)(bbox_min.y / tile_size)))
    };
    int2 rect_max = int2{
        max(0, min((int)tile_width, (int)(bbox_max.x / tile_size + 1))),
        max(0, min((int)tile_height, (int)(bbox_max.y / tile_size + 1)))
    };

    int y_span = rect_max.y - rect_min.y;
    int x_span = rect_max.x - rect_min.x;

    // If no tiles are touched, return 0
    if (y_span * x_span == 0) {
        return 0;
    }

    // If fewer y tiles, loop over y slices else loop over x slices
    bool isY = y_span < x_span;
    return processTiles(
        conic, opacity, disc, t, p,
        bbox_min, bbox_max,
        bbox_argmin, bbox_argmax,
        rect_min, rect_max,
        tile_width, tile_height, tile_size,
        isY, idx, off, depth, cid_enc,
        isect_ids, flatten_ids
    );
}

/****************************************************************************
 * Gaussian Tile Intersection with Compact Box
 ****************************************************************************/

template <typename scalar_t>
__global__ void points_isect_tiles_cb(
    // if the data is [C, N, ...] or [nnz, ...] (packed)
    const bool packed,
    // parallelize over C * N, only used if packed is False
    const uint32_t C,
    const uint32_t N,
    // parallelize over nnz, only used if packed is True
    const uint32_t nnz,
    const int64_t *__restrict__ camera_ids,   // [nnz] optional
    const int64_t *__restrict__ gaussian_ids, // [nnz] optional
    // data
    const scalar_t *__restrict__ means2d,            // [C, N, 2] or [nnz, 2]
    const scalar_t *__restrict__ conics,             // [C, N, 3] or [nnz, 3]
    const scalar_t *__restrict__ opacities,          // [C, N] or [nnz]
    const scalar_t *__restrict__ depths,             // [C, N] or [nnz]
    const int64_t *__restrict__ cum_tiles_per_gauss, // [C, N] or [nnz]
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_n_bits,
    const float compact_box_mult,
    int64_t *__restrict__ tiles_per_gauss, // [C*N] or [nnz] - for counting pass
    int64_t *__restrict__ isect_ids,       // [n_isects] - for writing pass
    int32_t *__restrict__ flatten_ids      // [n_isects] - for writing pass
) {
    // parallelize over C * N or nnz
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= (packed ? nnz : C * N)) {
        return;
    }

    // Load data for this Gaussian
    vec2 mean2d = glm::make_vec2(means2d + 2 * idx);
    vec3 conic = glm::make_vec3(conics + 3 * idx);
    float opacity = opacities[idx];
    float depth = depths[idx];

    int64_t cid; // camera id
    if (packed) {
        cid = camera_ids[idx];
    } else {
        cid = idx / N;
    }
    const int64_t cid_enc = cid << (32 + tile_n_bits);

    // Determine if this is counting pass or writing pass
    bool counting_pass = (tiles_per_gauss != nullptr);
    int64_t cur_idx = 0;
    if (!counting_pass) {
        cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
    }

    // Compute tile intersections using Compact Box
    uint32_t num_tiles = duplicateToTilesTouched(
        mean2d,
        conic,
        opacity,
        tile_width,
        tile_height,
        tile_size,
        compact_box_mult,
        idx,
        cur_idx,
        depth,
        cid_enc,
        counting_pass ? nullptr : isect_ids,
        counting_pass ? nullptr : flatten_ids
    );

    // Write tile count if counting pass
    if (counting_pass) {
        tiles_per_gauss[idx] = num_tiles;
    }
}

std::tuple<at::Tensor, at::Tensor> points_isect_tiles_cb_tensor(
    const at::Tensor means2d,                    // [C, N, 2] or [nnz, 2]
    const at::Tensor conics,                     // [C, N, 3] or [nnz, 3]
    const at::Tensor opacities,                  // [C, N] or [nnz]
    const at::Tensor depths,                     // [C, N] or [nnz]
    const at::optional<at::Tensor> camera_ids,   // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t C,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const float compact_box_mult,
    const bool sort,
    const bool double_buffer
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(opacities);
    CHECK_INPUT(depths);
    if (camera_ids.has_value()) {
        CHECK_INPUT(camera_ids.value());
    }
    if (gaussian_ids.has_value()) {
        CHECK_INPUT(gaussian_ids.value());
    }
    bool packed = means2d.dim() == 2;

    uint32_t N = 0, nnz = 0, total_elems = 0;
    int64_t *camera_ids_ptr = nullptr;
    int64_t *gaussian_ids_ptr = nullptr;
    if (packed) {
        nnz = means2d.size(0);
        total_elems = nnz;
        TORCH_CHECK(
            camera_ids.has_value() && gaussian_ids.has_value(),
            "When packed is set, camera_ids and gaussian_ids must be provided."
        );
        camera_ids_ptr = camera_ids.value().data_ptr<int64_t>();
        gaussian_ids_ptr = gaussian_ids.value().data_ptr<int64_t>();
    } else {
        N = means2d.size(1); // number of gaussians
        total_elems = C * N;
    }

    uint32_t n_tiles = tile_width * tile_height;
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
    uint32_t cam_n_bits = (uint32_t)floor(log2(C)) + 1;
    assert(tile_n_bits + cam_n_bits <= 32);

    // First pass: count tiles per Gaussian
    at::Tensor tiles_per_gauss = at::zeros({(int64_t)total_elems}, means2d.options().dtype(at::kLong));

    AT_DISPATCH_FLOATING_TYPES(
        means2d.scalar_type(),
        "points_isect_tiles_cb_count",
        [&]() {
            points_isect_tiles_cb<scalar_t>
                <<<(total_elems + N_THREADS_PACKED - 1) / N_THREADS_PACKED,
                   N_THREADS_PACKED,
                   0,
                   stream>>>(
                    packed,
                    C,
                    N,
                    nnz,
                    camera_ids_ptr,
                    gaussian_ids_ptr,
                    means2d.data_ptr<scalar_t>(),
                    conics.data_ptr<scalar_t>(),
                    opacities.data_ptr<scalar_t>(),
                    depths.data_ptr<scalar_t>(),
                    nullptr,  // cum_tiles_per_gauss = nullptr for counting pass
                    tile_size,
                    tile_width,
                    tile_height,
                    tile_n_bits,
                    compact_box_mult,
                    tiles_per_gauss.data_ptr<int64_t>(),  // tiles_per_gauss for counting pass
                    nullptr,  // isect_ids = nullptr for counting pass
                    nullptr   // flatten_ids = nullptr for counting pass
                );
        }
    );

    at::Tensor cum_tiles_per_gauss = at::cumsum(tiles_per_gauss, 0);
    int64_t n_isects = cum_tiles_per_gauss[-1].item<int64_t>();

    at::Tensor isect_ids = at::empty({n_isects}, depths.options().dtype(at::kLong));
    at::Tensor flatten_ids = at::empty({n_isects}, depths.options().dtype(at::kInt));

    if (n_isects) {
        // Second pass: write isect_ids and flatten_ids
        AT_DISPATCH_FLOATING_TYPES(
            means2d.scalar_type(),
            "points_isect_tiles_cb_write",
            [&]() {
                points_isect_tiles_cb<scalar_t>
                    <<<(total_elems + N_THREADS_PACKED - 1) / N_THREADS_PACKED,
                       N_THREADS_PACKED,
                       0,
                       stream>>>(
                        packed,
                        C,
                        N,
                        nnz,
                        camera_ids_ptr,
                        gaussian_ids_ptr,
                        means2d.data_ptr<scalar_t>(),
                        conics.data_ptr<scalar_t>(),
                        opacities.data_ptr<scalar_t>(),
                        depths.data_ptr<scalar_t>(),
                        cum_tiles_per_gauss.data_ptr<int64_t>(),
                        tile_size,
                        tile_width,
                        tile_height,
                        tile_n_bits,
                        compact_box_mult,
                        nullptr,  // tiles_per_gauss = nullptr for writing pass
                        isect_ids.data_ptr<int64_t>(),
                        flatten_ids.data_ptr<int32_t>()
                    );
            }
        );
    }

    // optionally sort the Gaussians by isect_ids
    if (n_isects && sort) {
        at::Tensor isect_ids_sorted = at::empty_like(isect_ids);
        at::Tensor flatten_ids_sorted = at::empty_like(flatten_ids);

        if (double_buffer) {
            cub::DoubleBuffer<int64_t> d_keys(
                isect_ids.data_ptr<int64_t>(),
                isect_ids_sorted.data_ptr<int64_t>()
            );
            cub::DoubleBuffer<int32_t> d_values(
                flatten_ids.data_ptr<int32_t>(),
                flatten_ids_sorted.data_ptr<int32_t>()
            );
            CUB_WRAPPER(
                cub::DeviceRadixSort::SortPairs,
                d_keys,
                d_values,
                n_isects,
                0,
                32 + tile_n_bits + cam_n_bits,
                stream
            );
            switch (d_keys.selector) {
            case 0:
                isect_ids_sorted = isect_ids;
                break;
            case 1:
                break;
            }
            switch (d_values.selector) {
            case 0:
                flatten_ids_sorted = flatten_ids;
                break;
            case 1:
                break;
            }
        } else {
            CUB_WRAPPER(
                cub::DeviceRadixSort::SortPairs,
                isect_ids.data_ptr<int64_t>(),
                isect_ids_sorted.data_ptr<int64_t>(),
                flatten_ids.data_ptr<int32_t>(),
                flatten_ids_sorted.data_ptr<int32_t>(),
                n_isects,
                0,
                32 + tile_n_bits + cam_n_bits,
                stream
            );
        }
        return std::make_tuple(isect_ids_sorted, flatten_ids_sorted);
    } else {
        return std::make_tuple(isect_ids, flatten_ids);
    }
}

} // namespace gsplat
