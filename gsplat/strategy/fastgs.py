"""FastGS Strategy Implementation.

This module implements the FastGS strategy from the paper:
"FastGS: Training 3D Gaussian Splatting in 100 Seconds"

The strategy combines:
1. VCD (Multi-view Consistent Densification): Densifies Gaussians based on multi-view error consistency
2. VCP (Multi-view Consistent Pruning): Prunes Gaussians with low multi-view contribution
3. Importance-based filtering: Only densifies Gaussians that show consistent high error across views
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor

from .base import Strategy
from .ops import duplicate, remove, reset_opa, split


@dataclass
class FastGSStrategy(Strategy):
    """FastGS strategy for efficient 3D Gaussian splatting training.

    This strategy implements the FastGS paper's VCD and VCP methods to achieve
    3-4x training speedup while maintaining rendering quality and reducing Gaussian count.

    The strategy will:
    - Use multi-view consistent densification (VCD) to select high-importance Gaussians
    - Apply importance-based filtering (importance_score > threshold) during densification
    - Periodically apply VCP final pruning to remove low-contribution Gaussians
    - Support both standard and absolute gradients for splitting

    Args:
        loss_thresh (float): Threshold for high-error pixel detection. Default: 0.1
        grad_thresh (float): Gradient threshold for cloning (standard gradients). Default: 0.0002
        grad_abs_thresh (float): Gradient threshold for splitting (absolute gradients). Default: 0.0012
        dense (float): Scale threshold to distinguish small/large Gaussians. Default: 0.001
        importance_thresh (float): Importance score threshold for densification. Default: 5.0
        pruning_thresh (float): Pruning score threshold for VCP final pruning. Default: 0.9
        prune_opa (float): Opacity threshold for pruning. Default: 0.005
        prune_scale3d (float): 3D scale threshold for pruning (normalized by scene_scale). Default: 0.1
        num_views_sample (int): Number of views to sample for VCD/VCP. Default: 10
        refine_start_iter (int): Start densification after this iteration. Default: 500
        refine_stop_iter (int): Stop densification after this iteration. Default: 15_000
        refine_every (int): Perform densification every N iterations. Default: 500
        final_prune_start (int): Start final VCP pruning after this iteration. Default: 15_000
        final_prune_stop (int): Stop final VCP pruning after this iteration. Default: 30_000
        final_prune_every (int): Perform final pruning every N iterations. Default: 3000
        reset_every (int): Reset opacities every N iterations. Default: 3000
        verbose (bool): Print verbose information. Default: False

    Examples:

        >>> from gsplat import FastGSStrategy, rasterization
        >>> params: Dict[str, torch.nn.Parameter] = ...
        >>> optimizers: Dict[str, torch.optim.Optimizer] = ...
        >>> strategy = FastGSStrategy()
        >>> strategy.check_sanity(params, optimizers)
        >>> strategy_state = strategy.initialize_state()
        >>> for step in range(30000):
        ...     render_image, render_alpha, info = rasterization(..., absgrad=True)
        ...     strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        ...     loss = ...
        ...     loss.backward()
        ...     strategy.step_post_backward(
        ...         params, optimizers, strategy_state, step, info,
        ...         cameras=cameras, trainset=trainset
        ...     )
    """

    # FastGS hyperparameters
    loss_thresh: float = 0.1
    grad_thresh: float = 0.0002
    grad_abs_thresh: float = 0.0012
    dense: float = 0.001
    importance_thresh: float = 5.0
    pruning_thresh: float = 0.9
    prune_opa: float = 0.005
    prune_scale3d: float = 0.1

    # Multi-view sampling
    num_views_sample: int = 10

    # Scheduling
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    refine_every: int = 500
    final_prune_start: int = 15_000
    final_prune_stop: int = 30_000
    final_prune_every: int = 3000
    reset_every: int = 3000

    # Options
    verbose: bool = False

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy.

        The returned state should be passed to the `step_pre_backward()` and
        `step_post_backward()` functions.

        Args:
            scene_scale (float): The scale of the scene, used for normalizing 3D scales.

        Returns:
            Dict[str, Any]: The initialized state dictionary.
        """
        # FastGS uses both standard and absolute gradients
        state = {
            "grad2d": None,  # Standard gradients for cloning
            "grad2d_abs": None,  # Absolute gradients for splitting
            "count": None,  # Visibility counter
            "radii": None,  # 2D radii for tracking
            "scene_scale": scene_scale,
            # FastGS-specific: importance and pruning scores
            "importance_score": None,  # Multi-view consistent densification score
            "pruning_score": None,  # Multi-view consistent pruning score
        }
        return state

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for the parameters and optimizers.

        Checks if:
            * `params` and `optimizers` have the same keys.
            * Each optimizer has exactly one param_group.
            * Required keys are present: {"means", "scales", "quats", "opacities"}.

        Raises:
            AssertionError: If any of the conditions is not met.
        """
        super().check_sanity(params, optimizers)
        # The following keys are required for this strategy.
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """Callback function to be executed before the `loss.backward()` call.

        Args:
            params: The parameters of the model.
            optimizers: The optimizers for the parameters.
            state: The running state of the strategy.
            step: The current training step.
            info: The information dictionary from the rasterization.
        """
        # Retain gradients for means2d (both standard and absolute)
        assert "means2d" in info, "means2d is required but missing in info."
        info["means2d"].retain_grad()

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
        # FastGS-specific: need access to training data for multi-view sampling
        cameras: Optional[Any] = None,
        trainset: Optional[Any] = None,
    ):
        """Callback function to be executed after the `loss.backward()` call.

        Args:
            params: The parameters of the model.
            optimizers: The optimizers for the parameters.
            state: The running state of the strategy.
            step: The current training step.
            info: The information dictionary from the rasterization.
            packed: Whether the tensors are in packed mode.
            cameras: The camera parameters for multi-view sampling (FastGS-specific).
            trainset: The training dataset for multi-view sampling (FastGS-specific).
        """
        if packed:
            raise NotImplementedError("FastGS strategy does not support packed mode yet.")

        N = params["means"].shape[0]
        device = params["means"].device

        # Initialize state on first call or resize if N changed
        if state["grad2d"] is None or state["grad2d"].shape[0] != N:
            # Resize or initialize state tensors to match current N
            old_N = 0 if state["grad2d"] is None else state["grad2d"].shape[0]

            # Create new tensors
            new_grad2d = torch.zeros(N, device=device)
            new_grad2d_abs = torch.zeros(N, device=device)
            new_count = torch.zeros(N, device=device, dtype=torch.int)
            new_radii = torch.zeros(N, device=device)
            new_importance_score = torch.zeros(N, device=device)
            new_pruning_score = torch.zeros(N, device=device)

            # Copy old values if resizing
            if old_N > 0:
                copy_N = min(old_N, N)
                new_grad2d[:copy_N] = state["grad2d"][:copy_N]
                new_grad2d_abs[:copy_N] = state["grad2d_abs"][:copy_N]
                new_count[:copy_N] = state["count"][:copy_N]
                new_radii[:copy_N] = state["radii"][:copy_N]
                new_importance_score[:copy_N] = state["importance_score"][:copy_N]
                new_pruning_score[:copy_N] = state["pruning_score"][:copy_N]

            # Update state
            state["grad2d"] = new_grad2d
            state["grad2d_abs"] = new_grad2d_abs
            state["count"] = new_count
            state["radii"] = new_radii
            state["importance_score"] = new_importance_score
            state["pruning_score"] = new_pruning_score

        # Get visibility mask
        assert "radii" in info, "radii is required in info."
        radii = info["radii"]  # [C, N, 2] or [C, N]

        # Handle both [C, N, 2] and [C, N] shapes
        if radii.ndim == 3:
            # [C, N, 2] - check if any radius dimension > 0
            valid_mask = (radii > 0.0).any(-1).any(0)  # [N]
            radii_max = radii.max(-1)[0]  # [C, N]
        else:
            # [C, N]
            valid_mask = (radii > 0.0).any(0)  # [N]
            radii_max = radii

        # Update gradient accumulators
        means2d = info["means2d"]  # [C, N, 2]

        # Standard gradients for cloning
        if means2d.grad is not None:
            # means2d.grad is [C, N, 2], we need per-Gaussian norms
            grad2d_norm = torch.norm(means2d.grad, dim=-1).mean(0)  # [N]
            state["grad2d"][valid_mask] += grad2d_norm[valid_mask]
            state["count"][valid_mask] += 1

        # Absolute gradients for splitting (if available)
        if hasattr(means2d, "absgrad") and means2d.absgrad is not None:
            grad2d_abs_norm = torch.norm(means2d.absgrad, dim=-1).mean(0)  # [N]
            state["grad2d_abs"][valid_mask] += grad2d_abs_norm[valid_mask]

        # Update radii - take max across cameras
        state["radii"][valid_mask] = torch.maximum(
            state["radii"][valid_mask],
            radii_max.max(0)[0][valid_mask]  # [N]
        )

        # Perform densification and pruning
        do_refine = (
            step >= self.refine_start_iter
            and step < self.refine_stop_iter
            and step % self.refine_every == 0
        )

        if do_refine:
            if cameras is not None and trainset is not None:
                # Compute multi-view scores for VCD/VCP
                self._compute_multi_view_scores(params, cameras, trainset, state)
            else:
                if self.verbose:
                    print(
                        f"Warning: cameras and trainset not provided at step {step}. "
                        "Skipping multi-view score computation."
                    )

            # Perform densification with importance filtering
            self._densify_and_prune_fastgs(params, optimizers, state, step)

        # Perform VCP final pruning
        do_final_prune = (
            step >= self.final_prune_start
            and step < self.final_prune_stop
            and step % self.final_prune_every == 0
        )

        if do_final_prune:
            if cameras is not None and trainset is not None:
                # Update pruning scores
                self._compute_multi_view_scores(params, cameras, trainset, state)
            self._final_prune_fastgs(params, optimizers, state)

        # Reset opacity periodically
        if step > 0 and step % self.reset_every == 0:
            reset_opa(params, optimizers, state, value=0.01)

    def _compute_multi_view_scores(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        cameras: Any,
        trainset: Any,
        state: Dict[str, Any],
    ) -> None:
        """Compute importance and pruning scores using multi-view consistency.

        This implements VCD (multi-view consistent densification) and VCP
        (multi-view consistent pruning) from the FastGS paper.

        Args:
            params: The parameters of the model.
            cameras: The camera parameters.
            trainset: The training dataset.
            state: The running state to update with scores.
        """
        from gsplat import rasterization
        from gsplat.cuda._wrapper import rasterize_to_pixels

        N = params["means"].shape[0]
        device = params["means"].device

        # Sample random views
        num_cameras = len(trainset)
        num_sample = min(self.num_views_sample, num_cameras)
        sampled_indices = torch.randperm(num_cameras, device=device)[:num_sample].cpu().tolist()

        # Accumulators for multi-view metrics
        importance_counts = torch.zeros(N, device=device, dtype=torch.float32)
        pruning_scores = torch.zeros(N, device=device, dtype=torch.float32)
        photometric_losses = []

        with torch.no_grad():
            for idx in sampled_indices:
                # Get camera and ground truth image
                camera_data = trainset[idx]
                # Assume trainset returns a dict with camera params and GT image
                # This interface may need adjustment based on actual trainset structure

                # Render the view
                # Note: This is a simplified rendering call - actual implementation
                # may need to match the training loop's rendering setup
                try:
                    render_colors, render_alphas, info = rasterization(
                        means=params["means"],
                        quats=params["quats"],
                        scales=params["scales"],
                        opacities=params["opacities"],
                        colors=params["colors"] if "colors" in params else params["features_dc"],
                        viewmats=camera_data["viewmat"].unsqueeze(0),
                        Ks=camera_data["K"].unsqueeze(0),
                        width=camera_data["width"],
                        height=camera_data["height"],
                        packed=False,
                        absgrad=False,
                    )

                    gt_image = camera_data["image"]  # [H, W, 3]
                    render_image = render_colors[0]  # [H, W, 3]

                    # Compute per-pixel L1 loss
                    pixel_loss = torch.abs(render_image - gt_image).mean(dim=-1)  # [H, W]

                    # Create binary mask for high-error pixels
                    metric_map = (pixel_loss > self.loss_thresh).bool()  # [H, W]

                    # Compute photometric loss (L1 + SSIM) for VCP
                    l1_loss = pixel_loss.mean()

                    # Compute SSIM loss using fused_ssim (same as training)
                    try:
                        from fused_ssim import fused_ssim
                        ssim_loss = 1.0 - fused_ssim(
                            render_image.unsqueeze(0).permute(0, 3, 1, 2),
                            gt_image.unsqueeze(0).permute(0, 3, 1, 2),
                            padding="valid"
                        )
                        # Combine L1 and SSIM with default 0.2 weight (like training)
                        photo_loss = l1_loss * 0.8 + ssim_loss * 0.2
                    except ImportError:
                        # Fallback to L1 only if fused_ssim not available
                        photo_loss = l1_loss

                    photometric_losses.append(photo_loss.item())

                    # Initialize metric_counts for this view
                    metric_counts = torch.zeros(N, device=device, dtype=torch.int32)

                    # Render again with metric accumulation
                    # We need to call the lower-level rasterize_to_pixels with metric parameters
                    # This requires having the intermediate results from the first render

                    # Extract required tensors from info
                    means2d = info["means2d"][0]  # [N, 2]
                    conics = info["conics"][0]  # [N, 3]
                    opacities_2d = info["opacities"][0]  # [N]

                    # Prepare colors for rasterization
                    if "colors" in params:
                        colors_2d = params["colors"]
                    else:
                        colors_2d = params["features_dc"]

                    if colors_2d.dim() == 2:
                        colors_2d = colors_2d.unsqueeze(0)  # [1, N, C]

                    # Call rasterize_to_pixels with metric accumulation
                    metric_map_3d = metric_map.unsqueeze(0)  # [1, H, W]

                    _, _ = rasterize_to_pixels(
                        means2d=means2d.unsqueeze(0),  # [1, N, 2]
                        conics=conics.unsqueeze(0),  # [1, N, 3]
                        colors=colors_2d,  # [1, N, C]
                        opacities=opacities_2d.unsqueeze(0),  # [1, N]
                        image_width=camera_data["width"],
                        image_height=camera_data["height"],
                        tile_size=16,
                        isect_offsets=info["isect_offsets"],
                        flatten_ids=info["flatten_ids"],
                        packed=False,
                        absgrad=False,
                        metric_map=metric_map_3d,
                        metric_counts=metric_counts,
                    )

                    # Accumulate importance counts (for VCD)
                    importance_counts += metric_counts.float()

                    # Accumulate pruning scores (for VCP)
                    # Pruning score = metric_counts * photometric_loss
                    pruning_scores += metric_counts.float() * photo_loss

                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Failed to process view {idx}: {e}")
                    continue

        # Average importance scores across views
        state["importance_score"] = importance_counts / max(num_sample, 1)

        # Normalize pruning scores to [0, 1]
        if pruning_scores.max() > 0:
            state["pruning_score"] = pruning_scores / pruning_scores.max()
        else:
            state["pruning_score"] = pruning_scores

        if self.verbose:
            print(f"Multi-view scores computed from {num_sample} views")
            print(f"  Mean importance score: {state['importance_score'].mean():.4f}")
            print(f"  Mean pruning score: {state['pruning_score'].mean():.4f}")

    def _densify_and_prune_fastgs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ):
        """Perform FastGS densification with importance filtering and pruning.

        Args:
            params: The parameters of the model.
            optimizers: The optimizers for the parameters.
            state: The running state of the strategy.
            step: The current training step.
        """
        N = params["means"].shape[0]
        device = params["means"].device
        count = state["count"]

        # Compute average gradients
        is_grad_valid = count > 0
        grad2d_avg = torch.zeros_like(state["grad2d"])
        grad2d_avg[is_grad_valid] = state["grad2d"][is_grad_valid] / count[is_grad_valid]

        grad2d_abs_avg = torch.zeros_like(state["grad2d_abs"])
        grad2d_abs_avg[is_grad_valid] = state["grad2d_abs"][is_grad_valid] / count[is_grad_valid]

        # Get 3D scales
        scales = torch.exp(params["scales"])  # Assuming log-space scales
        scene_scale = state["scene_scale"]

        # Candidates for densification: high gradient and high importance
        grad_mask_clone = grad2d_avg >= self.grad_thresh
        grad_mask_split = grad2d_abs_avg >= self.grad_abs_thresh

        # FastGS: Apply importance filtering
        importance_mask = state["importance_score"] > self.importance_thresh

        # Small Gaussians: clone
        is_small = (scales.max(dim=-1).values <= self.dense * scene_scale)
        clone_mask = grad_mask_clone & is_small & importance_mask

        # Large Gaussians: split
        is_large = ~is_small
        split_mask = grad_mask_split & is_large & importance_mask

        if self.verbose:
            print(f"Step {step}: Densification")
            print(f"  Candidates: {importance_mask.sum()} with high importance")
            print(f"  Cloning: {clone_mask.sum()} small Gaussians")
            print(f"  Splitting: {split_mask.sum()} large Gaussians")

        # Perform cloning
        if clone_mask.sum() > 0:
            duplicate(params, optimizers, state, clone_mask)

        # Perform splitting
        if split_mask.sum() > 0:
            split(params, optimizers, state, split_mask)

        # Pruning based on opacity and scale
        prune_mask = torch.zeros(params["means"].shape[0], dtype=torch.bool, device=device)

        # Prune low opacity
        prune_mask |= (torch.sigmoid(params["opacities"]) < self.prune_opa).squeeze()

        # Prune large scales
        scales_after = torch.exp(params["scales"])
        prune_mask |= (scales_after.max(dim=-1).values > self.prune_scale3d * scene_scale)

        # FastGS: Budget-based pruning using pruning_score
        # Select worst 50% of Gaussians above a certain opacity threshold
        opacity_threshold = 0.005
        valid_for_pruning = (torch.sigmoid(params["opacities"]) >= opacity_threshold).squeeze()

        if valid_for_pruning.sum() > 0:
            pruning_scores_valid = state["pruning_score"][valid_for_pruning]
            if pruning_scores_valid.max() > 0:
                # Get median pruning score
                median_score = pruning_scores_valid.median()
                # Prune Gaussians with pruning score above median
                budget_prune_mask = torch.zeros_like(prune_mask)
                budget_prune_mask[valid_for_pruning] = state["pruning_score"][valid_for_pruning] > median_score
                prune_mask |= budget_prune_mask

        if self.verbose:
            print(f"  Pruning: {prune_mask.sum()} Gaussians")

        if prune_mask.sum() > 0:
            remove(params, optimizers, state, prune_mask)

        # Reset gradients
        state["grad2d"].zero_()
        state["grad2d_abs"].zero_()
        state["count"].zero_()
        torch.cuda.empty_cache()

    def _final_prune_fastgs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
    ):
        """Perform VCP final pruning.

        Removes Gaussians with opacity < 0.1 OR pruning_score > threshold.

        Args:
            params: The parameters of the model.
            optimizers: The optimizers for the parameters.
            state: The running state of the strategy.
        """
        device = params["means"].device
        prune_mask = torch.zeros(params["means"].shape[0], dtype=torch.bool, device=device)

        # Prune by opacity
        prune_mask |= (torch.sigmoid(params["opacities"]) < 0.1).squeeze()

        # Prune by pruning score
        prune_mask |= (state["pruning_score"] > self.pruning_thresh)

        if self.verbose:
            n_prune = prune_mask.sum()
            print(f"VCP Final Pruning: Removing {n_prune} Gaussians")

        if prune_mask.sum() > 0:
            remove(params, optimizers, state, prune_mask)
            torch.cuda.empty_cache()
