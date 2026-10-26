"""Motion-compensated 4D MRI reconstruction with TV regularization."""

import time

import torch

from .computational_functions import (
    build_motion_optimizer_groups,
    fit_motion_fields,
    chambolle_pock_tv,
    foreground_mask,
    make_grid,
)
from .helper_functions import scale_tv_weight_by_signal
from .modules import GaussianModel
from .mri_operators import A_adjoint, A_forward, warp_3d_batch_bspline


def DoReMe(
    readouts,
    S,
    sampling_coordinates,
    image_shape,
    data_consistency_step_size=0.01,
    tv_weight=0.1,
    max_iter=5,
    gradient_dims=[0],
    enable_motion_estimation=True,
    motion_ref_state=0,
    motion_fit_start_iter=0,
    motion_fit_interval=1,
    motion_fit_first_iters=None,
    motion_fit_iters=150,
    motion_fit_lr=5e-2,
    motion_fit_components=5000,
    motion_fit_neighbors=64,
    motion_fit_sample_voxels=16000,
    motion_smooth_weight=10.0,
    motion_ref_free_anchor_state=3,
    motion_ref_free_anchor_weight=0.0,
    motion_ref_free_anchor_hard_enforce=False,
    motion_ref_free_zero_mean_enforce=True,
    motion_jacobian_weight=0.0,
    motion_jacobian_identity_weight=0.0,
    motion_jacobian_det_min=0.0,
    motion_jacobian_sample_voxels=4096,
    motion_jacobian_eps=1.0,
    motion_init_method="mask",
    motion_mask_threshold=0.1,
    motion_inverse_fixed_point_iters=30,
    motion_sample_mode="uniform",
    return_first_motion_fields=False,
    motion_reg_scale=None,
    scale_tv_weight_by_data=False,
    tv_scale_percentage=100.0,
    tv_scale_lipschitz_L=1.0,
    motion_fit_parameter_mode="all",
    motion_covariance="diagonal",
    circular_gradient_dims=None,
    motion_fit_image_mode="complex",
    motion_respiratory_smooth_weight=0.0,
):
    """Reconstruct a 4D image with TV regularization and fitted motion fields.

    Args:
        readouts: Complex CUDA tensor (C, N, X), with C coils and N readouts.
        S: Complex coil sensitivity maps (C, Z, Y, X), on the same device.
        sampling_coordinates: Integer tensor (N, >=3), aligned with readouts.
            Columns 0, 1, 2 contain y, z, and motion-state indices. Each row
            describes a full X readout; additional columns are ignored.
        image_shape: Tuple (M, Z, Y, X), where M is the motion-state count.
        data_consistency_step_size: Data-consistency gradient step size.
        tv_weight: TV weight. Accepts a scalar, sequence, or tensor. 
        max_iter: Number of reconstruction iterations.
        gradient_dims: Image axes for TV differences. Axis 0 is motion state;
        enable_motion_estimation: Fit and apply motion fields when True.
            Otherwise reconstruct with TV only.
        motion_ref_state: Reference-state index for reference-image L1 fitting.
            None selects reference-free circular adjacent-state TV fitting.
        motion_fit_start_iter: motion fit start. 
        motion_fit_interval: Refit motion every N iterations
        motion_fit_first_iters: Adam steps for the first fit. 
        motion_fit_iters: Adam steps per subsequent fit.
        motion_fit_lr: Adam learning rate. 
        motion_fit_components: Number of Gaussian control components.
        motion_fit_neighbors: Nearest components used to evaluate each voxel.
            Values <=0 or >=motion_fit_components use all components.
        motion_fit_sample_voxels: Voxels sampled per Adam step. Values <=0 or
            >=Z*Y*X use the entire spatial grid.
        motion_smooth_weight: Weight of neighboring control-point displacement
            smoothness during fitting.
        motion_respiratory_smooth_weight: second-difference penalty on
            Gaussian displacement across respiratory states for smooth motion transition. Zero disables it.
        motion_ref_free_anchor_state: Motion state used by the reference-free
            soft anchor or hard zero-displacement constraint.
        motion_ref_free_anchor_weight: Weight penalizing squared control-point
            displacement at the anchor state. Zero disables the soft anchor.
        motion_ref_free_anchor_hard_enforce: Force anchor-state displacement
            to zero in reference-free fitting.
        motion_ref_free_zero_mean_enforce: Remove mean displacement across
            states in reference-free fitting. Use as an alternative to the
            hard anchor, not together with it. 
        motion_jacobian_weight: Weight of mean relu(det_min - det(I + grad(u)))^2.
            Zero disables this fold-floor penalty.
        motion_jacobian_identity_weight: Weight of mean (det(I + grad(u)) - 1)^2.
            Zero disables this volume-change penalty.
        motion_jacobian_det_min: Determinant floor for the fold penalty.
        motion_jacobian_sample_voxels: Maximum sampled locations for Jacobian
            training penalties. Values <=0 use all current fitting samples.
            Penalties are evaluated every five Adam steps for now.
        motion_jacobian_eps: Finite-difference step in voxels. Jacobians use
            centered differences inside the volume and one-sided boundaries.
        motion_init_method: "mask" seeds control positions from foreground;
            "random" keeps their random volume-wide initialization.
        motion_mask_threshold: Foreground threshold relative to the maximum
            magnitude, used only for mask initialization.
        motion_inverse_fixed_point_iters: Fixed-point steps for inverting
            the fitted forward displacement field.
        motion_sample_mode: "uniform", "body", or "edge". Body/edge sampling
            requires reference-free fitting and a spatial voxel subset.
            Edge sampling uses Gaussian-blurred magnitude gradients.
        return_first_motion_fields: Append the first inverse/forward field
            pair to the result when True.
        motion_reg_scale: Reporting multiplier for motion regularization.
            None uses Z*Y*X / clamp(motion_fit_sample_voxels, 1, Z*Y*X).
            This does not change motion training or the reconstruction update.
        scale_tv_weight_by_data: Scale the TV weight once by a magnitude
            percentile of the first data-consistency image update.
        tv_scale_percentage: Magnitude percentile for that TV scaling;
            clamped to [0, 100], with 100 selecting the maximum.
        tv_scale_lipschitz_L: Must be positive when rescaling is enabled.
            The current scaling helper does not otherwise use this value.
        motion_fit_parameter_mode: "all" trains all Gaussian parameters;
            "displacement" trains displacement only; "positions_covariance"
            also trains positions and covariance; "covariance" also trains
            covariance. Displacements are trained in every mode.
        motion_covariance: Gaussian kernel covariance: "isotropic",
            "diagonal", or "full".
        circular_gradient_dims: TV axes with periodic boundaries. None makes
            every gradient_dims axis periodic; an empty sequence makes none
            periodic. Other selected axes have zero boundary differences.
        motion_fit_image_mode: "complex" preserves phase during fitting;
            "magnitude" fits absolute images. Reconstruction remains complex.

    Returns:
        Tuple containing, in order:
            image: Complex64 CUDA tensor (M, Z, Y, X).
            elapsed_seconds: Wall-clock reconstruction time, including fitting.
            errors: Dictionary of per-iteration float lists: data_consistency
                (0.5 * ||A(image) - readouts||_2^2), residual (the L2 norm),
                reconstruction_regularization (TV weight times mean TV),
                motion_regularization (scaled motion penalty), motion_smoothness,
                motion_anchor_bias, motion_jacobian_floor, and
                motion_jacobian_identity. Motion component metrics include their
                fitting weights; only motion_regularization includes motion_reg_scale.
            relative_errors: Per-iteration residual / (||readouts||_2 + 1e-12).
            inverse_motion_fields: Real tensor (M, Z, Y, X, 3), used to warp
                the common-space TV result back to individual motion states.
            forward_motion_fields: Real tensor (M, Z, Y, X, 3), used to warp
                motion states into the common space before TV.
        Field channels are voxel displacements ordered (dz, dy, dx), sampled
        using grid + displacement. Both fields are zero if no fit occurs.
        If return_first_motion_fields=True, two more entries follow:
            first_inverse_motion_fields, first_forward_motion_fields:
                Copies of the first fitted field pair, or None if no fit occurs.

    FISTA acceleration runs on every iteration without restarting after refits.
    """
    print(f"Device: {S.device}")

    # Scale the sampled motion regularization for reporting.
    _, lin, par, col = image_shape
    n_spatial_voxels = int(lin) * int(par) * int(col)
    _eff_sample_voxels = max(1, min(int(motion_fit_sample_voxels), n_spatial_voxels))
    _motion_reg_scale = float(motion_reg_scale) if motion_reg_scale is not None else float(n_spatial_voxels) / _eff_sample_voxels
    print(f"[motion-reg-scale] n_spatial_voxels={n_spatial_voxels} sample_voxels={_eff_sample_voxels} scale={_motion_reg_scale:.2f}")

    def zero_motion_metrics():
        return {
            "smoothness": 0.0,
            "respiratory_smoothness": 0.0,
            "anchor_bias": 0.0,
            "jacobian_floor": 0.0,
            "jacobian_identity": 0.0,
            "regularization": 0.0,
        }

    def tv_scalar(weight):
        if torch.is_tensor(weight):
            vals = weight.detach().to(device=readouts.device, dtype=torch.float32).flatten()
        elif isinstance(weight, (list, tuple)):
            vals = torch.tensor(weight, device=readouts.device, dtype=torch.float32).flatten()
        else:
            vals = torch.tensor([float(weight)], device=readouts.device, dtype=torch.float32)
        if vals.numel() == 0:
            raise ValueError("tv_weight must not be empty")
        return float(vals.mean().item())

    x = torch.zeros(image_shape, dtype=torch.complex64, device=readouts.device)
    y = x.clone()
    t = torch.tensor(1.0, dtype=torch.float32, device=readouts.device)
    start_time = time.time()
    errors = {
        "data_consistency": [],
        "reconstruction_regularization": [],
        "motion_regularization": [],
        "motion_smoothness": [],
        "motion_respiratory_smoothness": [],
        "motion_anchor_bias": [],
        "motion_jacobian_floor": [],
        "motion_jacobian_identity": [],
        "residual": [],
    }
    relative_errors = []
    motion_states, lin, par, col = image_shape
    inverse_motion_fields = torch.zeros(
        (motion_states, lin, par, col, 3),
        dtype=torch.float32,
        device=readouts.device,
    )
    forward_motion_fields = torch.zeros_like(inverse_motion_fields)
    first_motion_fit_done = False
    first_inverse_motion_fields = None
    first_forward_motion_fields = None
    last_motion_metrics = zero_motion_metrics()
    readouts_norm = torch.norm(readouts).item()
    base_regularizer_weight = tv_weight
    effective_regularizer_weight = tv_scalar(base_regularizer_weight)

    motion_model = None
    motion_optimizer = None
    motion_coords_all = None

    for i in range(max_iter):
        grad_f = A_adjoint(A_forward(y, S, sampling_coordinates) - readouts, S, sampling_coordinates, image_shape)
        tv_weight_iter = effective_regularizer_weight
        motion_metrics = zero_motion_metrics()

        z_before_moco = y - data_consistency_step_size * grad_f
        if i == 0 and scale_tv_weight_by_data:
            effective_regularizer_weight = scale_tv_weight_by_signal(
                tv_scalar(base_regularizer_weight),
                tv_scale_lipschitz_L,
                z_before_moco,
                percentage=tv_scale_percentage,
            )
            tv_weight_iter = effective_regularizer_weight
            print(
                "[tv-weight-scale] "
                f"base={tv_scalar(base_regularizer_weight):.8g} "
                f"effective={effective_regularizer_weight:.8g} "
                f"percentage={float(tv_scale_percentage):.3f}"
            )

        is_first_fit = not first_motion_fit_done
        if (
            enable_motion_estimation
            and i >= motion_fit_start_iter
            and (is_first_fit or (i - motion_fit_start_iter) % max(1, motion_fit_interval) == 0)
        ):
            fit_iters_now = int(motion_fit_iters)
            if is_first_fit:
                if motion_fit_first_iters is not None:
                    fit_iters_now = int(motion_fit_first_iters)
                device = y.device
                motion_model = GaussianModel(
                    image_shape[1:], motion_states, motion_fit_components,
                    motion_fit_neighbors, seed=i, covariance=motion_covariance,
                ).to(device)
                motion_optimizer = torch.optim.Adam(
                    build_motion_optimizer_groups(
                        motion_model, motion_fit_lr, motion_fit_parameter_mode
                    ),
                    lr=motion_fit_lr,
                )
                motion_coords_all = make_grid(image_shape[1:], device, torch.float32)
                if motion_init_method == "mask":
                    abs_image = y.detach().abs().to(torch.float32)
                    mask_source = (abs_image.mean(dim=0) if motion_ref_state is None else
                                   abs_image[int(motion_ref_state) % motion_states])
                    mask = foreground_mask(mask_source, motion_mask_threshold)
                    grid_vol = motion_coords_all.reshape(*image_shape[1:], 3)
                    motion_model.initialize_positions_from_mask(mask, grid_vol)

            inverse_motion_fields, forward_motion_fields, last_motion_metrics = fit_motion_fields(
                y,
                model=motion_model,
                opt=motion_optimizer,
                coords_all=motion_coords_all,
                ref_state=motion_ref_state,
                image_mode=motion_fit_image_mode,
                fit_iters=fit_iters_now,
                sample_voxels=motion_fit_sample_voxels,
                smooth_weight=motion_smooth_weight,
                respiratory_smooth_weight=motion_respiratory_smooth_weight,
                ref_free_anchor_state=motion_ref_free_anchor_state,
                ref_free_anchor_weight=motion_ref_free_anchor_weight,
                ref_free_anchor_hard_enforce=motion_ref_free_anchor_hard_enforce,
                ref_free_zero_mean_enforce=motion_ref_free_zero_mean_enforce,
                seed=i,
                inverse_fixed_point_iters=motion_inverse_fixed_point_iters,
                jacobian_weight=motion_jacobian_weight,
                jacobian_identity_weight=motion_jacobian_identity_weight,
                jacobian_det_min=motion_jacobian_det_min,
                jacobian_sample_voxels=motion_jacobian_sample_voxels,
                jacobian_eps=motion_jacobian_eps,
                return_loss_breakdown=True,
                sample_mode=motion_sample_mode,
            )

            if is_first_fit:
                first_inverse_motion_fields = inverse_motion_fields.clone()
                first_forward_motion_fields = forward_motion_fields.clone()
            first_motion_fit_done = True

        if enable_motion_estimation and first_motion_fit_done:
            motion_metrics = dict(last_motion_metrics)
            z_after_moco = warp_3d_batch_bspline(z_before_moco, forward_motion_fields)
            x_new_with_moco, tv_term = chambolle_pock_tv(
                z_after_moco, tv_weight_iter, gradient_dims=gradient_dims,
                circular_gradient_dims=circular_gradient_dims)
            x_new = warp_3d_batch_bspline(x_new_with_moco, inverse_motion_fields)
            del x_new_with_moco
        else:
            x_new, tv_term = chambolle_pock_tv(
                z_before_moco, tv_weight_iter, gradient_dims=gradient_dims,
                circular_gradient_dims=circular_gradient_dims)

        # FISTA acceleration. If reconstruction is unstable or overshoots, replace
        # these two lines with t_new = torch.ones_like(t) and y_candidate = x_new
        # to skip acceleration and use ISTA-style updates.
        t_new = (1 + torch.sqrt(1 + 4 * t**2)) / 2
        y_candidate = x_new + ((t - 1) / t_new) * (x_new - x)

        residual = A_forward(x_new, S, sampling_coordinates) - readouts
        residual_norm_sq_sum = torch.sum(torch.abs(residual) ** 2).item()
        residual_norm = residual_norm_sq_sum ** 0.5
        data_term = 0.5 * residual_norm_sq_sum

        reconstruction_reg = tv_scalar(tv_weight_iter) * tv_term

        motion_reg = float(motion_metrics["regularization"])
        motion_reg_scaled = _motion_reg_scale * motion_reg

        y = y_candidate

        relative_error = residual_norm / (readouts_norm + 1e-12)

        errors["data_consistency"].append(data_term)
        errors["reconstruction_regularization"].append(reconstruction_reg)
        errors["motion_regularization"].append(motion_reg_scaled)
        errors["motion_smoothness"].append(float(motion_metrics["smoothness"]))
        errors["motion_respiratory_smoothness"].append(float(motion_metrics["respiratory_smoothness"]))
        errors["motion_anchor_bias"].append(float(motion_metrics["anchor_bias"]))
        errors["motion_jacobian_floor"].append(float(motion_metrics["jacobian_floor"]))
        errors["motion_jacobian_identity"].append(float(motion_metrics["jacobian_identity"]))
        errors["residual"].append(residual_norm)
        relative_errors.append(relative_error)
        x = x_new
        t = t_new
        print(f"Iteration {i}: data consistency error = {data_term:.6g}")

    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    print(f"Reconstruction completed in {elapsed_time:.2f} s")

    if return_first_motion_fields:
        return (
            x,
            elapsed_time,
            errors,
            relative_errors,
            inverse_motion_fields,
            forward_motion_fields,
            first_inverse_motion_fields,
            first_forward_motion_fields,
        )

    return x, elapsed_time, errors, relative_errors, inverse_motion_fields, forward_motion_fields
