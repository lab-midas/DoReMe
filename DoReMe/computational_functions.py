"""TV, readout handling, and 4D motion fitting used by DoReMe."""

import numpy as np
import torch
import torch.nn.functional as F

from .helper_functions import foreground_mask, make_grid, sample_points
from .modules import GaussianModel


def normalize_tv_weights(tv_weight, n_dims, device, dtype):
    """Return nonnegative per-dimension TV weights as a 1D tensor."""
    if torch.is_tensor(tv_weight):
        w = tv_weight.to(device=device, dtype=dtype).flatten()
    elif isinstance(tv_weight, (list, tuple)):
        w = torch.tensor(tv_weight, device=device, dtype=dtype).flatten()
    else:
        w = torch.tensor([float(tv_weight)], device=device, dtype=dtype)

    if w.numel() == 1:
        w = w.repeat(int(n_dims))
    return w


def proximal_operator_l2(g, tv_weight=1.0):
    """Project complex gradients onto the per-dimension L2 bounds."""
    n_dims = g.shape[0]
    weights = normalize_tv_weights(tv_weight, n_dims=n_dims, device=g.device, dtype=g.real.dtype)

    if torch.all(weights == 0):
        return g.clone()

    w_reshaped = weights.view(n_dims, *([1] * (g.ndim - 1)))
    norm = torch.sqrt(torch.sum(g.real**2 + g.imag**2, dim=0, keepdim=True) + 1e-16)
    threshold = torch.clamp(norm / w_reshaped, min=1.0)
    return g / threshold


def canonical_dim(dim, ndim):
    return dim if dim >= 0 else ndim + dim


def resolve_circular_dims(gradient_dims, circular_gradient_dims, ndim):
    grad_canonical = [canonical_dim(int(d), ndim) for d in gradient_dims]
    if circular_gradient_dims is None:
        circular_set = set(grad_canonical)
    else:
        circular_set = {canonical_dim(int(d), ndim) for d in circular_gradient_dims}
    return grad_canonical, circular_set


def set_boundary_zero(x, dim, at_start):
    idx = [slice(None)] * x.ndim
    idx[dim] = 0 if at_start else -1
    x[tuple(idx)] = 0


def gradient(img, gradient_dims, circular_gradient_dims=None):
    """Finite-difference gradient along selected dimensions."""
    grad = torch.zeros((len(gradient_dims), *img.shape), dtype=img.dtype, device=img.device)
    grad_canonical, circular_set = resolve_circular_dims(gradient_dims, circular_gradient_dims, img.ndim)
    for i, (dim, cdim) in enumerate(zip(gradient_dims, grad_canonical)):
        diff = torch.roll(img, shifts=-1, dims=dim) - img
        if cdim not in circular_set:
            set_boundary_zero(diff, dim, at_start=False)
        grad[i] = diff
    return grad


def divergence(p, gradient_dims, circular_gradient_dims=None):
    """Adjoint of finite-difference gradient."""
    div = torch.zeros_like(p[0])
    grad_canonical, circular_set = resolve_circular_dims(gradient_dims, circular_gradient_dims, p.ndim - 1)
    for i, (dim, cdim) in enumerate(zip(gradient_dims, grad_canonical)):
        if cdim in circular_set:
            div += p[i] - torch.roll(p[i], shifts=1, dims=dim)
        else:
            p_zero = p[i].clone()
            idx_last = [slice(None)] * p_zero.ndim
            idx_last[dim] = -1
            p_zero[tuple(idx_last)] = 0
            div += p_zero - torch.roll(p_zero, shifts=1, dims=dim)
    return div


def isotropic_tv_value(img, gradient_dims, circular_gradient_dims=None, reduction="mean"):
    """Compute isotropic TV value from finite differences.

    Args:
        reduction: "sum" for standard TV sum, "mean" for scale-normalized TV.
    """
    grad_x = gradient(img, gradient_dims, circular_gradient_dims=circular_gradient_dims)
    tv_map = torch.sqrt(torch.sum(torch.abs(grad_x) ** 2, dim=0) + 1e-12)
    if reduction == "sum":
        return tv_map.sum()
    if reduction == "mean":
        return tv_map.mean()
    raise ValueError("reduction must be 'sum' or 'mean'")


@torch.no_grad()
def chambolle_pock_tv(x, tv_weight, gradient_dims, L=4.0, circular_gradient_dims=None):
    """Return the TV-denoised image and its mean isotropic TV value."""
    sigma = 0.99 / L
    tau = 0.99 / L
    p = torch.zeros((len(gradient_dims), *x.shape), dtype=x.dtype, device=x.device)
    x_tilde = x.clone()

    for _ in range(5):
        p = proximal_operator_l2(
            p + sigma * gradient(x_tilde, gradient_dims, circular_gradient_dims=circular_gradient_dims),
            tv_weight,
        )
        x_old = x
        x = x + tau * divergence(p, gradient_dims, circular_gradient_dims=circular_gradient_dims)
        x_tilde = 2 * x - x_old

    tv_val = isotropic_tv_value(
        x, gradient_dims, circular_gradient_dims=circular_gradient_dims,
        reduction="mean",
    ).item()
    return x, tv_val


def compute_inverse_flow_fixed_point(full_flow, n_iter=20, chunk_size=4):
    """Compute inverse displacement via fixed-point iteration with slice chunking to conserve VRAM."""
    orig_shape = full_flow.shape
    d_dim, h_dim, w_dim = orig_shape[-4:-1]
    batch_dims = orig_shape[:-4]
    b_dim = int(np.prod(batch_dims)) if len(batch_dims) > 0 else 1
    flat_flow = full_flow.reshape(b_dim, d_dim, h_dim, w_dim, 3)

    device = full_flow.device
    dtype = full_flow.dtype

    scale_x = 2.0 / max(w_dim - 1, 1)
    scale_y = 2.0 / max(h_dim - 1, 1)
    scale_z = 2.0 / max(d_dim - 1, 1)

    v_out = torch.empty_like(flat_flow)

    for b_start in range(0, b_dim, chunk_size):
        b_end = min(b_start + chunk_size, b_dim)
        b_curr = b_end - b_start

        z, y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, d_dim, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, h_dim, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w_dim, device=device, dtype=dtype),
            indexing="ij",
        )
        identity = torch.stack((x, y, z), dim=-1).expand(b_curr, -1, -1, -1, -1).contiguous()
        del z, y, x

        sub_flow = flat_flow[b_start:b_end]
        field_chw = sub_flow.permute(0, 4, 1, 2, 3).contiguous()
        v = -sub_flow.clone()

        for _ in range(int(n_iter)):
            displacement = torch.stack((v[..., 2] * scale_x, v[..., 1] * scale_y, v[..., 0] * scale_z), dim=-1)
            displaced = identity + displacement
            del displacement
            # bspline can also be used here bzt not really necessary
            u_at_query = F.grid_sample(
                field_chw,
                displaced,
                mode="bilinear",
                align_corners=True,
                padding_mode="border",
            ).permute(0, 2, 3, 4, 1).contiguous()
            del displaced
            v = -u_at_query
            del u_at_query

        v_out[b_start:b_end] = v
        del identity, field_chw, v

    return v_out.reshape(orig_shape)


def build_motion_optimizer_groups(model, lr, parameter_mode="all"):
    """Adam parameter groups for a motion model.

    Displacements are always fitted. parameter_mode selects additional parameters:
    "all" includes positions, covariance and blend weights;
    "positions_covariance" includes positions and covariance;
    "covariance" includes covariance only;
    "displacement" includes no additional parameters.
    """
    if parameter_mode not in ("all", "displacement", "positions_covariance", "covariance"):
        raise ValueError(
            "parameter_mode must be 'all', 'displacement', "
            "'positions_covariance', or 'covariance'"
        )
    groups = []
    if isinstance(model, GaussianModel):
        groups.append({"params": model.gaussian_displacement, "lr": lr})
        shape_params = model.covariance_parameters()
        if parameter_mode == "covariance":
            geometry = shape_params
        elif parameter_mode == "positions_covariance":
            geometry = [model.gaussian_position, *shape_params]
        elif parameter_mode == "all":
            geometry = [model.gaussian_position, *shape_params, model.gaussian_weight]
        else:
            geometry = []
        if geometry:
            groups.append({"params": geometry, "lr": lr})
    else:
        raise TypeError("model must be a GaussianModel")
    return groups


def groupwise_adjacent_state_tv(moving, coords, flow, sample_weights=None):
    """Circular neighboring-state TV for a 4D image."""
    query = coords.unsqueeze(0) + flow
    warped = sample_points(moving, query)
    state_dim = 0
    adjacent_differences = (torch.roll(warped, shifts=-1, dims=state_dim) - warped).abs()
    per_voxel = adjacent_differences.sum(dim=state_dim)
    if sample_weights is not None:
        per_voxel = per_voxel * sample_weights
    return per_voxel.mean()


def gaussian_blur_volumes(volumes, sigma, chunk_size=8):
    """Separable Gaussian blur of (*batch, D, H, W) volumes; sigma in voxels, same on all axes.

    Leading dimensions are flattened and processed in chunks to limit memory.
    """
    if sigma <= 0.0:
        return volumes
    if torch.is_complex(volumes):
        return torch.complex(
            gaussian_blur_volumes(volumes.real, sigma, chunk_size),
            gaussian_blur_volumes(volumes.imag, sigma, chunk_size),
        )
    radius = max(1, int(round(3.0 * sigma)))
    axis = torch.arange(-radius, radius + 1, device=volumes.device, dtype=volumes.dtype)
    kernel = torch.exp(-0.5 * (axis / sigma) ** 2)
    kernel = kernel / kernel.sum()
    spatial = volumes.shape[-3:]
    flat = volumes.reshape(-1, *spatial)
    result = torch.empty_like(flat)
    for start in range(0, flat.shape[0], max(1, int(chunk_size))):
        out = flat[start:start + max(1, int(chunk_size)), None]
        for dim in range(3):
            pad = [0] * 6
            pad[2 * (2 - dim)] = pad[2 * (2 - dim) + 1] = radius
            shape = [1, 1, 1, 1, 1]
            shape[2 + dim] = kernel.numel()
            out = F.conv3d(F.pad(out, pad, mode="replicate"), kernel.view(shape))
        result[start:start + out.shape[0]] = out[:, 0]
    return result.reshape(volumes.shape)


@torch.no_grad()
def voxel_sampling_probabilities(abs_image, mode, mask_threshold, floor, blur_sigma=2.0):
    """Per-voxel sampling probabilities: inside the body, or proportional to edge strength.

    Accepts (M, D, H, W); the edge map is built per motion state.
    Edges are measured on a blurred copy so voxel noise is not mistaken for structure uses gaussian_blur_volumes .
    """
    if mode == "body":
        base = foreground_mask(abs_image.mean(dim=0), mask_threshold).float()
    else:
        gradients = torch.gradient(gaussian_blur_volumes(abs_image, float(blur_sigma)), dim=(1, 2, 3))
        base = torch.sqrt(sum(g ** 2 for g in gradients)).mean(dim=0)
        del gradients
    base = base.reshape(-1)
    base = base + float(floor) * base.mean()
    return base / base.sum()


def sampled_det_jacobian(model, coords, shape, eps=1.0):
    """Differentiable det(I + grad(u)): centered inside, one-sided near boundaries."""
    derivatives = []
    for axis, size in enumerate(shape):
        if size == 1:
            derivatives.append(model.flow(coords) * 0.0)
            continue
        step = min(float(eps), float(size - 1))
        position = coords[:, axis]
        lower = coords.clone()
        upper = coords.clone()
        # If a centered stencil would leave the volume, use a one-sided stencil.
        near_lower = position < step
        near_upper = position + step > size - 1
        lower[:, axis] = torch.where(
            near_lower, position, (position - step).clamp_min(0.0))
        upper[:, axis] = torch.where(
            near_lower, (position + step).clamp_max(float(size - 1)),
            torch.where(near_upper, position, position + step))
        spacing = upper[:, axis] - lower[:, axis]
        derivatives.append((model.flow(upper) - model.flow(lower)) / spacing[:, None])
    jac_u = torch.stack(derivatives, dim=-1)
    return torch.linalg.det(jac_u + torch.eye(3, device=coords.device, dtype=jac_u.dtype))


def dense_det_jacobian(full_flow, chunk_size=4):
    """Dense det(I + grad(u)), centered inside and one-sided at boundaries (unit voxels)."""
    orig_shape = full_flow.shape
    d_dim, h_dim, w_dim = orig_shape[-4:-1]
    batch_dims = orig_shape[:-4]
    b_dim = int(np.prod(batch_dims)) if len(batch_dims) > 0 else 1
    flat_flow = full_flow.reshape(b_dim, d_dim, h_dim, w_dim, 3)

    def centered_diff(field, axis):
        if field.shape[axis] == 1:
            return torch.zeros_like(field)
        return torch.gradient(field, dim=(axis,), edge_order=1)[0]

    det_out = torch.empty((b_dim, d_dim, h_dim, w_dim), dtype=full_flow.dtype, device=full_flow.device)
    for b_start in range(0, b_dim, chunk_size):
        b_end = min(b_start + chunk_size, b_dim)
        sub_flow = flat_flow[b_start:b_end]
        dudz = centered_diff(sub_flow, axis=1)
        dudy = centered_diff(sub_flow, axis=2)
        dudx = centered_diff(sub_flow, axis=3)
        jac_u = torch.stack((dudz, dudy, dudx), dim=-1)
        del dudz, dudy, dudx
        eye = torch.eye(3, device=full_flow.device, dtype=full_flow.dtype).view(1, 1, 1, 1, 3, 3)
        jac_phi = jac_u + eye
        del jac_u
        det_out[b_start:b_end] = torch.linalg.det(jac_phi)
        del jac_phi

    return det_out.reshape(*batch_dims, d_dim, h_dim, w_dim)


def motion_fit_image(recon, image_mode):
    """Preserve complex signal for fitting; magnitude is an explicit alternative, works as well but does not hold everything ."""
    image = recon.detach()
    if image_mode == "magnitude":
        image = image.abs()
    return image.to(torch.complex64 if torch.is_complex(image) else torch.float32)


def fit_motion_fields(
    recon,
    model,
    opt,
    coords_all,
    ref_state=0,
    fit_iters=10,
    sample_voxels=16000,
    smooth_weight=10.0,
    ref_free_anchor_state=3,
    ref_free_anchor_weight=0.0,
    ref_free_anchor_hard_enforce=False,
    ref_free_zero_mean_enforce=True,
    seed=0,
    inverse_fixed_point_iters=30,
    jacobian_weight=0.0,
    jacobian_identity_weight=0.0,
    jacobian_det_min=0.0,
    jacobian_sample_voxels=4096,
    jacobian_eps=1.0,
    jacobian_every=5,
    return_loss_breakdown=False,
    sample_mode="uniform",
    sample_floor=0.1,
    image_loss_weight=1.0,
    parameter_mode="all",
    image_mode="complex",
    respiratory_smooth_weight=0.0,
):
    """Fit/refine a caller-owned 4D Gaussian motion model and Adam optimizer.

    Reference-based fitting uses reference-image L1; reference-free fitting uses
    circular adjacent-state TV. Importance sampling estimates the uniform voxel mean.
    parameter_mode selects parameters when opt is None; see build_motion_optimizer_groups.
    respiratory_smooth_weight penalizes squared respiratory coefficient second differences to smooth motion transition.
    """
    with torch.enable_grad():
        fit_image = motion_fit_image(recon, image_mode)
        abs_image = fit_image.abs()
        device = abs_image.device

        n_states, d_dim, h_dim, w_dim = abs_image.shape

        use_reference = ref_state is not None
        ref = int(ref_state) % n_states if use_reference else n_states // 2

        moving = fit_image
        moving_vals = moving.reshape(n_states, -1)
        ref_vals = moving_vals[ref] if use_reference else moving_vals.mean(dim=0)

        if coords_all is None:
            coords_all = make_grid((d_dim, h_dim, w_dim), device, torch.float32)
        if opt is None and model is not None:
            opt = torch.optim.Adam(build_motion_optimizer_groups(model, 1e-2, parameter_mode), lr=1e-2)

        n_vox = coords_all.shape[0]

        gen = torch.Generator(device=device).manual_seed(seed)
        use_subset = 0 < sample_voxels < n_vox
        jacobian_active = float(jacobian_weight) > 0.0 or float(jacobian_identity_weight) > 0.0
        jacobian_every_int = int(jacobian_every)
        if not use_reference and ref_free_zero_mean_enforce:
            model.enforce_zero_mean_displacement()
        if not use_reference and ref_free_anchor_hard_enforce:
            model.enforce_anchor_state_zero(ref_free_anchor_state)

        moving_states_idx = ([s for s in range(n_states) if s != ref]
                     if use_reference else list(range(n_states)))

        if sample_mode != "uniform" and use_reference:
            raise ValueError("sample_mode importance sampling supports reference-free fitting only")
        sample_probs = sample_importance = None
        if use_subset and sample_mode != "uniform":
            sample_probs = voxel_sampling_probabilities(abs_image, sample_mode, 0.1, sample_floor)
            sample_importance = 1.0 / (n_vox * sample_probs)

        for _fit_iter in range(int(fit_iters)):
            if sample_probs is not None:
                idx = torch.multinomial(sample_probs, int(sample_voxels), replacement=True, generator=gen)
                sample_w = sample_importance[idx]
            elif use_subset:
                idx = torch.randint(0, n_vox, (sample_voxels,), device=device, generator=gen)
                sample_w = None
            else:
                idx = torch.arange(n_vox, device=device)
                sample_w = None

            coords = coords_all[idx]
            ref_sub = ref_vals[idx]

            flow_fwd = model.flow(coords)

            coords_expanded = coords.unsqueeze(0) + flow_fwd
            if use_reference:
                sampled_fwd = sample_points(moving, coords_expanded)
                image_loss = torch.mean(torch.abs(sampled_fwd[moving_states_idx] - ref_sub[None]))
            else:
                image_loss = groupwise_adjacent_state_tv(
                    moving, coords, flow_fwd, sample_w
                )

            image_loss = float(image_loss_weight) * image_loss
            smooth_loss = model.displacement_smoothness_loss()
            respiratory_loss = (
                model.respiratory_displacement_smoothness_loss()
                if respiratory_smooth_weight else smooth_loss * 0.0
            )
            anchor_bias_loss = smooth_loss * 0.0
            if not use_reference and float(ref_free_anchor_weight) > 0.0:
                anchor_bias_loss = model.anchor_state_bias_loss(ref_free_anchor_state)
            loss = (
                image_loss
                + float(smooth_weight) * smooth_loss
                + float(respiratory_smooth_weight) * respiratory_loss
                + float(ref_free_anchor_weight) * anchor_bias_loss
            )
            if jacobian_active and (_fit_iter % jacobian_every_int == 0):
                if idx.numel() > int(jacobian_sample_voxels) > 0:
                    perm = torch.randperm(idx.numel(), device=device, generator=gen)[: int(jacobian_sample_voxels)]
                    coords_j = coords[perm]
                else:
                    coords_j = coords
                det_jac = sampled_det_jacobian(
                    model, coords_j, (d_dim, h_dim, w_dim), eps=jacobian_eps)

                if float(jacobian_weight) > 0.0:
                    inv_loss = F.relu(float(jacobian_det_min) - det_jac).pow(2).mean()
                    loss = loss + float(jacobian_weight) * inv_loss

                if float(jacobian_identity_weight) > 0.0:
                    id_loss = (det_jac - 1.0).pow(2).mean()
                    loss = loss + float(jacobian_identity_weight) * id_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if use_reference:
                model.enforce_anchor_state_zero(ref)
            elif ref_free_zero_mean_enforce:
                model.enforce_zero_mean_displacement()
            if not use_reference and ref_free_anchor_hard_enforce:
                model.enforce_anchor_state_zero(ref_free_anchor_state)

        with torch.no_grad():
            full_flow = model.full_flow(coords_all)
            if use_reference:
                full_flow[ref] = 0.0
            elif ref_free_zero_mean_enforce:
                full_flow.sub_(full_flow.mean(dim=0, keepdim=True))
            if not use_reference and ref_free_anchor_hard_enforce:
                anchor = int(ref_free_anchor_state) % n_states
                full_flow[anchor] = 0.0
            full_flow_inv = compute_inverse_flow_fixed_point(full_flow, n_iter=inverse_fixed_point_iters)
            if use_reference:
                full_flow_inv[ref] = 0.0
            elif ref_free_anchor_hard_enforce:
                anchor = int(ref_free_anchor_state) % n_states
                full_flow_inv[anchor] = 0.0

            if not use_reference:
                image_loss_val = float(image_loss.detach().item())
            else:
                query_fwd_full = coords_all.unsqueeze(0) + full_flow.reshape(n_states, -1, 3)
                sampled_fwd_full = sample_points(moving, query_fwd_full, chunk_size=20_000)
                image_loss_val = float(torch.mean(torch.abs(sampled_fwd_full[moving_states_idx] - ref_vals[None, :])).item())

            smooth_loss_val = float(model.displacement_smoothness_loss().item())
            smooth_reg_val = float(smooth_weight) * smooth_loss_val
            respiratory_reg_val = (
                float(respiratory_smooth_weight) * float(model.respiratory_displacement_smoothness_loss())
                if respiratory_smooth_weight else 0.0
            )
            anchor_bias_reg_val = 0.0
            if not use_reference and float(ref_free_anchor_weight) > 0.0:
                anchor_bias_reg_val = float(ref_free_anchor_weight) * float(model.anchor_state_bias_loss(ref_free_anchor_state).item())

            jacobian_floor_val = 0.0
            jacobian_identity_val = 0.0
            if jacobian_active:
                det_jac_full = dense_det_jacobian(full_flow)
                if float(jacobian_weight) > 0.0:
                    inv_loss_val = float(F.relu(float(jacobian_det_min) - det_jac_full).pow(2).mean().item())
                    jacobian_floor_val = float(jacobian_weight) * inv_loss_val
                if float(jacobian_identity_weight) > 0.0:
                    id_loss_val = float((det_jac_full - 1.0).pow(2).mean().item())
                    jacobian_identity_val = float(jacobian_identity_weight) * id_loss_val

            reg_total_val = (
                smooth_reg_val
                + respiratory_reg_val
                + anchor_bias_reg_val
                + jacobian_floor_val
                + jacobian_identity_val
            )
            loss_breakdown = {
                "image_loss": image_loss_val,
                "smoothness": smooth_reg_val,
                "respiratory_smoothness": respiratory_reg_val,
                "anchor_bias": anchor_bias_reg_val,
                "jacobian_floor": jacobian_floor_val,
                "jacobian_identity": jacobian_identity_val,
                "regularization": reg_total_val,
                "loss": image_loss_val + reg_total_val,
            }

    if return_loss_breakdown:
        return full_flow_inv, full_flow, loss_breakdown

    return full_flow_inv, full_flow
