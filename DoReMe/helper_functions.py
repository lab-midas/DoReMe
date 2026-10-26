"""Signal normalization, coordinate grids, masks, and complex point sampling."""

import numpy as np
import torch
import torch.nn.functional as F


def signal_magnitude_scale(x: torch.Tensor, percentage: float = 100.0) -> float:
    """Estimate signal scale from a percentile of voxel magnitudes."""
    if not torch.is_tensor(x):
        raise TypeError("x must be a torch.Tensor")
    if x.numel() == 0:
        raise ValueError("x must not be empty")
    percentage = min(max(float(percentage), 0.0), 100.0)
    power = torch.abs(x).pow(2).flatten()
    quantile_power = power.max() if percentage >= 100.0 else torch.quantile(power, percentage / 100.0)
    return float(torch.sqrt(quantile_power).item())


def scale_tv_weight_by_signal(
    tv_weight: float,
    lipschitz_L: float,
    gradient_step_image: torch.Tensor,
    percentage: float = 100.0,
) -> float:
    """Scale the TV weight by the signal magnitude of the first image update."""
    if lipschitz_L <= 0:
        raise ValueError("lipschitz_L must be > 0")
    scale = signal_magnitude_scale(gradient_step_image, percentage)
    return float(tv_weight) * scale


def make_grid(shape, device, dtype):
    """Return flattened voxel coordinate grid (N, 3) ordered as (z, y, x)."""
    d, h, w = shape
    z = torch.arange(d, device=device, dtype=dtype)
    y = torch.arange(h, device=device, dtype=dtype)
    x = torch.arange(w, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack([zz, yy, xx], dim=-1).reshape(-1, 3)


def foreground_mask(vol, threshold_rel=0.1):
    """Build a boolean foreground mask from a magnitude volume."""
    return vol > (threshold_rel * vol.max())


def sample_points(mov, query, chunk_size=None):
    """Sample (*batch, Z, Y, X) volumes at (*batch, N, 3) voxel coordinates.

    Coordinates are ordered (z, y, x). Complex channels share the same grid.
    chunk_size limits the number of query points interpolated at once.
    """
    if chunk_size is not None and int(chunk_size) > 0 and query.shape[-2] > int(chunk_size):
        chunks = [sample_points(mov, query[..., start:start + int(chunk_size), :], chunk_size=None)
                  for start in range(0, query.shape[-2], int(chunk_size))]
        return torch.cat(chunks, dim=-1)

    if torch.is_complex(mov):
        return torch.complex(sample_points(mov.real, query), sample_points(mov.imag, query))

    batch_shape = mov.shape[:-3]
    b = int(np.prod(batch_shape)) if len(batch_shape) > 0 else 1
    d, h, w = mov.shape[-3:]

    mov_flat = mov.reshape(b, 1, d, h, w)
    query_flat = query.reshape(b, -1, 3)

    zc, yc, xc = query_flat.unbind(dim=-1)
    xn = 2.0 * xc / max(w - 1, 1) - 1.0
    yn = 2.0 * yc / max(h - 1, 1) - 1.0
    zn = 2.0 * zc / max(d - 1, 1) - 1.0
    grid = torch.stack([xn, yn, zn], dim=-1).view(b, -1, 1, 1, 3)
    out = F.grid_sample(mov_flat, grid, mode="bilinear",
                        align_corners=True, padding_mode="border")
    return out.view(*batch_shape, -1)
