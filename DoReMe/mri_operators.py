"""4D MRI encoding/adjoint and real or complex B-spline warping."""

import torch
import torch.nn.functional as F

import bspline_interp_withgrad


def readouts_to_kspace(readouts, sampling_coordinates, image_shape):
    """Accumulate readouts (coils, N, X) into full k-space (coils, M, Z, Y, X)."""
    coils, n_readouts, col = readouts.shape
    kspace_out_vec = torch.zeros((coils,) + image_shape, dtype=readouts.dtype, device=readouts.device)

    _, m_dim, z_dim, y_dim, x_dim = kspace_out_vec.shape
    y = sampling_coordinates[:, 0].long()
    z = sampling_coordinates[:, 1].long()
    m = sampling_coordinates[:, 2].long()

    m_exp = m.view(1, -1, 1).expand(coils, n_readouts, col)
    z_exp = z.view(1, -1, 1).expand(coils, n_readouts, col)
    y_exp = y.view(1, -1, 1).expand(coils, n_readouts, col)
    x_exp = torch.arange(col, device=readouts.device).view(1, 1, -1).expand(coils, n_readouts, col)

    linear_idx = (m_exp * z_dim * y_dim * x_dim) + (z_exp * y_dim * x_dim) + (y_exp * x_dim) + x_exp

    src = readouts.view(coils, -1)
    idx = linear_idx.view(coils, -1)

    kspace_out_flat = kspace_out_vec.view(coils, -1)
    kspace_out_flat.scatter_add_(dim=1, index=idx, src=src)
    return kspace_out_flat.view(coils, m_dim, z_dim, y_dim, x_dim)


def kspace_to_readouts(kspace, sampling_coordinates):
    """Sample k-space (coils, M, Z, Y, X) at readout coordinates."""
    coils, _, _, _, _ = kspace.shape
    n_readouts, _ = sampling_coordinates.shape
    y = sampling_coordinates[:, 0].long()
    z = sampling_coordinates[:, 1].long()
    m = sampling_coordinates[:, 2].long()

    coil_idx = torch.arange(coils, device=kspace.device).view(-1, 1).expand(coils, n_readouts)
    m_idx = m.view(1, -1).expand(coils, n_readouts)
    z_idx = z.view(1, -1).expand(coils, n_readouts)
    y_idx = y.view(1, -1).expand(coils, n_readouts)
    return kspace[coil_idx, m_idx, z_idx, y_idx, :]


def A_forward(img, S, sampling_coordinates):
    coils, lin, par, col = S.shape
    motion_states = img.shape[0]
    s_expanded = S.unsqueeze(1).expand(coils, motion_states, lin, par, col)
    img_encoded = img.unsqueeze(0).mul(s_expanded)
    kspace_full = torch.fft.ifftn(img_encoded, dim=(-3, -2, -1))
    return kspace_to_readouts(kspace_full, sampling_coordinates)


def A_adjoint(synthetic_readouts, S, sampling_coordinates, image_shape):
    kspace_filled = readouts_to_kspace(synthetic_readouts, sampling_coordinates, image_shape)
    img_coil_space = torch.fft.fftn(kspace_filled, dim=(-3, -2, -1))
    return torch.sum(img_coil_space.mul_(torch.conj(S.unsqueeze(1))), dim=0)


def pad_image_for_bspline(img: torch.Tensor, padding: int = 2) -> torch.Tensor:
    """Pad (Z, Y, X) axes symmetrically for cubic B-spline support."""
    return F.pad(img, pad=(padding, padding, padding, padding, padding, padding), mode="reflect")


def warp_3d_batch_bspline(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp a 3D image with cubic B-spline interpolation."""
    pad = 2
    m_dim, d_dim, h_dim, w_dim = img.shape
    is_complex = torch.is_complex(img)

    img_real = img.real if is_complex else img
    img_real = pad_image_for_bspline(img_real, pad)
    img_real = bspline_interp_withgrad.bspline_prefilter_autograd(img_real.contiguous())

    if is_complex:
        img_imag = pad_image_for_bspline(img.imag, pad)
        img_imag = bspline_interp_withgrad.bspline_prefilter_autograd(img_imag.contiguous())

    z = torch.arange(d_dim, device=img.device)
    y = torch.arange(h_dim, device=img.device)
    x = torch.arange(w_dim, device=img.device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    grid = torch.stack((zz, yy, xx), dim=-1).float().unsqueeze(0).repeat(m_dim, 1, 1, 1, 1)

    coords = grid + flow + pad

    warped_real = bspline_interp_withgrad.bspline_interp_autograd(
        img_real.contiguous(),
        coords.contiguous(),
    )
    if is_complex:
        warped_imag = bspline_interp_withgrad.bspline_interp_autograd(
            img_imag.contiguous(),
            coords.contiguous(),
        )
        return warped_real + 1j * warped_imag
    return warped_real
