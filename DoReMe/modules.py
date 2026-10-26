import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

COVARIANCE_TYPES = ("isotropic", "diagonal", "full")


def inverse_softplus(value):
    return float(np.log(np.expm1(max(float(value), 1e-6))))


class GaussianModel(nn.Module):
    """Per-state Gaussian displacement field with configurable covariance.

    Axis-aligned widths use softplus(width) + sigma_floor. Full covariance uses
    L L^T + sigma_floor^2 I, with a learned lower-triangular factor L.
    """

    def __init__(self, shape, n_states, n_components=5000, neighbors=64,
                 sigma_floor=1.5, seed=0, init_sigma=None, covariance="diagonal"):
        super().__init__()
        d, h, w = shape
        self.shape = (d, h, w)
        self.neighbors = neighbors
        self.sigma_floor = sigma_floor
        if type(n_states) is not int or n_states < 1:
            raise ValueError("n_states must be a positive integer")
        self.n_states = n_states

        g = torch.Generator().manual_seed(seed)
        init_position = (torch.rand(n_components, 3, generator=g)
                         * torch.tensor([d - 1, h - 1, w - 1], dtype=torch.float32))
        self.gaussian_position = nn.Parameter(init_position)
        if init_sigma is None:
            init_sigma = max(2.0, 0.04 * min(d, h, w))
        self.init_kernel_parameters(n_components, init_sigma, covariance)
        self.gaussian_weight = nn.Parameter(torch.zeros(n_components))
        self.gaussian_displacement = nn.Parameter(torch.zeros(n_states, n_components, 3))

    def init_kernel_parameters(self, n_components, init_sigma, covariance):
        if covariance not in COVARIANCE_TYPES:
            raise ValueError(f"covariance must be one of {COVARIANCE_TYPES}, got {covariance!r}")
        self.covariance = str(covariance)
        width_shape = (n_components, 1) if self.covariance == "isotropic" else (n_components, 3)
        if self.covariance == "full":
            # Match the diagonal model's effective width (init_sigma + floor) at initialisation.
            effective = float(init_sigma) + float(self.sigma_floor)
            diag_init = float(np.sqrt(max(effective ** 2 - float(self.sigma_floor) ** 2, 1e-6)))
            self.gaussian_width_param = nn.Parameter(
                torch.full(width_shape, inverse_softplus(diag_init)))
            self.gaussian_shape_offdiag = nn.Parameter(torch.zeros(n_components, 3))
        else:
            self.gaussian_width_param = nn.Parameter(
                torch.full(width_shape, inverse_softplus(init_sigma)))

    def covariance_parameters(self):
        """Kernel-shape parameters (widths and, for 'full', off-diagonals)."""
        params = [self.gaussian_width_param]
        if self.covariance == "full":
            params.append(self.gaussian_shape_offdiag)
        return params

    def inverse_covariance(self):
        """(K, 3, 3) inverse covariance for covariance='full'."""
        diag = F.softplus(self.gaussian_width_param)  # (K, 3)
        off = self.gaussian_shape_offdiag  # (K, 3)
        zero = torch.zeros_like(diag[:, 0])
        lower = torch.stack([
            torch.stack([diag[:, 0], zero, zero], dim=-1),
            torch.stack([off[:, 0], diag[:, 1], zero], dim=-1),
            torch.stack([off[:, 1], off[:, 2], diag[:, 2]], dim=-1),
        ], dim=-2)
        eye = torch.eye(3, device=diag.device, dtype=diag.dtype)
        covariance = lower @ lower.transpose(-1, -2) + (float(self.sigma_floor) ** 2) * eye
        return torch.linalg.inv(covariance)

    def squared_kernel_distance(self, diff, index=None):
        """Squared kernel distance of diff (N, K', 3) to Gaussians index (N, K'), or to all K if None."""
        if self.covariance == "full":
            precision = self.inverse_covariance()
            if index is not None:
                return torch.einsum("nki,nkij,nkj->nk", diff, precision[index], diff)
            return torch.einsum("nki,kij,nkj->nk", diff, precision, diff)
        sigma = F.softplus(self.gaussian_width_param) + self.sigma_floor  # (K, 3) or (K, 1)
        sigma = sigma[index] if index is not None else sigma[None]
        return torch.sum((diff / sigma) ** 2, dim=-1)


    def nearest_gaussian_indices(self, coords, knn_chunk=20_000):
        """Top-k nearest Gaussian indices per voxel."""
        positions = self.gaussian_position.detach()
        out = []
        for start in range(0, coords.shape[0], knn_chunk):
            block = coords[start:start + knn_chunk]
            d2 = torch.cdist(block, positions)
            out.append(torch.topk(d2, self.neighbors, dim=-1, largest=False).indices)
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def initialize_positions_from_mask(self, mask, grid, bg_ratio=0.2):
        """Initialise Gaussian positions from coordinates inside the mask plus uniform background."""
        n_gauss, n_dim = self.gaussian_position.shape
        if mask.dtype != torch.bool:
            raise TypeError(f"mask must be a bool tensor, got {mask.dtype}")
        if grid.shape[-1] != n_dim:
            raise ValueError(f"grid last dim must be {n_dim}, got {grid.shape[-1]}")
        if grid.shape[:-1] != mask.shape:
            raise ValueError(f"grid spatial shape {grid.shape[:-1]} != mask shape {mask.shape}")

        fg_coords = grid[mask].to(self.gaussian_position.dtype)
        all_coords = grid.reshape(-1, n_dim).to(self.gaussian_position.dtype)

        n_bg = max(1, int(n_gauss * bg_ratio))
        n_fg = n_gauss - n_bg

        if fg_coords.shape[0] > 0:
            if fg_coords.shape[0] > n_fg:
                idx_fg = torch.randperm(fg_coords.shape[0], device=fg_coords.device)[:n_fg]
                chosen_fg = fg_coords[idx_fg]
            else:
                repeats = (n_fg + fg_coords.shape[0] - 1) // fg_coords.shape[0]
                chosen_fg = fg_coords.repeat(repeats, 1)[:n_fg]
        else:
            chosen_fg = all_coords[:n_fg]

        idx_bg = torch.randperm(all_coords.shape[0], device=all_coords.device)[:n_bg]
        chosen_bg = all_coords[idx_bg]

        chosen = torch.cat([chosen_fg, chosen_bg], dim=0)
        self.gaussian_position.copy_(chosen)

    def displacement_smoothness_loss(self, neighbors=8):
        """Mean squared displacement differences between neighboring centers."""
        with torch.no_grad():
            dist = torch.cdist(
                self.gaussian_position,
                self.gaussian_position
            )                                   # (K,K)

            nbr = torch.topk(
                dist,
                k=neighbors + 1,
                largest=False
            ).indices[:, 1:]                   # (K,neighbors)

        disp_i = self.gaussian_displacement[:, :, None, :]
        disp_j = self.gaussian_displacement[:, nbr, :]

        return ((disp_i - disp_j) ** 2).mean()

    def respiratory_displacement_smoothness_loss(self):
        """Open-chain curvature across respiratory states at fixed Gaussian index."""
        if self.gaussian_displacement.shape[0] < 3:
            return self.gaussian_displacement.sum() * 0.0
        return self.gaussian_displacement.diff(n=2, dim=0).square().mean()

    def enforce_zero_mean_displacement(self):
        """Remove the mean control-point displacement across motion states."""
        with torch.no_grad():
            mean_over_states = self.gaussian_displacement.mean(dim=0, keepdim=True)
            self.gaussian_displacement.sub_(mean_over_states)

    def anchor_state_bias_loss(self, anchor_state=3):
        """Softly bias one state toward zero displacement (ref-free only)."""
        anchor = int(anchor_state) % self.n_states
        return self.gaussian_displacement[anchor].square().mean()

    def enforce_anchor_state_zero(self, anchor_state=3):
        """Hard constraint: set one state's displacement exactly to zero."""
        anchor = int(anchor_state) % self.n_states
        with torch.no_grad():
            self.gaussian_displacement[anchor].zero_()

    def flow(self, coords):
        """Displacement (M, N, 3), ordered as (dz, dy, dx)."""
        use_knn = 0 < self.neighbors < self.gaussian_position.shape[0]
        if use_knn:
            nbr = self.nearest_gaussian_indices(coords)
            nbr_position = self.gaussian_position[nbr]
            nbr_weight = self.gaussian_weight[nbr]
            diff = coords[:, None, :] - nbr_position
            sq_dist = self.squared_kernel_distance(diff, nbr)
            blend = torch.softmax(nbr_weight - 0.5 * sq_dist, dim=-1)
            nbr_displacement = self.gaussian_displacement[:, nbr, :]
            return (nbr_displacement * blend[None, :, :, None]).sum(dim=2)
        diff = coords[:, None, :] - self.gaussian_position[None, :, :]
        sq_dist = self.squared_kernel_distance(diff)
        blend = torch.softmax(self.gaussian_weight[None, :] - 0.5 * sq_dist, dim=-1)
        return (self.gaussian_displacement[:, None, :, :] * blend[None, :, :, None]).sum(dim=2)

    @torch.no_grad()
    def full_flow(self, coords_all, chunk=20_000):
        """Evaluate the full field as (M, Z, Y, X, 3)."""
        n = coords_all.shape[0]
        flow = torch.empty(self.n_states, n, 3, device=coords_all.device)
        for start in range(0, n, chunk):
            sl = slice(start, start + chunk)
            flow[:, sl, :] = self.flow(coords_all[sl])
        return flow.reshape(self.n_states, *self.shape, 3)
