"""Run DoReMe with synthetic tensors; no dataset files are read or written."""

import torch

from DoReMe import DoReMe


DEVICE = "cuda"
COILS = 2
N_READOUTS = 256
IMAGE_SHAPE = (4, 16, 16, 16)  # (motion_states, Z, Y, X)
SEED = 0

DATA_CONSISTENCY_STEP_SIZE = 0.5
TV_WEIGHT = 0.05
MAX_ITER = 30
GRADIENT_DIMS = [0]
ENABLE_MOTION_ESTIMATION = True

MOTION_COVARIANCE = "diagonal"
MOTION_REFERENCE_FREE = True  # false will use MOTION_REF_STATE to compute loss-
MOTION_REF_STATE = 0
MOTION_REF_FREE_ANCHOR_STATE = 3
MOTION_REF_FREE_ANCHOR_WEIGHT = 0.0
MOTION_REF_FREE_ANCHOR_HARD_ENFORCE = False # you use this to anchor 1 state
MOTION_REF_FREE_ZERO_MEAN_ENFORCE = True   # use this to not anchor any state. 
MOTION_FIT_START_ITER = 0
MOTION_FIT_INTERVAL = 1
MOTION_FIT_FIRST_ITERS = 3
MOTION_FIT_ITERS = 3
MOTION_FIT_LR = 5e-2
MOTION_FIT_PARAMETER_MODE = "all"  # "all", "displacement", "positions_covariance", or "covariance".
MOTION_FIT_COMPONENTS = 64
MOTION_FIT_NEIGHBORS = 16
MOTION_FIT_SAMPLE_VOXELS = 512
MOTION_SAMPLE_MODE = "uniform"  # "uniform", "body", or "edge"; body/edge require reference-free fitting.
MOTION_SMOOTH_WEIGHT = 1.0
MOTION_FIT_IMAGE_MODE = "complex"

ENABLE_JACOBIAN = True
MOTION_JACOBIAN_WEIGHT = 0.1
MOTION_JACOBIAN_IDENTITY_WEIGHT = 0.0
MOTION_JACOBIAN_DET_MIN = 0.2
MOTION_JACOBIAN_SAMPLE_VOXELS = 128
MOTION_JACOBIAN_EPS = 1.0
MOTION_INIT_METHOD = "random"
MOTION_MASK_THRESHOLD = 0.1
MOTION_INVERSE_FIXED_POINT_ITERS = 10


def make_dummy_tensors(
    coils=COILS,
    n_readouts=N_READOUTS,
    image_shape=IMAGE_SHAPE,
    device=DEVICE,
    seed=SEED,
):
    """Return readouts, coil maps, sampling coordinates, and image shape."""
    if len(image_shape) != 4 or any(type(size) is not int or size < 1 for size in image_shape):
        raise ValueError("image_shape must contain four positive integers")
    if type(coils) is not int or coils < 1 or type(n_readouts) is not int or n_readouts < 1:
        raise ValueError("coils and n_readouts must be positive integers")

    motion_states, z_size, y_size, x_size = image_shape
    generator = torch.Generator().manual_seed(seed)
    readouts = 0.01 * torch.randn(
        coils, n_readouts, x_size, dtype=torch.complex64, generator=generator)
    coil_maps = torch.randn(
        coils, z_size, y_size, x_size, dtype=torch.complex64, generator=generator)
    coil_maps = coil_maps / coil_maps.abs().square().sum(dim=0, keepdim=True).sqrt().clamp_min(1e-8)

    sampling_coordinates = torch.zeros(n_readouts, 3, dtype=torch.int64)
    sampling_coordinates[:, 0] = torch.randint(y_size, (n_readouts,), generator=generator)
    sampling_coordinates[:, 1] = torch.randint(z_size, (n_readouts,), generator=generator)
    sampling_coordinates[:, 2] = torch.arange(n_readouts) % motion_states

    return (
        readouts.to(device),
        coil_maps.to(device),
        sampling_coordinates.to(device),
        tuple(image_shape),
    )


def run_single_reconstruction(device=DEVICE):
    """Run the configured synthetic example and return DoReMe's result tuple."""
    compute_device = torch.device(device)
    if compute_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("DoReMe requires CUDA and the compiled B-spline extension")

    readouts, coil_maps, sampling_coordinates, image_shape = make_dummy_tensors(
        coils=COILS, n_readouts=N_READOUTS, image_shape=IMAGE_SHAPE,
        device=compute_device, seed=SEED,
    )
    with torch.no_grad():
        result = DoReMe(
            readouts,
            coil_maps,
            sampling_coordinates,
            image_shape,
            data_consistency_step_size=DATA_CONSISTENCY_STEP_SIZE,
            tv_weight=TV_WEIGHT,
            max_iter=MAX_ITER,
            gradient_dims=GRADIENT_DIMS,
            enable_motion_estimation=ENABLE_MOTION_ESTIMATION,
            motion_covariance=MOTION_COVARIANCE,
            motion_fit_parameter_mode=MOTION_FIT_PARAMETER_MODE,
            motion_ref_state=None if MOTION_REFERENCE_FREE else MOTION_REF_STATE,
            motion_ref_free_anchor_state=MOTION_REF_FREE_ANCHOR_STATE,
            motion_ref_free_anchor_weight=MOTION_REF_FREE_ANCHOR_WEIGHT,
            motion_ref_free_anchor_hard_enforce=MOTION_REF_FREE_ANCHOR_HARD_ENFORCE,
            motion_ref_free_zero_mean_enforce=MOTION_REF_FREE_ZERO_MEAN_ENFORCE,
            motion_fit_start_iter=MOTION_FIT_START_ITER,
            motion_fit_interval=MOTION_FIT_INTERVAL,
            motion_fit_first_iters=MOTION_FIT_FIRST_ITERS,
            motion_fit_iters=MOTION_FIT_ITERS,
            motion_fit_lr=MOTION_FIT_LR,
            motion_fit_components=MOTION_FIT_COMPONENTS,
            motion_fit_neighbors=MOTION_FIT_NEIGHBORS,
            motion_fit_sample_voxels=MOTION_FIT_SAMPLE_VOXELS,
            motion_sample_mode=MOTION_SAMPLE_MODE,
            motion_smooth_weight=MOTION_SMOOTH_WEIGHT,
            motion_jacobian_weight=MOTION_JACOBIAN_WEIGHT if ENABLE_JACOBIAN else 0.0,
            motion_jacobian_identity_weight=MOTION_JACOBIAN_IDENTITY_WEIGHT if ENABLE_JACOBIAN else 0.0,
            motion_jacobian_det_min=MOTION_JACOBIAN_DET_MIN,
            motion_jacobian_sample_voxels=MOTION_JACOBIAN_SAMPLE_VOXELS,
            motion_jacobian_eps=MOTION_JACOBIAN_EPS,
            motion_fit_image_mode=MOTION_FIT_IMAGE_MODE,
            motion_init_method=MOTION_INIT_METHOD,
            motion_mask_threshold=MOTION_MASK_THRESHOLD,
            motion_inverse_fixed_point_iters=MOTION_INVERSE_FIXED_POINT_ITERS,
            return_first_motion_fields=True,
        )
    print(f"Reconstructed image shape: {tuple(result[0].shape)}")
    return result


if __name__ == "__main__":
    run_single_reconstruction()
