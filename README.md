# DoReMe

**Dual optimization of Reconstruction and Motion estimation** for
self-navigated, free-breathing abdominal MRI using Gaussian Splatting.

DoReMe alternates between multi-coil image reconstruction and nonrigid
respiratory motion estimation. Compact Gaussian primitives share spatial
parameters across states, with state-specific displacement vectors describing
motion. Motion-aligned total variation regularization shares information
between states while accounting for anatomical displacement, and both images
and motion fields are refined throughout reconstruction.

## Getting started

Requires an NVIDIA GPU, CUDA-enabled PyTorch, NumPy, and a compatible CUDA
toolkit and C++ compiler. From the repository root, install the B-spline
extension and run the dummy example:

```powershell
cd bspline_gradient
python -m pip install --no-build-isolation .
cd ..
python run.py
```

[run.py](./run.py) checks the pipeline with random tensors; no dataset is
required. It does not reproduce the paper's experimental results, and the
study data cannot be made publicly available.

### Configuration

Settings are at the top of [run.py](./run.py). Motion fitting has many
configurations to explore: Gaussian covariance, component and neighbor counts,
voxel sampling, fitting schedules, reference-based or reference-free alignment,
and smoothness or Jacobian penalties. The defaults are a starting point,
not a universal configuration.

**Memory usage:** The algorithm is not optimized for memory. Large volumes,
many motion states, or large Gaussian models can require substantial GPU
memory.

## Repository

- [run.py](./run.py): dummy example entry point.
- [DoReMe/](./DoReMe/)
  - [DoReMe.py](./DoReMe/DoReMe.py): main reconstruction loop.
  - [modules.py](./DoReMe/modules.py): Gaussian motion model, covariance options, and displacement constraints.
  - [computational_functions.py](./DoReMe/computational_functions.py): TV regularization, motion fitting, field inversion, and Jacobian penalties.
  - [mri_operators.py](./DoReMe/mri_operators.py): multi-coil MRI encoding, k-space/readout conversion, and image warping.
  - [helper_functions.py](./DoReMe/helper_functions.py): signal scaling, coordinate grids, masks, and point sampling.
- [bspline_gradient/](./bspline_gradient/)
  - [bspline_cuda_autograd.cpp](./bspline_gradient/bspline_cuda_autograd.cpp): PyTorch bindings and automatic differentiation.
  - [bspline_cuda_kernel.cu](./bspline_gradient/bspline_cuda_kernel.cu): CUDA interpolation and gradient kernels.
  - [bspline_cuda_optimized.cu](./bspline_gradient/bspline_cuda_optimized.cu): parallel B-spline prefilter kernels.
  - [setup.py](./bspline_gradient/setup.py): builds and installs the CUDA extension.
- [Docker/](./Docker/)
  - [Dockerfile](./Docker/Dockerfile): CUDA/PyTorch container environment.
  - [requirements.txt](./Docker/requirements.txt): Python dependencies for the container.

## Citation

If you use DoReMe in your research, please cite:

Majd Helo, Dominik Nickel, and Thomas Kuestner.
**DoReMe: Dual optimization of Reconstruction and Motion estimation for
self-navigated free-breathing abdominal MRI using Gaussian Splatting.**
[arXiv:2610.19120](https://arxiv.org/abs/2610.19120).
