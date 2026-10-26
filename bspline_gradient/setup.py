import os
import sys
from pathlib import Path

from setuptools import setup


if sys.platform.startswith("win") and not os.environ.get("CUDA_HOME"):
    cuda_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    cuda_versions = sorted(cuda_root.glob("v*"), reverse=True) if cuda_root.exists() else []
    if cuda_versions:
        os.environ["CUDA_HOME"] = str(cuda_versions[0])

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def compile_args():
    if sys.platform.startswith("win"):
        return {
            "cxx": ["/std:c++17", "/O2"],
            "nvcc": ["-O3", "--use_fast_math"],
        }

    return {
        "cxx": ["-std=c++17", "-O3"],
        "nvcc": ["-std=c++17", "-O3", "--use_fast_math"],
    }

setup(
    name='bspline_interp_withgrad',
    ext_modules=[
        CUDAExtension(
            name='bspline_interp_withgrad',
            sources=[
                'bspline_cuda_autograd.cpp',
                'bspline_cuda_kernel.cu',
                'bspline_cuda_optimized.cu',
            ],
            extra_compile_args=compile_args(),
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
