

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <cmath>


__device__ __forceinline__ float cubic_beta_f(float x) {
    x = fabsf(x);
    if (x < 1.0f)      return (2.0f/3.0f) - 0.5f * x * x * (2.0f - x);
    else if (x < 2.0f) return ((2.0f - x)*(2.0f - x)*(2.0f - x)) / 6.0f;
    return 0.0f;
}

__device__ __forceinline__ float cubic_beta_prime_f(float x) {
    float ax = fabsf(x);
    if (ax < 1.0f) {
        // -2*x + 1.5 * x * |x|
        return -2.0f * x + 1.5f * x * ax;
    } else if (ax < 2.0f) {
        // -(1/2) * (2-ax)^2 * sign(x)
        float t = 2.0f - ax;
        float s = (x < 0 ? -1.0f : 1.0f);
        return -0.5f * t * t * s;
    } else {
        return 0.0f;
    }
}

__global__ void bspline_interp_kernel_f32(
    const float* __restrict__ img,
    const float* __restrict__ coords,
    float*       __restrict__ output,
    int M, int D, int H, int W,
    int Dp, int Hp, int Wp
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int bz = blockIdx.z;
    int z = bz % D;
    int m = bz / D;
    if (x>=W||y>=H||z>=D||m>=M) return;

    int out_idx = ((m*D + z)*H + y)*W + x;
    const float* coord = coords + out_idx*3;
    float zf=coord[0], yf=coord[1], xf=coord[2];
    int iz=floorf(zf), iy=floorf(yf), ix=floorf(xf);
    float dz=zf-iz, dy=yf-iy, dx=xf-ix;

    float result=0.0f;
    for(int mz=-1;mz<=2;++mz){
      int sz = min(max(iz+mz,0),Dp-1);
      float wz = cubic_beta_f(dz-mz);
      for(int my=-1;my<=2;++my){
        int sy = min(max(iy+my,0),Hp-1);
        float wy = cubic_beta_f(dy-my);
        for(int mx=-1;mx<=2;++mx){
          int sx = min(max(ix+mx,0),Wp-1);
          float wx = cubic_beta_f(dx-mx);
          float w = wz*wy*wx;
          int img_idx = ((m*Dp + sz)*Hp + sy)*Wp + sx;
          result += img[img_idx] * w;
        }
      }
    }
    output[out_idx] = result;
}

__global__ void bspline_interp_grad_both_kernel_f32(
    const float* __restrict__ img,
    const float* __restrict__ grad_out,
    const float* __restrict__ coords,
    float*       __restrict__ grad_img,
    float*       __restrict__ grad_coords,
    int M, int D, int H, int W,
    int Dp, int Hp, int Wp
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int bz = blockIdx.z;
    int z = bz % D;
    int m = bz / D;
    if (x>=W||y>=H||z>=D||m>=M) return;

    int out_idx = ((m*D + z)*H + y)*W + x;

    const float* coord = coords + out_idx*3;
    float zf=coord[0], yf=coord[1], xf=coord[2];
    int iz=floorf(zf), iy=floorf(yf), ix=floorf(xf);
    float dz=zf-iz, dy=yf-iy, dx=xf-ix;
    float g = grad_out[out_idx];

    for(int mz=-1;mz<=2;++mz){
      int sz = min(max(iz+mz,0),Dp-1);
      float wz=cubic_beta_f(dz-mz), pwz=cubic_beta_prime_f(dz-mz);
      for(int my=-1;my<=2;++my){
        int sy=min(max(iy+my,0),Hp-1);
        float wy=cubic_beta_f(dy-my), pwy=cubic_beta_prime_f(dy-my);
        for(int mx=-1;mx<=2;++mx){
          int sx=min(max(ix+mx,0),Wp-1);
          float wx=cubic_beta_f(dx-mx), pwx=cubic_beta_prime_f(dx-mx);
          float w = wz*wy*wx;
          int img_idx = ((m*Dp + sz)*Hp + sy)*Wp + sx;
          float val = img[img_idx];

          atomicAdd(&grad_img[img_idx],    g * w);
          atomicAdd(&grad_coords[out_idx*3+0], g * val * pwz * wy * wx);
          atomicAdd(&grad_coords[out_idx*3+1], g * val * wz * pwy * wx);
          atomicAdd(&grad_coords[out_idx*3+2], g * val * wz * wy * pwx);
        }
      }
    }
}


__device__ __forceinline__ double cubic_beta_d(double x) {
    x = fabs(x);
    if (x < 1.0)      return (2.0/3.0) - 0.5 * x * x * (2.0 - x);
    else if (x < 2.0) return ((2.0 - x)*(2.0 - x)*(2.0 - x)) / 6.0;
    return 0.0;
}

__device__ __forceinline__ double cubic_beta_prime_d(double x) {
    double ax = fabs(x);
    if (ax < 1.0) {
        return -2.0 * x + 1.5 * x * ax;
    } else if (ax < 2.0) {
        double t = 2.0 - ax;
        double s = (x < 0 ? -1.0 : 1.0);
        return -0.5 * t * t * s;
    } else {
        return 0.0;
    }
}

__device__ inline void atomicAddDouble(double* addr, double val) {
#if __CUDA_ARCH__ >= 600
    atomicAdd(addr, val);
#else
    unsigned long long int* address_as_ull =
        reinterpret_cast<unsigned long long int*>(addr);
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        double new_val = val + __longlong_as_double(assumed);
        old = atomicCAS(address_as_ull,
                        assumed,
                        __double_as_longlong(new_val));
    } while (assumed != old);
#endif
}

__global__ void bspline_interp_kernel_f64(
    const double* __restrict__ img,
    const double* __restrict__ coords,
    double*       __restrict__ output,
    int M, int D, int H, int W,
    int Dp, int Hp, int Wp
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int bz = blockIdx.z;
    int z = bz % D;
    int m = bz / D;
    if (x>=W||y>=H||z>=D||m>=M) return;

    int out_idx = ((m*D + z)*H + y)*W + x;
    const double* coord = coords + out_idx*3;
    double zf=coord[0], yf=coord[1], xf=coord[2];
    int iz = static_cast<int>(floor(zf)), 
        iy = static_cast<int>(floor(yf)),
        ix = static_cast<int>(floor(xf));
    double dz=zf-iz, dy=yf-iy, dx=xf-ix;

    double result=0.0;
    for(int mz=-1;mz<=2;++mz){
      int sz = min(max(iz+mz,0),Dp-1);
      double wz = cubic_beta_d(dz-mz);
      for(int my=-1;my<=2;++my){
        int sy = min(max(iy+my,0),Hp-1);
        double wy = cubic_beta_d(dy-my);
        for(int mx=-1;mx<=2;++mx){
          int sx = min(max(ix+mx,0),Wp-1);
          double wx = cubic_beta_d(dx-mx);
          double w = wz*wy*wx;
          int img_idx = ((m*Dp + sz)*Hp + sy)*Wp + sx;
          result += img[img_idx] * w;
        }
      }
    }
    output[out_idx] = result;
}

__global__ void bspline_interp_grad_both_kernel_f64(
    const double* __restrict__ img,
    const double* __restrict__ grad_out,
    const double* __restrict__ coords,
    double*       __restrict__ grad_img,
    double*       __restrict__ grad_coords,
    int M, int D, int H, int W,
    int Dp, int Hp, int Wp
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int bz = blockIdx.z;
    int z = bz % D;
    int m = bz / D;
    if (x>=W||y>=H||z>=D||m>=M) return;

    int out_idx = ((m*D + z)*H + y)*W + x;

    const double* coord = coords + out_idx*3;
    double zf=coord[0], yf=coord[1], xf=coord[2];
    int iz = static_cast<int>(floor(zf)),
        iy = static_cast<int>(floor(yf)),
        ix = static_cast<int>(floor(xf));
    double dz=zf-iz, dy=yf-iy, dx=xf-ix;
    double g = grad_out[out_idx];

    for(int mz=-1;mz<=2;++mz){
      int sz = min(max(iz+mz,0),Dp-1);
      double wz=cubic_beta_d(dz-mz), pwz=cubic_beta_prime_d(dz-mz);
      for(int my=-1;my<=2;++my){
        int sy=min(max(iy+my,0),Hp-1);
        double wy=cubic_beta_d(dy-my), pwy=cubic_beta_prime_d(dy-my);
        for(int mx=-1;mx<=2;++mx){
          int sx=min(max(ix+mx,0),Wp-1);
          double wx=cubic_beta_d(dx-mx), pwx=cubic_beta_prime_d(dx-mx);
          double w = wz*wy*wx;
          int img_idx = ((m*Dp + sz)*Hp + sy)*Wp + sx;
          double val = img[img_idx];

          atomicAddDouble(&grad_img[img_idx],    g * w);
          atomicAddDouble(&grad_coords[out_idx*3+0], g * val * pwz * wy * wx);
          atomicAddDouble(&grad_coords[out_idx*3+1], g * val * wz * pwy * wx);
          atomicAddDouble(&grad_coords[out_idx*3+2], g * val * wz * wy * pwx);
        }
      }
    }
}


extern "C" {

void bspline_interp_forward_cuda(
    const at::Tensor& img,
    const at::Tensor& coords,
          at::Tensor& output
) {
    TORCH_CHECK(img.device().is_cuda(),   "img must be CUDA");
    TORCH_CHECK(coords.device().is_cuda(),"coords must be CUDA");
    TORCH_CHECK(output.device().is_cuda(),"output must be CUDA");
    TORCH_CHECK(img.is_floating_point(),  "img must be floating type");
    TORCH_CHECK(coords.is_floating_point(),"coords must be floating type");
    TORCH_CHECK(img.scalar_type() == coords.scalar_type(),
        "img and coords must have same dtype");
    TORCH_CHECK(img.scalar_type() == output.scalar_type(),
        "img and output must have same dtype");

    int M  = img.size(0);
    int Dp = img.size(1);
    int Hp = img.size(2);
    int Wp = img.size(3);

    int D = coords.size(1);
    int H = coords.size(2);
    int W = coords.size(3);

    dim3 th(8,8);
    dim3 bl((W+th.x-1)/th.x,(H+th.y-1)/th.y,M*D);

    if (img.scalar_type() == at::kFloat) {
        bspline_interp_kernel_f32<<<bl,th>>>(
          img.data_ptr<float>(),
          coords.data_ptr<float>(),
          output.data_ptr<float>(),
          M,D,H,W,Dp,Hp,Wp
        );
    } else if (img.scalar_type() == at::kDouble) {
        bspline_interp_kernel_f64<<<bl,th>>>(
          img.data_ptr<double>(),
          coords.data_ptr<double>(),
          output.data_ptr<double>(),
          M,D,H,W,Dp,Hp,Wp
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype in bspline_interp_forward_cuda");
    }

    cudaDeviceSynchronize();
}

void bspline_interp_backward_cuda(
    const at::Tensor& img,
    const at::Tensor& grad_out,
    const at::Tensor& coords,
          at::Tensor& grad_img,
          at::Tensor& grad_coords
) {
    TORCH_CHECK(img.device().is_cuda(),        "img must be CUDA");
    TORCH_CHECK(grad_out.device().is_cuda(),   "grad_out must be CUDA");
    TORCH_CHECK(coords.device().is_cuda(),     "coords must be CUDA");
    TORCH_CHECK(grad_img.device().is_cuda(),   "grad_img must be CUDA");
    TORCH_CHECK(grad_coords.device().is_cuda(),"grad_coords must be CUDA");

    TORCH_CHECK(img.is_floating_point(), "img must be floating type");
    TORCH_CHECK(img.scalar_type() == grad_out.scalar_type(),
        "img and grad_out must have same dtype");
    TORCH_CHECK(img.scalar_type() == coords.scalar_type(),
        "img and coords must have same dtype");
    TORCH_CHECK(img.scalar_type() == grad_img.scalar_type(),
        "img and grad_img must have same dtype");
    TORCH_CHECK(img.scalar_type() == grad_coords.scalar_type(),
        "img and grad_coords must have same dtype");

    int M  = grad_img.size(0);
    int Dp = grad_img.size(1);
    int Hp = grad_img.size(2);
    int Wp = grad_img.size(3);

    int D = grad_out.size(1);
    int H = grad_out.size(2);
    int W = grad_out.size(3);

    dim3 th(8,8);
    dim3 bl((W+th.x-1)/th.x,(H+th.y-1)/th.y,M*D);

    if (img.scalar_type() == at::kFloat) {
        bspline_interp_grad_both_kernel_f32<<<bl,th>>>(
          img.data_ptr<float>(),
          grad_out.data_ptr<float>(),
          coords.data_ptr<float>(),
          grad_img.data_ptr<float>(),
          grad_coords.data_ptr<float>(),
          M,D,H,W,Dp,Hp,Wp
        );
    } else if (img.scalar_type() == at::kDouble) {
        bspline_interp_grad_both_kernel_f64<<<bl,th>>>(
          img.data_ptr<double>(),
          grad_out.data_ptr<double>(),
          coords.data_ptr<double>(),
          grad_img.data_ptr<double>(),
          grad_coords.data_ptr<double>(),
          M,D,H,W,Dp,Hp,Wp
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype in bspline_interp_backward_cuda");
    }

    cudaDeviceSynchronize();
}

} 
