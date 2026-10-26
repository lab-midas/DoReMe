#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
#include <ATen/ATen.h>
#include <cmath>
#include <vector>

using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::variable_list;

extern "C" {
  void bspline_prefilter_forward_c(
      void*   img,
      int     M, int D, int H, int W,
      int     scalar_type,
      float   S, float   Dcoef);

  void bspline_prefilter_backward_c(
      const void* grad_out,
      void*       grad_img,
      int         M, int D, int H, int W,
      int         scalar_type,
      float       S, float   Dcoef);

  void bspline_prefilter_forward_optimized(
      void*   img,
      int     M, int D, int H, int W,
      int     scalar_type);

  void bspline_prefilter_backward_optimized(
      const void* grad_out,
      void*       grad_img,
      int         M, int D, int H, int W,
      int         scalar_type);
}

extern "C" {
  void bspline_interp_forward_cuda(
      const at::Tensor &img,
      const at::Tensor &coords,
            at::Tensor &output);

  void bspline_interp_backward_cuda(
      const at::Tensor &img,
      const at::Tensor &grad_out,
      const at::Tensor &coords,
            at::Tensor &grad_img,
            at::Tensor &grad_coords);
}


inline void compute_s_and_d(float &S, float &Dcoef) {
  auto beta = [](float x) {
    x = std::fabs(x);
    if (x < 1.0f)      return (2.0f/3.0f) - 0.5f*x*x*(2.0f - x);
    else if (x < 2.0f) return ((2.0f - x)*(2.0f - x)*(2.0f - x))/6.0f;
    else               return 0.0f;
  };
  S     = beta(0.0f);
  Dcoef = beta(1.0f);
}


class BsplineInterpFunction
  : public Function<BsplineInterpFunction>
{
 public:
  static at::Tensor forward(
      AutogradContext *ctx,
      at::Tensor img,
      at::Tensor coords)
  {
    TORCH_CHECK(img.device() == coords.device(),
                "bspline_interp_autograd: img and coords must be on same device");
    TORCH_CHECK(img.is_floating_point(),
                "bspline_interp_autograd: img must be float32 or float64");
    TORCH_CHECK(coords.is_floating_point(),
                "bspline_interp_autograd: coords must be float32 or float64");
    TORCH_CHECK(img.scalar_type() == coords.scalar_type(),
                "bspline_interp_autograd: img and coords must have same dtype");

    auto out = at::zeros(
      { img.size(0), coords.size(1), coords.size(2), coords.size(3) },
      img.options());

    bspline_interp_forward_cuda(img, coords, out);

    ctx->save_for_backward({coords, img});
    ctx->saved_data["img_shape"] = img.sizes().vec();
    return out;
  }

  static variable_list backward(
      AutogradContext *ctx,
      variable_list grad_outs)
  {
    auto saved      = ctx->get_saved_variables();
    auto coords     = saved[0];
    auto img        = saved[1];
    auto ishape     = ctx->saved_data["img_shape"].toIntVector();

    at::Tensor grad_out    = grad_outs[0];
    at::Tensor grad_img    = at::zeros(ishape, grad_out.options());
    at::Tensor grad_coords = at::zeros_like(coords);

    bspline_interp_backward_cuda(
      img, grad_out, coords,
      grad_img, grad_coords
    );

    return { grad_img, grad_coords };
  }
};


class BsplinePrefilterFunction
  : public Function<BsplinePrefilterFunction>
{
 public:
  static at::Tensor forward(
      AutogradContext       *ctx,
      at::Tensor             img,
      c10::optional<float>   ,
      c10::optional<float>   )
  {
    TORCH_CHECK(img.is_floating_point(),
      "bspline_prefilter_autograd: img must be float32 or float64");
    
    TORCH_CHECK(img.size(1) <= 512 && img.size(2) <= 512 && img.size(3) <= 512,
      "bspline_prefilter_autograd: dimensions must not exceed 512 (MAX_DIM)");

    auto img_clone = img.clone();
    auto sizes     = img_clone.sizes();
    int M    = sizes[0], D = sizes[1], H = sizes[2], W = sizes[3];

    float S, Dcoef;
    compute_s_and_d(S, Dcoef);
    int st = (img_clone.scalar_type() == at::kDouble ? 1 : 0);

    bspline_prefilter_forward_optimized(
      img_clone.data_ptr(), M, D, H, W, st
    );

    ctx->saved_data["sizes"]  = std::vector<int64_t>(sizes.vec());
    ctx->saved_data["stype"]  = st;
    ctx->saved_data["S"]      = S;
    ctx->saved_data["Dcoef"]  = Dcoef;
    return img_clone;
  }

  static variable_list backward(
      AutogradContext *ctx,
      variable_list    grad_outs)
  {
    auto S      = ctx->saved_data["S"].toDouble();
    auto Dcoef  = ctx->saved_data["Dcoef"].toDouble();
    auto sizes  = ctx->saved_data["sizes"].toIntVector();
    auto st     = ctx->saved_data["stype"].toInt();

    at::Tensor grad_out = grad_outs[0];
    at::Tensor grad_img = at::zeros(sizes, grad_out.options());

    bspline_prefilter_backward_optimized(
      grad_out.data_ptr(),
      grad_img.data_ptr(),
      sizes[0], sizes[1], sizes[2], sizes[3],
      st
    );

    return { grad_img, at::Tensor(), at::Tensor() };
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bspline_interp_autograd",
        [](at::Tensor img, at::Tensor coords) {
          return BsplineInterpFunction::apply(img, coords);
        },
        "Autograd-enabled B-spline interpolation (float32 & float64)");

  m.def("bspline_prefilter_autograd",
        [](at::Tensor img,
           c10::optional<float> s,
           c10::optional<float> d) {
          return BsplinePrefilterFunction::apply(img, s, d);
        },
        "Autograd-enabled B-spline prefilter (float32 & float64)",
        py::arg("img"),
        py::arg("s") = py::none(),
        py::arg("d") = py::none());
}
