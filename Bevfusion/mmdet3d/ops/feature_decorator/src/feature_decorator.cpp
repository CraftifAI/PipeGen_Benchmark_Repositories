#include <torch/extension.h>

// 1. Correct CUDA Declaration (Matching your .cu file exactly)
// Tera .cu file sirf 'out' pointer mang raha hai, toh hum wahi denge.
void feature_decorator(float* out);

// 2. PyTorch Wrapper
// Python side se saare arguments aayenge, hum unhe receive karenge
// taaki Python code crash na ho, par internal CUDA call mein sirf 'out' bhejenge.
at::Tensor feature_decorator_forward(
  const at::Tensor _x, 
  const at::Tensor _y, 
  const at::Tensor _z, 
  const double vx, const double vy, const double x_offset, const double y_offset, 
  int64_t normalize_coords, int64_t use_cluster, int64_t use_center // int64_t for PyTorch compatibility
) {
  int n = _x.size(0);
  int c = _x.size(1);
  int a = _x.size(2);
  auto options = torch::TensorOptions().dtype(_x.dtype()).device(_x.device());
  
  int decorate_dims = 0;
  if (use_cluster > 0) { decorate_dims += 3; }
  if (use_center > 0) { decorate_dims += 2; }

  // Output tensor banana
  at::Tensor _out = torch::zeros({n, c, a+decorate_dims}, options);
  float* out = _out.data_ptr<float>();
  
  // 3. Call the Dummy CUDA function
  // Sirf 'out' pass kar rahe hain kyunki teri .cu file yahi maang rahi hai.
  feature_decorator(out);
  
  return _out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("feature_decorator_forward", &feature_decorator_forward, "feature_decorator_forward");
}

// Register operator
static auto registry =
    torch::RegisterOperators("feature_decorator_ext::feature_decorator_forward", &feature_decorator_forward);