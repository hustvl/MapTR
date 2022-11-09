#include "geometric_kernel_attn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("geometric_kernel_attn_cuda_forward", &geometric_kernel_attn_cuda_forward, "geometric_kernel_attn_cuda_forward");
  m.def("geometric_kernel_attn_cuda_backward", &geometric_kernel_attn_cuda_backward, "geometric_kernel_attn_cuda_backward");
}
