#include <torch/extension.h>
#include "ATen/ATen.h"

void cuda_forward(int64_t N, float *X, float *Y);

void forward(int64_t N, torch::Tensor &X, torch::Tensor &Y) {
    cuda_forward(N, X.data_ptr<float>(), Y.data_ptr<float>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Function that does forward pass for FusedDummy");
}

TORCH_LIBRARY(torch_dummy, m) {
    m.def("forward", forward);
}
