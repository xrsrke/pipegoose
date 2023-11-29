#include <torch/extension.h>
#include "ATen/ATen.h"

void cuda_forward();
void cuda_backward();

void forward(torch::Tensor &X, int N) {
    cuda_forward(X.data_ptr<float>(), N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Docstring
    m.def("forward", &forward, "Function that does forward pass for FusedDummy");
}

TORCH_LIBRARY(wkv5, m) {
    m.def("forward", forward);
}
