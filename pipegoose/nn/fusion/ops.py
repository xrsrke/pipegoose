from torch.utils.cpp_extension import load
import os
import torch
class FusedDummy:

    def __init__(self):
        pass

    def load(self):
        dir_path = os.path.dirname(os.path.realpath(__file__)) 
        cuda_file = os.path.join(dir_path, "dummy.cu")
        cpp_file = os.path.join(dir_path, "dummy.cpp")

        self.kernel_cuda = load(
            name="torch_dummy",
            sources=[cuda_file, cpp_file],
            verbose=True,
            extra_cuda_cflags=["-O3"]
        )

        return self

    def forward(self, x):
        n = x.shape[0]
        # Move tensor to GPU
        y_cuda = torch.zeros(n, dtype=torch.float32).to("cuda")
        x_cuda = x.to("cuda")
        # Compute kernel
        self.kernel_cuda.forward(n, x_cuda, y_cuda)
        # Send result back to CPU
        y_cpu = y_cuda.to("cpu")
        return y_cpu