from torch.utils.cpp_extension import load

class FusedDummy():

    def __init__(self):

        self.kernel_cuda = load(
            name="dummy",
            sources=["/home/bouteille/pipegoose/pipegoose/nn/fusion/dummy.cu", "/home/bouteille/pipegoose/pipegoose/nn/fusion/dummy.cpp"],
            verbose=True,
            extra_cuda_cflags=["-O3"]
        )

    def forward(self, x):
        n = x.shape[0]
        return self.kernel_cuda.forward(x, n)
