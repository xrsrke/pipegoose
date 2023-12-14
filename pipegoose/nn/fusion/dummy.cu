#include <stdio.h>
#include <torch/extension.h>

__global__ void kernel_forward(float *X, float *Y)
{
    // Copy element of X into Y;
    Y[threadIdx.x] = X[threadIdx.x];
}


void cuda_forward(int64_t N, float *X, float *Y)
{
    kernel_forward<<<1, N>>>(X, Y);
    cudaDeviceSynchronize();
}

