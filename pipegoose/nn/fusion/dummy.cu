#include <stdio.h>
#include <torch/extension.h>

__global__ void kernel_foward(const float X, const int N)
{
    for (int i = 0; i < N; i++)
        printf("%f\n", X[i]);
}

void cuda_foward(float *X, int N)
{
    kernel_foward<<<1,1>>>(X, N);
}

