import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <int K>
__global__ void matmul3d_kernel(const float* A, const float* B, float* C, int N, int M, int L) {
    int n = blockIdx.x / M;
    int m = blockIdx.x % M;
    int l = threadIdx.x;

    if (l >= L) return;

    __shared__ float shared_A[K];

    int tid = threadIdx.x;
    for (int k = tid; k < K; k += blockDim.x) {
        shared_A[k] = A[ n * M * K + m * K + k ];
    }
    __syncthreads();

    float sum = 0.0f;
    for (int k = 0; k < K; k += 4) {
        if (k + 3 < K) {
            float a0 = shared_A[k];
            float a1 = shared_A[k+1];
            float a2 = shared_A[k+2];
            float a3 = shared_A[k+3];
            float b0 = B[ l * K + k ];
            float b1 = B[ l * K + (k+1) ];
            float b2 = B[ l * K + (k+2) ];
            float b3 = B[ l * K + (k+3) ];
            sum += a0*b0 + a1*b1 + a2*b2 + a3*b3;
        }
    }

    C[ n * M * L + m * L + l ] = sum;
}

torch::Tensor matmul3d_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    int M = A.size(1);
    int L = B.size(1);

    auto B_t = B.t().contiguous();

    auto output = torch::empty({N, M, L}, A.options());

    int threads_per_block = L;
    int blocks_per_grid = N * M;

    matmul3d_kernel<2048><<<blocks_per_grid, threads_per_block>>>(A.data_ptr<float>(), B_t.data_ptr<float>(), output.data_ptr<float>(), N, M, L);

    return output;
}
"""

cpp_src = (
    "torch::Tensor matmul3d_cuda(torch::Tensor A, torch::Tensor B);"
)

matmul3d_cuda = load_inline(
    name="matmul3d_cuda",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["matmul3d_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul3d_cuda = matmul3d_cuda

    def forward(self, A, B):
        return self.matmul3d_cuda(A, B)
