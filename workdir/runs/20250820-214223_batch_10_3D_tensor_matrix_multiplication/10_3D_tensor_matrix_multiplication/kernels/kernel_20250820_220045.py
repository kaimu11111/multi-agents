import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul3d_kernel(const float* A, const float* B, float* C, int N, int M, int K, int L) {
    int blockId = blockIdx.x;
    int n = blockId / M;
    int m = blockId % M;
    int l = threadIdx.x;
    if (l >= L) return;

    extern __shared__ float shared_data[];
    float* shared_A = shared_data;
    float* shared_B = shared_data + K;

    // Load A_row into shared_A.
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        shared_A[k] = A[n * M * K + m * K + k];
    }

    // Load B_col into shared_B.
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        shared_B[k] = B[k * L + l];
    }

    __syncthreads();

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += shared_A[k] * shared_B[k];
    }

    C[n * M * L + m * L + l] = sum;
}

torch::Tensor matmul3d_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);

    auto C = torch::empty({N, M, L}, A.options());

    dim3 blocks(N * M);
    dim3 threads(L);
    size_t shared_mem = 2 * K * sizeof(float);

    matmul3d_kernel<<<blocks, threads, shared_mem, at::cuda::getCurrentCUDAStream()>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M, K, L
    );

    return C;
}
"""

cpp_src = (
    "torch::Tensor matmul3d_cuda(torch::Tensor A, torch::Tensor B);"
)

matmul3d = load_inline(
    name="matmul3d",
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
        self.matmul3d = matmul3d

    def forward(self, A, B):
        return self.matmul3d.matmul3d_cuda(A, B)
