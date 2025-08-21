import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tensor_matmul_kernel(float* A, float* B, float* C, int N, int M, int K, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * M * L) return;

    int l = idx % L;
    int m_n = idx / L;
    int m = m_n % M;
    int n = m_n / M;

    float sum = 0.0f;
    for (int k = 0; k < K; k += 4) {
        int a_offset = n * M * K + m * K + k;
        int b_offset = l * K + k;
        float a0 = A[a_offset];
        float a1 = A[a_offset + 1];
        float a2 = A[a_offset + 2];
        float a3 = A[a_offset + 3];
        float b0 = B[b_offset];
        float b1 = B[b_offset + 1];
        float b2 = B[b_offset + 2];
        float b3 = B[b_offset + 3];
        sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
    }
    C[n * M * L + m * L + l] = sum;
}

torch::Tensor tensor_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(0); // B is transposed to (L, K)

    auto C = torch::empty({N, M, L}, A.options());

    int threads_per_block = 256;
    int num_blocks = (N * M * L + threads_per_block - 1) / threads_per_block;

    tensor_matmul_kernel<<<num_blocks, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N, M, K, L
    );

    return C;
}
"""

cpp_src = "torch::Tensor tensor_matmul_cuda(torch::Tensor A, torch::Tensor B);"

tensor_matmul = load_inline(
    name="tensor_matmul",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["tensor_matmul_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.tensor_matmul = tensor_matmul

    def forward(self, A, B):
        B_t = B.t().contiguous()
        return self.tensor_matmul.tensor_matmul_cuda(A, B_t)
