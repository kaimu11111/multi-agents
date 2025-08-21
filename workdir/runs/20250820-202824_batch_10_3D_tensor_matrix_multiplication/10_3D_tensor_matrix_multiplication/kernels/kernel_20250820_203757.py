import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void batched_matmul_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int N, int M, int K, int L,
    int lda, int ldb, int ldc) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N * M * L) return;

    int i = idx / (M * L);
    int rem = idx % (M * L);
    int j = rem / L;
    int l = rem % L;

    scalar_t sum = 0.0;
    for (int k = 0; k < K; ++k) {
        sum += A[i * lda + j * K + k] * B[k * ldb + l];
    }
    C[i * ldc + j * L + l] = sum;
}

torch::Tensor batched_matmul_cuda(
    torch::Tensor A,
    torch::Tensor B) {

    int N = A.size(0);
    int M = A.size(1);
    int K_A = A.size(2);
    int K_B = B.size(0);
    int L = B.size(1);

    assert(K_A == K_B);

    auto C = torch::empty({N, M, L}, A.options());

    int threads_per_block = 256;
    int num_elements = N * M * L;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "batched_matmul_cuda", ([&] {
        batched_matmul_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            N, M, K_A, L,
            A.stride(0),
            B.stride(0),
            C.stride(0)
        );
    }));

    return C;
}
"""

cpp_src = (
    "torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

batched_matmul = load_inline(
    name="batched_matmul",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["batched_matmul_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.batched_matmul = batched_matmul

    def forward(self, A, B):
        return self.batched_matmul.batched_matmul_cuda(A, B)
