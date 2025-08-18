import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#define BLOCK_M 32
#define BLOCK_N 32
#define BLOCK_K 16

template <typename T>
__device__ __forceinline__ void load_vectorized(const T* ptr, T* dest, int offset, int vec_size) {
    static_assert(vec_size == 4, "Vector size must be 4 for float4");
    using Vec = float4;
    const Vec* vec_ptr = reinterpret_cast<const Vec*>(ptr + offset);
    Vec val = *vec_ptr;
    dest[0] = val.x;
    dest[1] = val.y;
    dest[2] = val.z;
    dest[3] = val.w;
}

template <typename T>
__device__ __forceinline__ void store_vectorized(T* ptr, const T* src, int offset, int vec_size) {
    static_assert(vec_size == 4, "Vector size must be 4 for float4");
    using Vec = float4;
    Vec val;
    val.x = src[0];
    val.y = src[1];
    val.z = src[2];
    val.w = src[3];
    Vec* vec_ptr = reinterpret_cast<Vec*>(ptr + offset);
    *vec_ptr = val;
}

__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb, int ldc) {

    extern __shared__ float shared[];
    float* sA = shared;
    float* sB = shared + (BLOCK_M * BLOCK_K * 2); // Double buffered

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = bx * BLOCK_M + ty;
    int col = by * BLOCK_N + tx;

    float Csub = 0.0f;

    for (int k = 0; k < K; k += BLOCK_K) {
        // Load A tile into shared memory (double buffering)
        int a_row = bx * BLOCK_M + ty;
        int a_col = k + tx;
        if (a_row < M && a_col < K) {
            sA[ty * BLOCK_K + tx] = A[a_row * lda + a_col];
        } else {
            sA[ty * BLOCK_K + tx] = 0.0f;
        }

        // Load B tile into shared memory (double buffering)
        int b_row = k + ty;
        int b_col = by * BLOCK_N + tx;
        if (b_row < K && b_col < N) {
            sB[ty * BLOCK_N + tx] = B[b_row * ldb + b_col];
        } else {
            sB[ty * BLOCK_N + tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial products using shared memory tiles
        for (int i = 0; i < BLOCK_K; ++i) {
            Csub += sA[ty * BLOCK_K + i] * sB[i * BLOCK_N + tx];
        }

        __syncthreads();
    }

    // Write result with bounds check
    if (row < M && col < N) {
        atomicAdd(&C[row * ldc + col], Csub);
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    assert(A.size(1) == B.size(0));

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(BLOCK_M, BLOCK_N);
    dim3 blocks((M + BLOCK_M - 1)/BLOCK_M, (N + BLOCK_N - 1)/BLOCK_N);

    int shared_size = 2 * (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * sizeof(float);
    matmul_kernel<<<blocks, threads, shared_size, at::cuda::getCurrentCUDAStream()>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, N, K, A.stride(0), B.stride(0), C.stride(0));

    return C;
}
"""

cpp_src = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_cuda = load_inline(
    name="matmul_cuda",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["matmul_cuda"],
    verbose=False,
    extra_cflags=["-arch=sm_75"],
    extra_cuda_cflags=["-arch=sm_75"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = matmul_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul(A, B)
