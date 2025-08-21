import torch
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void matmul_kernel(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    int N, int M, int K, int L,
    int stride_A_M, int stride_A_K,
    int stride_B_K, int stride_B_L,
    int stride_C_M, int stride_C_L
) {
    extern __shared__ float s[];
    float* s_A = s;
    float* s_B = s_A + 2 * BLOCK_M * BLOCK_K;

    int batch = blockIdx.x;
    int tile_m = blockIdx.y;
    int tile_n = blockIdx.z;

    int m_start = tile_m * BLOCK_M;
    int l_start = tile_n * BLOCK_N;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = tx * blockDim.y + ty;

    float c[2][2] = {0.0f};

    int stage = 0;
    for (int k_outer = 0; k_outer < K; k_outer += BLOCK_K) {
        // Load A tile
        for (int row = tx * 2; row < (tx + 1)*2 && row < BLOCK_M; row += 1) {
            int a_offset = batch * M * K + (m_start + row) * stride_A_M + k_outer * stride_A_K;
            float4* a_ptr = (float4*)(A + a_offset);
            s_A[stage * BLOCK_M * BLOCK_K + row * BLOCK_K + 0] = a_ptr[0].x;
            s_A[stage * BLOCK_M * BLOCK_K + row * BLOCK_K + 1] = a_ptr[0].y;
            s_A[stage * BLOCK_M * BLOCK_K + row * BLOCK_K + 2] = a_ptr[0].z;
            s_A[stage * BLOCK_M * BLOCK_K + row * BLOCK_K + 3] = a_ptr[0].w;
        }

        // Load B tile
        for (int col = ty * 2; col < (ty + 1)*2 && col < BLOCK_N; col += 1) {
            for (int row_b = 0; row_b < BLOCK_K; ++row_b) {
                int b_offset = (k_outer + row_b) * stride_B_K + (l_start + col);
                s_B[stage * BLOCK_K * BLOCK_N + row_b * BLOCK_N + col] = B[b_offset];
            }
        }

        __syncthreads();

        // Compute contributions
        for (int kk = 0; kk < BLOCK_K; ++kk) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    int c_row = tx * 2 + i;
                    int c_col = ty * 2 + j;
                    if (c_row < BLOCK_M && c_col < BLOCK_N) {
                        float a_val = s_A[stage * BLOCK_M * BLOCK_K + c_row * BLOCK_K + kk];
                        float b_val = s_B[stage * BLOCK_K * BLOCK_N + kk * BLOCK_N + c_col];
                        c[i][j] += a_val * b_val;
                    }
                }
            }
        }

        stage ^= 1;
        __syncthreads();
    }

    // Write back to global memory
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            int c_row = tx * 2 + i;
            int c_col = ty * 2 + j;
            if (c_row < BLOCK_M && c_col < BLOCK_N) {
                int c_offset = batch * M * L + (m_start + c_row) * stride_C_M + (l_start + c_col);
                atomicAdd(C + c_offset, c[i][j]);
            }
        }
    }
}

extern "C" {
    __host__ void matmul_cuda(
        torch::Tensor C,
        torch::Tensor A,
        torch::Tensor B
    ) {
        const int BLOCK_M = 32;
        const int BLOCK_N = 16;
        const int BLOCK_K = 4;
        const dim3 threads(16, 8);
        int grid_m = (A.size(1) + BLOCK_M - 1) / BLOCK_M;
        int grid_l = (B.size(1) + BLOCK_N - 1) / BLOCK_N;
        dim3 grid(A.size(0), grid_m, grid_l);

        matmul_kernel<BLOCK_M, BLOCK_N, BLOCK_K><<<grid, threads, 2 * (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * sizeof(float)>>>(
            C.data_ptr<float>(),
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            A.size(0), A.size(1), A.size(2), B.size(1),
            A.stride(1), A.stride(2),
            B.stride(0), B.stride(1),
            C.stride(1), C.stride(2)
        );
    }
}
"""

cpp_src = """
void matmul_cuda(torch::Tensor C, torch::Tensor A, torch::Tensor B);
"""

module = load_inline(
    name='matmul_opt',
    cpp_sources=[cpp_src],
    cuda_sources=[source],
    extra_cuda_cflags=['-arch=sm_75', '-lineinfo'],
    with_cuda=True
)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        N, M, K = A.shape
        _, L = B.shape
        C = torch.empty(N, M, L, dtype=A.dtype, device=A.device)
        module.matmul_cuda(C, A, B)
        return C
