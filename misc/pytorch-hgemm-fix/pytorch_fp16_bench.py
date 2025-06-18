import torch
import time
import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, K, M, bias=True):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(K, M, bias=bias, dtype=torch.float16)

    def forward(self, x):
        return self.fc(x)


class MatmulModel(nn.Module):
    def __init__(self, bias=True):
        super(MatmulModel, self).__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)


class BatchedMatmulModel(nn.Module):
    def __init__(self, bias=True):
        super(BatchedMatmulModel, self).__init__()

    def forward(self, x, y):
        return torch.bmm(x, y)


def fp16_linear_nobias_bench(M, N, K, ITERS):
    print(f"  Linear w/o bias {M} x {N} x {K} => ", end='')
    x = torch.randn(M, K, device='cuda',
                    dtype=torch.float16, requires_grad=True)
    model = LinearModel(K, N, bias=False).to('cuda')
    start_time = time.time()
    for i in range(ITERS):
        model(x).sum().backward()
    torch.cuda.synchronize()
    end_time = time.time()
    FLOPS = 6.0 * (M * N * K) * ITERS / (end_time - start_time)
    print(f"{FLOPS/1e12:.3f} TFLOPS")


def fp16_linear_bias_bench(M, N, K, ITERS):
    print(f"  Linear w/ bias {M} x {N} x {K} => ", end='')
    x = torch.randn(M, K, device='cuda',
                    dtype=torch.float16, requires_grad=True)
    model = LinearModel(K, N, bias=True).to('cuda')
    start_time = time.time()
    for i in range(ITERS):
        model(x).sum().backward()
    torch.cuda.synchronize()
    end_time = time.time()
    FLOPS = 6.0 * (M * N * K) * ITERS / (end_time - start_time)
    print(f"{FLOPS/1e12:.3f} TFLOPS")


def fp16_gemm_bench(M, N, K, ITERS):
    print(f"  GEMM {M} x {N} x {K} => ", end='')
    x = torch.randn(M, K, device='cuda',
                    dtype=torch.float16, requires_grad=True)
    y = torch.randn(K, N, device='cuda',
                    dtype=torch.float16, requires_grad=True)
    model = MatmulModel().to('cuda')
    start_time = time.time()
    for i in range(ITERS):
        model(x, y).sum().backward()
    torch.cuda.synchronize()
    end_time = time.time()
    FLOPS = 6.0 * (M * N * K) * ITERS / (end_time - start_time)
    print(f"{FLOPS/1e12:.3f} TFLOPS")


def fp16_batched_gemm_bench(B, M, N, K, ITERS):
    print(f"  BMM {B} x {M} x {N} x {K} => ", end='')
    x = torch.randn(B, M, K, device='cuda',
                    dtype=torch.float16, requires_grad=True)
    y = torch.randn(B, K, N, device='cuda',
                    dtype=torch.float16, requires_grad=True)
    model = BatchedMatmulModel().to('cuda')
    start_time = time.time()
    for i in range(ITERS):
        model(x, y).sum().backward()
    torch.cuda.synchronize()
    end_time = time.time()
    FLOPS = 6.0 * B * M * N * K * ITERS / (end_time - start_time)
    print(f"{FLOPS/1e12:.3f} TFLOPS")


torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
print(
    f"FLAG torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction={torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction}")

fp16_linear_nobias_bench(8192, 8192, 8192, 100)
fp16_linear_bias_bench(8192, 8192, 8192, 100)
fp16_gemm_bench(8192, 8192, 8192, 100)
fp16_batched_gemm_bench(16, 4096, 4096, 4096, 100)

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
print(
    f"FLAG torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction={torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction}")

fp16_linear_nobias_bench(8192, 8192, 8192, 100)
fp16_linear_bias_bench(8192, 8192, 8192, 100)
fp16_gemm_bench(8192, 8192, 8192, 100)
fp16_batched_gemm_bench(16, 4096, 4096, 4096, 100)
