import time
import numpy as np
import sys

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from fduai.autograd import DataNode, Tensor, Device
    FDUAI_AVAILABLE = True
except ImportError:
    FDUAI_AVAILABLE = False

# 检查 CUDA 可用性
TORCH_CUDA = TORCH_AVAILABLE and torch.cuda.is_available()

SHAPE = (100, 100)
REPEAT = 20

# 线性回归/MLP参数
IN_FEATURES = 1024
OUT_FEATURES = 1024
BATCH_SIZE = 1024


def benchmark_fduai_matmul(shape, repeat, device=Device.CPU):
    # 前向+反向：z = x @ w; loss = sum(z); loss.backward()
    x = DataNode(Tensor.from_numpy(np.random.rand(
        *shape).astype(np.float32)), requires_grad=True)
    w = DataNode(Tensor.from_numpy(np.random.rand(
        *shape).astype(np.float32)), requires_grad=True)
    # 预热
    for _ in range(2):
        z = DataNode.matmul(x, w)
        loss = z * z
        item = Tensor.sum_all(loss.tensor)
        loss.backward()
        DataNode.zero_grad()
    start = time.time()
    for _ in range(repeat):
        z = DataNode.matmul(x, w)
        loss = z * z
        item = Tensor.sum_all(loss.tensor)
        loss.backward()
        DataNode.zero_grad()
    end = time.time()
    return (end - start) / repeat


def benchmark_fduai_linear(batch, in_features, out_features, repeat, device=Device.CPU):
    # y = x @ w + b; loss = sum((y - t) ** 2); loss.backward()
    x = DataNode(Tensor.from_numpy(np.random.rand(
        batch, in_features).astype(np.float32)), requires_grad=False)
    t = DataNode(Tensor.from_numpy(np.random.rand(
        batch, out_features).astype(np.float32)), requires_grad=False)
    w = DataNode(Tensor.from_numpy(np.random.rand(
        in_features, out_features).astype(np.float32)), requires_grad=True)
    b = DataNode(Tensor.from_numpy(np.random.rand(
        1, out_features).astype(np.float32)), requires_grad=True)
    start = time.time()
    for _ in range(repeat):
        y = DataNode.matmul(x, w) + b
        loss = y - t
        loss = loss * loss
        # sum_all
        item = Tensor.sum_all(loss.tensor)
        # 反向
        loss.backward()
        DataNode.zero_grad()
    end = time.time()
    return (end - start) / repeat


def benchmark_torch_matmul(shape, repeat, device):
    x = torch.rand(*shape, dtype=torch.float32,
                   device=device, requires_grad=True)
    w = torch.rand(*shape, dtype=torch.float32,
                   device=device, requires_grad=True)
    for _ in range(repeat):
        z = torch.matmul(x, w)
        z.backward(torch.ones_like(z))
        x.grad.zero_()
        w.grad.zero_()
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    for _ in range(repeat):
        z = torch.matmul(x, w)
        z.backward(torch.ones_like(z))
        x.grad.zero_()
        w.grad.zero_()
    torch.cuda.synchronize() if device == 'cuda' else None
    end = time.time()
    return (end - start) / repeat


def benchmark_torch_linear(batch, in_features, out_features, repeat, device):
    x = torch.rand(batch, in_features, dtype=torch.float32,
                   device=device, requires_grad=False)
    t = torch.rand(batch, out_features, dtype=torch.float32,
                   device=device, requires_grad=False)
    w = torch.rand(in_features, out_features, dtype=torch.float32,
                   device=device, requires_grad=True)
    b = torch.rand(1, out_features, dtype=torch.float32,
                   device=device, requires_grad=True)
    for _ in range(repeat):
        y = torch.matmul(x, w) + b
        loss = (y - t) ** 2
        loss = loss.sum()
        loss.backward()
        w.grad.zero_()
        b.grad.zero_()
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    for _ in range(repeat):
        y = torch.matmul(x, w) + b
        loss = (y - t) ** 2
        loss = loss.sum()
        loss.backward()
        w.grad.zero_()
        b.grad.zero_()
    torch.cuda.synchronize() if device == 'cuda' else None
    end = time.time()
    return (end - start) / repeat


def print_result_table(results):
    print("\n=== Autograd Benchmark (avg time per op, ms) ===")
    print(f"{'Case':<20}{'FDUAI(CPU)':>15}{'Torch(CPU)':>15}{'Torch(CUDA)':>15}")
    for case in ['matmul', 'linear']:
        row = [case]
        for plat in ['fduai_cpu', 'torch_cpu', 'torch_cuda']:
            val = results.get((case, plat), None)
            if val is None:
                row.append('   -   ')
            else:
                row.append(f"{val * 1e3:.2f}")
        print(f"{row[0]:<20}{row[1]:>15}{row[2]:>15}{row[3]:>15}")


if __name__ == "__main__":
    results = {}
    # matmul
    if FDUAI_AVAILABLE:
        try:
            t = benchmark_fduai_matmul(SHAPE, REPEAT)
            results[('matmul', 'fduai_cpu')] = t
            print(
                f"[FDUAI][CPU][matmul] avg: {t * 1e3:.2f}ms", file=sys.stderr)
        except Exception as e:
            print(f"[FDUAI][CPU][matmul] failed: {e}", file=sys.stderr)
    if TORCH_AVAILABLE:
        try:
            t = benchmark_torch_matmul(SHAPE, REPEAT, 'cpu')
            results[('matmul', 'torch_cpu')] = t
            print(
                f"[torch][CPU][matmul] avg: {t * 1e3:.2f}ms", file=sys.stderr)
        except Exception as e:
            print(f"[torch][CPU][matmul] failed: {e}", file=sys.stderr)
    if TORCH_CUDA:
        try:
            t = benchmark_torch_matmul(SHAPE, REPEAT, 'cuda')
            results[('matmul', 'torch_cuda')] = t
            print(
                f"[torch][CUDA][matmul] avg: {t * 1e3:.2f}ms", file=sys.stderr)
        except Exception as e:
            print(f"[torch][CUDA][matmul] failed: {e}", file=sys.stderr)
    # linear
    if FDUAI_AVAILABLE:
        try:
            t = benchmark_fduai_linear(
                BATCH_SIZE, IN_FEATURES, OUT_FEATURES, REPEAT)
            results[('linear', 'fduai_cpu')] = t
            print(
                f"[FDUAI][CPU][linear] avg: {t * 1e3:.2f}ms", file=sys.stderr)
        except Exception as e:
            print(f"[FDUAI][CPU][linear] failed: {e}", file=sys.stderr)
    if TORCH_AVAILABLE:
        try:
            t = benchmark_torch_linear(
                BATCH_SIZE, IN_FEATURES, OUT_FEATURES, REPEAT, 'cpu')
            results[('linear', 'torch_cpu')] = t
            print(
                f"[torch][CPU][linear] avg: {t * 1e3:.2f}ms", file=sys.stderr)
        except Exception as e:
            print(f"[torch][CPU][linear] failed: {e}", file=sys.stderr)
    if TORCH_CUDA:
        try:
            t = benchmark_torch_linear(
                BATCH_SIZE, IN_FEATURES, OUT_FEATURES, REPEAT, 'cuda')
            results[('linear', 'torch_cuda')] = t
            print(
                f"[torch][CUDA][linear] avg: {t * 1e3:.2f}ms", file=sys.stderr)
        except Exception as e:
            print(f"[torch][CUDA][linear] failed: {e}", file=sys.stderr)
    print_result_table(results)
