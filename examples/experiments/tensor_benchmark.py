import time
import numpy as np
import sys

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from fduai.compiler import Variable
    TENSOR_MODULE_AVAILABLE = True
except ImportError:
    TENSOR_MODULE_AVAILABLE = False

# 检查 CUDA 可用性
TORCH_CUDA = TORCH_AVAILABLE and torch.cuda.is_available()
TENSOR_CUDA = False
if TENSOR_MODULE_AVAILABLE:
    try:
        # 尝试创建 Variable 张量，判断 Variable 是否可用
        _ = Variable.ones([2, 2])
        TENSOR_CUDA = True  # Variable 不区分 device，这里只做可用性判断
    except Exception:
        TENSOR_CUDA = False

SHAPE = (1000, 1000)
REPEAT = 2000

# 操作列表
OPS = [
    'add',
    'mul',
    'matmul',
    'transpose',
    'relu',
    'broadcast_add',
]

# 结果收集
timing_results = []


def benchmark_numpy(op, shape, repeat):
    a = np.random.rand(*shape).astype(np.float32)
    b = np.random.rand(*shape).astype(np.float32)
    if op == 'add':
        start = time.time()
        for _ in range(repeat):
            c = a + b
        end = time.time()
    elif op == 'mul':
        start = time.time()
        for _ in range(repeat):
            c = a * b
        end = time.time()
    elif op == 'matmul':
        start = time.time()
        for _ in range(repeat):
            c = np.dot(a, b)
        end = time.time()
    elif op == 'transpose':
        start = time.time()
        for _ in range(repeat):
            c = a.T
        end = time.time()
    elif op == 'relu':
        start = time.time()
        for _ in range(repeat):
            c = np.maximum(a, 0)
        end = time.time()
    elif op == 'broadcast_add':
        b = np.random.rand(1, shape[1]).astype(np.float32)
        start = time.time()
        for _ in range(repeat):
            c = a + b
        end = time.time()
    else:
        return None
    return (end - start) / repeat


def benchmark_torch(op, shape, repeat, device):
    a = torch.rand(*shape, dtype=torch.float32, device=device)
    b = torch.rand(*shape, dtype=torch.float32, device=device)
    if op == 'add':
        start = time.time()
        for _ in range(repeat):
            c = a + b
        end = time.time()
    elif op == 'mul':
        start = time.time()
        for _ in range(repeat):
            c = a * b
        end = time.time()
    elif op == 'matmul':
        start = time.time()
        for _ in range(repeat):
            c = torch.matmul(a, b)
        end = time.time()
    elif op == 'transpose':
        start = time.time()
        for _ in range(repeat):
            c = a.t()
        end = time.time()
    elif op == 'relu':
        start = time.time()
        for _ in range(repeat):
            c = torch.relu(a)
        end = time.time()
    elif op == 'broadcast_add':
        b = torch.rand(1, shape[1], dtype=torch.float32, device=device)
        start = time.time()
        for _ in range(repeat):
            c = a + b
        end = time.time()
    else:
        return None
    if device == 'cuda':
        torch.cuda.synchronize()
    return (end - start) / repeat


def benchmark_variable(op, shape, repeat):
    # Variable 不区分 device，始终在 IR 层
    a = Variable.ones(list(shape))
    b = Variable.ones(list(shape))
    if op == 'add':
        start = time.time()
        for _ in range(repeat):
            c = a + b
        end = time.time()
    elif op == 'mul':
        start = time.time()
        for _ in range(repeat):
            c = a * b
        end = time.time()
    elif op == 'matmul':
        start = time.time()
        for _ in range(repeat):
            c = Variable.matmul(a, b)
        end = time.time()
    elif op == 'transpose':
        start = time.time()
        for _ in range(repeat):
            c = Variable.transpose(a)
        end = time.time()
    elif op == 'relu':
        start = time.time()
        for _ in range(repeat):
            c = Variable.relu(a)
        end = time.time()
    elif op == 'broadcast_add':
        b = Variable.ones([1, shape[1]])
        start = time.time()
        for _ in range(repeat):
            c = a + b
        end = time.time()
    else:
        return None
    return (end - start) / repeat


def print_result_table(results):
    # 打印表头
    print("\n=== Tensor Operation Benchmark (avg time per op, microseconds) ===")
    print(f"{'Operation':<15}{'Variable':>15}{'NumPy':>15}{'Torch(CPU)':>15}{'Torch(CUDA)':>15}")
    for op in OPS:
        row = [op]
        for plat in ['variable', 'numpy', 'torch_cpu', 'torch_cuda']:
            val = results.get((op, plat), None)
            if val is None:
                row.append('   -   ')
            else:
                row.append(f"{val * 1e6:.0f}")
        print(
            f"{row[0]:<15}{row[1]:>15}{row[2]:>15}{row[3]:>15}{row[4]:>15}")


if __name__ == "__main__":
    results = {}
    for op in OPS:
        # Variable (IR)
        if TENSOR_MODULE_AVAILABLE:
            try:
                t = benchmark_variable(op, SHAPE, REPEAT)
                if t is not None:
                    results[(op, 'variable')] = t
                    print(
                        f"[Variable][{op}] avg: {t * 1e6:.0f}μs", file=sys.stderr)
                else:
                    print(
                        f"[Variable][{op}] skipped (not supported)", file=sys.stderr)
            except Exception as e:
                print(
                    f"[Variable][{op}] failed: {e}", file=sys.stderr)
        # NumPy
        try:
            t = benchmark_numpy(op, SHAPE, REPEAT)
            results[(op, 'numpy')] = t
            print(f"[numpy][CPU][{op}] avg: {t * 1e6:.0f}μs", file=sys.stderr)
        except Exception as e:
            print(f"[numpy][CPU][{op}] failed: {e}", file=sys.stderr)
        # Torch CPU
        if TORCH_AVAILABLE:
            try:
                t = benchmark_torch(op, SHAPE, REPEAT, 'cpu')
                results[(op, 'torch_cpu')] = t
                print(
                    f"[torch][CPU][{op}] avg: {t * 1e6:.0f}μs", file=sys.stderr)
            except Exception as e:
                print(f"[torch][CPU][{op}] failed: {e}", file=sys.stderr)
        # Torch CUDA
        if TORCH_CUDA:
            try:
                t = benchmark_torch(op, SHAPE, REPEAT, 'cuda')
                results[(op, 'torch_cuda')] = t
                print(
                    f"[torch][CUDA][{op}] avg: {t * 1e6:.0f}μs", file=sys.stderr)
            except Exception as e:
                print(f"[torch][CUDA][{op}] failed: {e}", file=sys.stderr)
    print_result_table(results)
