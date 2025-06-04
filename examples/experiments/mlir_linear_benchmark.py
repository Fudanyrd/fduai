import time
import numpy as np
import subprocess
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

# 数据参数
n_sample, n_feature = 256, 1
np.random.seed(42)
X = np.random.random((n_sample, n_feature)).astype(np.float32) * 4.0
noise = np.random.random((n_sample, 1)).astype(np.float32) * 0.1
Y = 2 * X + 1 + noise

REPEAT = 50

# 1. MLIR 线性回归（只计可执行文件运行时间）


def benchmark_mlir_linear():
    # 先确保 examples/mixed/linear_regression.out 存在
    bin_path = 'examples/mixed/linear_regression.out'
    times = []
    for _ in range(REPEAT):
        start = time.time()
        res = subprocess.run([f'./{bin_path}'], capture_output=True)
        end = time.time()
        times.append(end - start)
    return np.mean(times)

# 2. FDUAI 纯 Python autograd 线性回归


def benchmark_fduai_autograd():
    times = []
    for _ in range(REPEAT):
        Xn = DataNode(Tensor.from_numpy(X), requires_grad=False)
        Yn = DataNode(Tensor.from_numpy(Y), requires_grad=False)
        w = DataNode(Tensor.zeros((n_feature, 1), Device.CPU))
        b = DataNode(Tensor.zeros((1, ), Device.CPU))
        lr = 1e-4
        max_iter = 100
        start = time.time()
        for _ in range(max_iter):
            l = DataNode.matmul(Xn, w) + b - Yn
            loss = l * l
            item = Tensor.sum_all(loss.tensor) / 2 / len(X)
            loss.backward()
            w.tensor -= w.grad * lr
            b.tensor -= b.grad * lr
            DataNode.zero_grad()
        end = time.time()
        times.append(end - start)
    return np.mean(times)

# 3. PyTorch 线性回归（可选）


def benchmark_torch():
    if not TORCH_AVAILABLE:
        return None
    times = []
    for _ in range(REPEAT):
        x = torch.from_numpy(X)
        y = torch.from_numpy(Y)
        w = torch.zeros((n_feature, 1), dtype=torch.float32,
                        requires_grad=True)
        b = torch.zeros((1,), dtype=torch.float32, requires_grad=True)
        lr = 1e-4
        max_iter = 100
        start = time.time()
        for _ in range(max_iter):
            y_pred = x @ w + b
            loss = ((y_pred - y) ** 2).sum() / 2 / len(x)
            loss.backward()
            with torch.no_grad():
                w -= w.grad * lr
                b -= b.grad * lr
                w.grad.zero_()
                b.grad.zero_()
        end = time.time()
        times.append(end - start)
    return np.mean(times)


def print_result_table(results):
    print("\n=== Linear Regression Benchmark (avg time, ms) ===")
    print(f"{'Case':<20}{'Time (ms)':>15}")
    for k, v in results.items():
        if v is None:
            print(f"{k:<20}{'-':>15}")
        else:
            print(f"{k:<20}{v * 1e3:>15.2f}")


if __name__ == "__main__":
    results = {}
    try:
        t = benchmark_mlir_linear()
        results['FDUAI-MLIR'] = t
        print(f"[FDUAI-MLIR] avg: {t * 1e3:.2f}ms", file=sys.stderr)
    except Exception as e:
        print(f"[FDUAI-MLIR] failed: {e}", file=sys.stderr)
        results['FDUAI-MLIR'] = None
    try:
        t = benchmark_fduai_autograd()
        results['FDUAI-Autograd'] = t
        print(f"[FDUAI-Autograd] avg: {t * 1e3:.2f}ms", file=sys.stderr)
    except Exception as e:
        print(f"[FDUAI-Autograd] failed: {e}", file=sys.stderr)
        results['FDUAI-Autograd'] = None
    try:
        t = benchmark_torch()
        results['PyTorch'] = t
        print(f"[PyTorch] avg: {t * 1e3:.2f}ms", file=sys.stderr)
    except Exception as e:
        print(f"[PyTorch] failed: {e}", file=sys.stderr)
        results['PyTorch'] = None
    print_result_table(results)
