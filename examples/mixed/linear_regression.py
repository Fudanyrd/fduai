import subprocess
import numpy as np
from fduai.autograd import DataNode
from fduai.compiler import *
from fduai.runner.pipeline import *
from fduai.runner.cpu import CPURunner
import sys

# Generate synthetic data (same as autograd example)
n_sample, n_feature = 256, 1
np.random.seed(42)
X = np.random.random((n_sample, n_feature)).astype(np.float32) * 4.0
noise = np.random.random((n_sample, 1)).astype(np.float32) * 0.1
Y = 2 * X + 1 + noise

# Flatten to 1D for MLIR embedding if needed
X_flat = X.flatten().tolist()
Y_flat = Y.flatten().tolist()

with Module() as m:
    with Function('main'):
        # Data as constants
        Xv = DataNode.from_list([n_sample, n_feature], X.tolist())
        Yv = DataNode.from_list([n_sample, 1], Y.tolist())
        w = DataNode.zeros([n_feature, 1])
        b = DataNode.zeros([1])
        lr = Variable.fill([1], 1e-4)
        max_iter = 100

        with Repeat(max_iter):
            # Forward: l = X @ w + b - Y
            l = DataNode.matmul(Xv, w) + b - Yv
            loss = l * l
            # Scalar loss for convergence check (not used for break here)
            # item = Tensor.sum_all(loss.tensor) / 2 / len(X_flat)

            loss.backward()
            # Gradient step
            w1 = w.tensor - w.grad * lr
            b1 = b.tensor - b.grad * lr
            move(w1, w.tensor)
            move(b1, b.tensor)
            DataNode.zero_grad()

        print(w.tensor)
        print(b.tensor)

ir = compile_module(m)
with open('linear_regression.mlir', 'w') as f:
    f.write(ir)
ir = auto_dealloc_pass(ir)
ir = convert_to_llvm_pass(ir)
runner = CPURunner(ir,
                   extra_link_args=['-o', 'linear_regression.out'],
                   extra_compile_args=['--O2'])

res = subprocess.run(['./linear_regression.out'], capture_output=False)
