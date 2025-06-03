from fduai.compiler import *
from fduai.runner.pipeline import convert_to_llvm_pass, auto_dealloc_pass, affine_accelerate_pass, PassPipeline
from fduai.runner.cpu import CPURunner
import sys
import time

shape = [1024, 1024]
with Module() as m:
    with Function('main'):
        a = Variable.zeros(shape)
        b = Variable.zeros(shape)
        with Timer():
            with Repeat(1000):
                c = a + b
parallelize = PassPipeline(
    '--affine-loop-fusion',
    '--affine-parallelize')

start = time.time()
code = (compile_module(m))
code = auto_dealloc_pass(code)
code = affine_accelerate_pass(code)
code = convert_to_llvm_pass(code)
runner = CPURunner(code, extra_link_args=['-o', 'add.out'])
end = time.time()

print(code)
print(f'Compile time: {round(end - start, 5)}', file=sys.stderr)

import numpy as np

start = time.time()
a = np.random.random(shape)
b = np.random.random(shape)
for _ in range(1000):
    c =a + b
    del c
end = time.time()

print(f"Time taken for numpy: {round(end-start, 5)}", file=sys.stderr)
del a 
del b

import torch
start = time.time()
a = torch.zeros(shape, requires_grad=False)
b = torch.zeros(shape, requires_grad=False)
for _ in range(1000):
    with torch.no_grad():
        c = a + b
        del c
end = time.time()
print(f"Time taken for torch: {round(end-start, 5)}", file=sys.stderr)
