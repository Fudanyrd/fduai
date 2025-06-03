
from fduai.autograd import DataNode
from fduai.compiler import *
from fduai.runner.pipeline import *
from fduai.runner.cpu import CPURunner

import sys

with Module() as m:
    with Function('main'):
        a = Variable.ones([2, 128])
        b = Variable.fill([128, 2], 2.0)
        c = Variable.zeros([2, 2])

        with Timer():
            with Repeat(1000):
                r = Variable.matmul(a, b)
                # c = r
                move(r, c)

        print(c)

ir = compile_module(m)
with open('matmul.mlir', 'w') as f:
    f.write(ir)
# print(ir)
ir = auto_dealloc_pass(ir)
# print(ir, file=sys.stderr)
ir = convert_to_llvm_pass(ir)
runner = CPURunner(ir, 
                   extra_link_args=['-o', 'matmul'],
                   extra_compile_args=['--O2'])

import subprocess
res = subprocess.run(['./matmul'], capture_output=False)
# l = eval(res.stdout.decode())
# assert l == [[6.0, 6.], [6.0, 6.0]]
# print(l, type(l))
