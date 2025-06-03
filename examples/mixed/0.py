"""
Usage:
    python 0.py && ./p.out

The output of `p.out` should be the following:
[-8.999998]
[11.000006]
"""
from fduai.autograd import DataNode
from fduai.compiler import *
from fduai.runner.pipeline import *
from fduai.runner.cpu import CPURunner

import sys

with Module() as m:
    with Function('main') as f:
        a = DataNode.ones([1, ])
        b = DataNode.ones([1, ])
        lr = Variable.fill([1, ], 1e-1)

        with Repeat(100):
            c = a - b
            c.backward()

            a1 = a.tensor - lr * a.grad
            b1 = b.tensor - lr * b.grad

            #
            # FIXME: instead of asking user to use move(), we should
            # automatically do it
            #

            # a.tensor = a1
            # b.tensor = b1
            move(a1, a.tensor)
            move(b1, b.tensor)

            DataNode.zero_grad()

        print(a.tensor)
        print(b.tensor)


ir = compile_module(m)
ir = auto_dealloc_pass(ir)
print(ir, file=sys.stderr)
ir = convert_to_llvm_pass(ir)
runner = CPURunner(ir, 
                   extra_link_args=['-o', 'p.out'],
                   extra_compile_args=['--O2'])
