"""
fduai.runner 
===========
For executing mlir code.

Environment Variables:
    CC: C compiler to use
    MLIR_OPT: mlir-opt executable to use
    MLIR_CPU_RUNNER: mlir-cpu-runner executable to use

Example:
>>> from fduai.compiler import *
>>> from fduai.runner.pipeline import convert_to_llvm_pass, auto_dealloc_pass
>>> from fduai.runner.cpu import CPURunner
>>> with Module() as m:
>>>     shape = [1024, 1024]
>>>     with Function('main'):
>>>         a = Variable.zeros(shape)
>>>         b = Variable.zeros(shape)
>>>         with Repeat(1000):
>>>             c = a + b
>>> code = (compile_module(m))
>>> code = auto_dealloc_pass(code)
>>> code = convert_to_llvm_pass(code)
>>> runner = CPURunner(code, extra_link_args=['-o', 'add.out'])

Example of printing a tensor in JSON format:
>>> from fduai.compiler import *
>>> from fduai.runner.pipeline import *
>>> from fduai.runner.cpu import CPURunner
>>> with Module() as m:
>>>     with Function('main') as f:
>>>         a = Variable.ones([2, 2, 2])
>>>         _ = a.__repr__()
>>> ir = compile_module(m)
>>> ir = convert_to_llvm_pass(ir)
>>> runner = CPURunner(ir, extra_link_args=['-o', 'p.out'])

Then execute `p.out` via shell, which should print:
>>> [[[1.000000,1.000000],[1.000000,1.000000]],[[1.000000,1.000000],[1.000000,1.000000]]]

This package is to be used together with `fduai.autograd` so that 
we can train neural networks. For example,
>>> from fduai.autograd import DataNode
>>> from fduai.compiler import *
>>> from fduai.runner.pipeline import *
>>> from fduai.runner.cpu import CPURunner
>>> with Module() as m:
>>>     with Function('main') as f:
>>>         a = DataNode.ones([4, 2])
>>>         b = DataNode.ones([1, 2])
>>>         c = a + b
>>>         c.backward()
>>>         grad_a = a.grad
>>>         grad_b = b.grad
>>>         _ = grad_a.__repr__()
>>>         _ = grad_b.__repr__()
>>> ir = compile_module(m)
>>> ir = convert_to_llvm_pass(ir)
>>> runner = CPURunner(ir, extra_link_args=['-o', 'p.out'],
>>>                    extra_compile_args=['--O3'])
"""
