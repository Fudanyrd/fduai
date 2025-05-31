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
"""
