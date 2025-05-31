import subprocess
import sys
import os

def _mlir_opt_exe() -> str:
    if 'MLIR_OPT' in os.environ:
        return os.environ['MLIR_OPT']
    else:
        return 'mlir-opt'

class PassPipeline():
    """
    Use `mlir-opt --help` to list all available passes.

    Example:
    >>> from fduai.runner.pipeline import convert_to_llvm_pass
    >>> print(convert_to_llvm_pass(ir))
    """
    def __init__(self, *args):
        self.args = list(args)

    def add_arg( self, arg):
        self.args.append(arg)

    def __call__(self, ir: str):
        """
        :param ir: input ir to be transformed by the pipeline
        :return: transformed ir
        """
        res = subprocess.run(
            [_mlir_opt_exe()] +  self.args + ['-o', '-'],
            input=ir.encode(),
            capture_output=True
        )

        if res.returncode != 0:
            print('=== mlir-opt stderr:', file=sys.stderr)
            print(res.stderr.decode(), file=sys.stderr)
            raise RuntimeError("mlir-opt failed.")

        return res.stdout.decode()

# add necessary memory free operations
auto_dealloc_pass = PassPipeline('--buffer-deallocation')

# convert dialects to llvm dialect
convert_to_llvm_pass = PassPipeline(
    '--lower-affine',
    '--convert-scf-to-cf',
    '--convert-to-llvm',
    '--reconcile-unrealized-casts',
)

affine_accelerate_pass = PassPipeline(
    '--affine-simplify-structures',
    '--affine-loop-fusion',
    '--affine-parallelize',
    '--affine-loop-unroll',
    '--affine-super-vectorize',
)
