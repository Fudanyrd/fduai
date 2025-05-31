import subprocess
import sys
import os

def _cc_compiler() -> str:
    if 'CC' in os.environ:
        return os.environ['CC']
    else:
        gcc = subprocess.check_output(['which', 'gcc']).decode('utf-8').strip()
        if  gcc:
            return gcc
        else:
            return 'cc'

def _mktemp() -> str:
    res = subprocess.run(
        ['mktemp'],
        capture_output=True
    )

    if res.returncode != 0:
        raise RuntimeError(res.stderr.decode('utf-8'))

    return res.stdout.decode('utf-8').strip()

def _cpu_runner_exe() -> str:
    if 'MLIR_CPU_RUNNER' in os.environ:
        return os.environ['MLIR_CPU_RUNNER']

    res = subprocess.run(['which', 'mlir-cpu-runner'])
    if res.returncode == 0:
        return 'mlir-cpu-runner'
    else:
        return 'mlir-runner'

class CPURunner():
    """
    Compile the input mlir and link it to an executable. If 
    object file and executable are not specified, they will be
    removed after the execution.

    Example:
    >>> from fduai.compiler import generate_mlir
    >>> from fduai.runner import CPURunner
    >>> ir = generate_mlir()
    >>> runner = CPURunner(ir, 
    ...                    extra_link_args=['-lm', '-o', 'a.out'])
    >>> print(runner.exe)
    a.out
    """
    def __init__(self, ir: str, 
                 extra_compile_args: list = ['--O0'],
                 extra_link_args: list = ["-lc", '-lm']):
        self.ir = ir
        self.rm_obj = False

        # create object file
        compile_args = ['--entry-point-result=i32', '--dump-object-file'] + extra_compile_args
        if '--object-filename' in extra_compile_args:
            self.obj = extra_compile_args[extra_compile_args.index('--object-filename') + 1]
        else:
            self.rm_obj = True
            self.obj = _mktemp()
            compile_args.append('--object-filename=' + self.obj)
        cpu_runner = _cpu_runner_exe()
        res = subprocess.run(
            [cpu_runner] + compile_args,
            input=self.ir.encode('utf-8'),
            capture_output=True)
        if res.returncode != 0:
            print(f"=== compile args are: {compile_args}", file=sys.stderr)
            print('=== mlir-cpu-runner output:', file=sys.stderr)
            print(res.stdout.decode(), file=sys.stderr)
            raise RuntimeError("Failed to compile input IR")
        
        # create executable
        cc = _cc_compiler()
        link_args = extra_link_args
        link_args.append(self.obj)
        self.rm_exe = False
        if '-o' in extra_link_args:
            self.exe =  extra_link_args[extra_link_args.index('-o') + 1]
        else:
            self.rm_exe = True
            self.exe = _mktemp()
            link_args.append('-o')
            link_args.append(self.exe)
        res = subprocess.run([cc] + link_args, capture_output=True)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.decode())

    def __del__(self):
        if self.rm_obj:
            os.unlink(self.obj)
        if self.rm_exe:
            os.unlink(self.exe)
