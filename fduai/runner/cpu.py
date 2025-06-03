import subprocess
import sys
import os

from fduai.common.lib import Library, libprinter, libfence, libtimer, cc, cxx

def _mktemp(dir = None) -> str:
    args = ['mktemp'] if dir is None else ['mktemp', '-p', dir]
    res = subprocess.run(
        args,
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
    Compile the input mlir to an object file and link it to an executable. If 
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
        """
        :param ir: mlir ir
        :para m extra_compile_args: extra arguments to mlir-cpu-runner.
        :param extra_link_args: extra arguments to cc as linker.

        Example to specify object file path(a.o):
        >>> CPURunner(extra_compile_args=['--object-filename', 'a.o'])

        Example to specify executable file path(a.out):
        >>> CPURunner(extra_link_args=['-o', 'a.out'])
        """
        self.ir = ir
        self.rm_obj = False
        self.rm_exe = False
        self.libprinter = libprinter
        self.libfence = libfence
        self.libtimer = libtimer

        self.shared_libs = [self.libprinter.shared_lib, self.libfence.shared_lib, libtimer.shared_lib]
        self.static_libs = [self.libprinter.static_lib, self.libfence.static_lib, libtimer.static_lib]

        # create object file
        compile_args = ['--entry-point-result=i32', '--dump-object-file'] + extra_compile_args
        compile_args.append('--shared-libs=' + ','.join(self.shared_libs))
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
            print(res.stderr.decode(), file=sys.stderr)
            raise RuntimeError("Failed to compile input IR")
        
        # create executable
        cc_exe = cxx()
        link_args = extra_link_args
        link_args.append(self.obj)

        for static_lib in self.static_libs:
            link_args.append(static_lib)
        if '-o' in extra_link_args:
            self.exe =  extra_link_args[extra_link_args.index('-o') + 1]
        else:
            self.rm_exe = True
            self.exe = _mktemp()
            link_args.append('-o')
            link_args.append(self.exe)
        res = subprocess.run([cc_exe] + link_args, capture_output=True)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.decode())

    def __del__(self):
        if self.rm_obj:
            os.unlink(self.obj)
        if self.rm_exe:
            os.unlink(self.exe)
