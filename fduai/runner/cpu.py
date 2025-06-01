import subprocess
import sys
import os

_printer = """
#include <stddef.h>
#include <stdio.h>

void json_list_start() {
    printf("[");
}

void json_list_end() {
    printf("]");
}

void json_list_sep() {
    printf(",");
}

void json_f32(float value) {
    printf("%f", value);
}

void json_list_data(const float *buf, size_t len) {
    json_list_start();

    json_f32(buf[0]);
    for (size_t i = 1; i < len; i++) {
        json_list_sep();
        json_f32(buf[i]);
    }

    json_list_end();
}

void new_line() {
    printf("\\n");
    fflush(stdout);
}
"""

def _cc_compiler() -> str:
    if 'CC' in os.environ:
        return os.environ['CC']
    elif 'CCLD' in os.environ:
        return os.environ['CCLD']
    else:
        gcc = subprocess.check_output(['which', 'gcc']).decode('utf-8').strip()
        if  gcc:
            return gcc
        else:
            return 'cc'

def _mktemp(dir = None) -> str:
    args = ['mktemp'] if dir is None else ['mktemp', '-p', dir]
    res = subprocess.run(
        args,
        capture_output=True
    )

    if res.returncode != 0:
        raise RuntimeError(res.stderr.decode('utf-8'))

    return res.stdout.decode('utf-8').strip()

def _printer_lib():
    src = _mktemp()

    with open(src, 'w') as fobj:
        fobj.write(_printer)

    obj = _mktemp()
    lib = _mktemp()

    # compile
    res = subprocess.run(
        [_cc_compiler()] + ['-fpic', '-c', '-O2', '-g', '-x', 'c', src, '-o', obj],
        capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.decode('utf-8'))

    # link shared library
    res = subprocess.run(
        ['/usr/bin/env', 'ld', obj, '-shared', '-o', lib, '-lc', '-lm'],
        capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.decode('utf-8'))

    # os.unlink(src)
    os.unlink(obj)

    static_lib = _mktemp()
    # compile static lib
    res = subprocess.run(
        [_cc_compiler()] + ['-c', '-O2', '-g', '-x', 'c', src, '-o', static_lib],
        capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.decode('utf-8'))

    os.unlink(src)
    return lib, static_lib

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
        self._libprinter, \
        self._libprinter_static = _printer_lib()

        # create object file
        compile_args = ['--entry-point-result=i32', '--dump-object-file'] + extra_compile_args
        compile_args.append('--shared-libs=' + self._libprinter)
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
        cc = _cc_compiler()
        link_args = extra_link_args
        link_args.append(self.obj)
        link_args.append(self._libprinter_static) # link libprinter static lib
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

        # remove libprinter
        os.unlink(self._libprinter)
        os.unlink(self._libprinter_static)
