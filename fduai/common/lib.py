"""
fduai.common.lib
=====================
Managing C/C++ libraries.
"""

import os 
import sys
import subprocess
import tempfile

def mktemp(dir = None) -> str:
    args = ['mktemp'] if dir is None else ['mktemp', '-p', dir]
    res = subprocess.run(
        args,
        capture_output=True
    )

    if res.returncode != 0:
        raise RuntimeError(res.stderr.decode('utf-8'))

    return res.stdout.decode('utf-8').strip()

def cc():
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

def ld():
    if 'LD' in os.environ:
        return os.environ['LD']
    else:
        return 'ld'

def ar():
    if 'AR' in os.environ:
        return os.environ['AR']
    else:
        return 'ar'

class Library():
    def __init__(self, src: str):
        self.opt = '-O2'
        self.debug = None
        self.link_libs = ['-lc', '-lm']
        self.src = src

        self.cc_exe = cc()
        self.ld_exe = ld()
        self.ar_exe = ar()

        self.c_source = mktemp()
        with open(self.c_source, 'w') as fobj:
            fobj.write(self.src)

        self.static_lib = tempfile.mktemp()
        self.shared_lib = mktemp()

        self._build_static_lib()
        self._build_shared_lib()

    def _build_static_lib(self):
        object_file = mktemp()
        args = [self.cc_exe, '-c', self.opt, '-x', 'c', self.c_source, '-o', object_file]
        if self.debug:
            args.append(self.debug)

        res = subprocess.run(args, capture_output=True)
        if res.returncode != 0:
            print('=== Arguments', file=sys.stderr)
            print(args, file=sys.stderr)
            print('=== stderr')
            print( res.stderr.decode(), file=sys.stderr)

            raise RuntimeError(f'=== CC failed with {res.returncode}')

        args = [self.ar_exe, 'rcs',  self.static_lib, object_file]
        res = subprocess.run(args, capture_output=True)
        if res.returncode != 0:
            print('=== Arguments', file=sys.stderr)
            print(args, file=sys.stderr)
            print('=== stderr')
            print( res.stderr.decode(), file=sys.stderr)

            raise RuntimeError(f'=== AR failed with {res.returncode}')

        os.unlink(object_file)

    def _build_shared_lib(self):
        object_file = mktemp()
        args = [self.cc_exe, '-fPIC', '-c', self.opt, '-x', 'c', self.c_source, '-o', object_file]
        if self.debug:
            args.append(self.debug)

        res = subprocess.run(args, capture_output=True)
        if res.returncode != 0:
            print('=== Arguments', file=sys.stderr)
            print(args, file=sys.stderr)
            print('=== stderr')
            print( res.stderr.decode(), file=sys.stderr)

            raise RuntimeError(f'=== CC failed with {res.returncode}')

        args = [self.ld_exe, '-fpic', '-shared', object_file, '-o', self.shared_lib]
        args += self.link_libs
        res = subprocess.run(args, capture_output=True)
        if res.returncode != 0:
            print('=== Arguments', file=sys.stderr)
            print(args, file=sys.stderr)
            print('=== stderr')
            print( res.stderr.decode(), file=sys.stderr)

            raise RuntimeError(f'=== LD failed with {res.returncode}')

        os.unlink(object_file)

    def __del__(self):
        os.unlink(self.static_lib)
        os.unlink(self.shared_lib)
        os.unlink(self.c_source)

class CXXLibrary():
    # FIXME: not implemented
    pass

libprinter = Library("""
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
""")

libfence = Library("""
void m_fence(const float value __attribute__((unused))) {
    asm volatile("" : : : "memory");
}
""")
