"""
fduai.common.lib
=====================
Managing C/C++ libraries.
"""

import os 
import sys
import subprocess
import tempfile
from enum import Enum

class Lang(Enum):
    C = 'c'
    CPP = 'c++'

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

def cxx():
    if 'CXX' in os.environ:
        return os.environ['CXX']
    else:
        return 'c++'

def ld(lang: Lang = Lang.C):
    if lang == Lang.CPP:
        return cxx()

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
    def __init__(self, src: str, lang=Lang.C):
        self.opt = '-O2'
        self.lang = lang
        self.debug = None
        self.link_libs = ['-lc', '-lm']
        self.src = src

        self.cc_exe = cc() if self.lang == Lang.CPP else cxx()
        self.ld_exe = ld(self.lang)
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
        args = [self.cc_exe, '-c', self.opt, '-x', self.lang.value, self.c_source, '-o', object_file]
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
        args = [self.cc_exe, '-fPIC', '-c', self.opt, '-x', self.lang.value, self.c_source, '-o', object_file]
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

libtimer = Library("""
#include <iostream>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

static std::chrono::_V2::system_clock::time_point timers[16];
static int n_timers = 0;

extern "C" void timer_start(void) {
    if (n_timers >= 16) {
        std::cerr << "Maximum number of timers reached." << std::endl;
        return;
    }
    auto timer = Clock::now();
    timers[n_timers++] = timer;
}

extern "C" void timer_stop(void) {
    auto timer = Clock::now();
    if (n_timers <= 0) {
        std::cerr << "No timers to stop." << std::endl;
        return;
    }
    auto duration = (timer - timers[n_timers - 1]);
    std::cerr << "[Timer " << n_timers - 1 << "] ";
    std::cerr << duration.count() << std::endl;
}
""", lang=Lang.CPP)

def _copy_libs(dir: str):
    libs = {
        "libprinter": libprinter,
        "libfence": libfence,
        "libtimer": libtimer,
    }

    for lib in libs:
        libobj = libs[lib]
        # copy staic and shared lib to dir.
        subprocess.run(['cp', libobj.shared_lib, os.path.join(dir, lib + '.so')])
        subprocess.run(['cp', libobj.static_lib, os.path.join(dir, lib + '.a')])
        # extract object files
        subprocess.run([ar(), 'x', os.path.join(dir, lib + '.a'), f'--output={dir}'])
