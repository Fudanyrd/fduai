from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os
import subprocess
import sys

def get_includes() -> list:
	res = subprocess.run(
		["python3", "-m", "pybind11", "--includes"],
		capture_output=True
	)
	if res.returncode != 0:
		return []
	
	ret= res.stdout.decode().split()

	for i in range(len(ret)):
		ret[i] = ret[i].strip()
		if len(ret[i]) == 0:
			ret.remove(ret[i])
	
	return ret

def find_cpp_source() -> list:
    """
    :return: list of cpp source files
    """
    return [f for f in os.listdir('.') if f.endswith('.cpp') or f.endswith('.cc')]

ext_modules = [
    Pybind11Extension(
        "tensor_module",
        find_cpp_source(),
        # extra_compile_args=["-I/usr/local/cuda/include", "-lcudart", "-lcublas", "-fopenmp"],
        extra_compile_args=["-I/usr/local/cuda/include", "-fopenmp"],
        extra_link_args=["-fopenmp", "-L/usr/local/cuda/lib64", 
                         "-lcudart", "-lcublas", "-lm", "-lpthread"],
    ),
]

def build_backend(cuda_source: list = ['backend.cu']):
    ret = subprocess.run(
        ['nvcc'] + get_includes() + cuda_source + ['-Xcompiler', '-fpic', '-shared', '-O0', '-g', '-lm',  '-o', 'backend.so'], 
        capture_output=True)
    
    if ret.returncode != 0:
        print(ret.stderr.decode('utf-8'), file=sys.stderr)
        raise RuntimeError("nvcc failed")

if __name__ == "__main__":
    # build_backend()
    setup(
        name="tensor_module",
        version="0.0.1",
        author="Your Name",
        author_email="your.email@example.com",
        description="A tensor computation library",
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
    )
