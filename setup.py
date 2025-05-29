"""
To install this package, run:
    CC=$PWD/tensor/cc CXX=$PWD/tensor/cc pip install .
"""
from setuptools import setup, Extension, find_packages, find_namespace_packages
import subprocess
import os
import glob
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "tensor_module",
        [os.path.join('tensor', f) for f in os.listdir('./tensor') 
         if f.endswith('.cpp') or f.endswith('.cc')],
        extra_compile_args=["-I/usr/local/cuda/include", "-fopenmp"],
        extra_link_args=["-fopenmp", "-L/usr/local/cuda/lib64", 
                         "-lcudart", "-lcublas", "-lm", "-lpthread"],
    ),
]

if __name__ == '__main__':
    os.environ["CC"] = os.path.join(os.getcwd(), 'tensor/cc')
    os.environ["DEFS"] = "__TEST__"

    setup(
        name='fduai',
        version='0.1.0',
        description='A simple deep learning framework',
        author='Rundong Yang',
        author_email='<EMAIL>',
        url='https://github.com/Fudanyrd/fduai',
        packages=find_packages(exclude=['examples']),
        install_requires=['pybind11', 'setuptools'],
        ext_modules=ext_modules,
    )
