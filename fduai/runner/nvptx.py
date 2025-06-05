"""
fduai.runner.nvptx
==========

Our mlir-opt configure options:
cmake -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" 
      -DMLIR_ENABLE_CUDA_RUNNER=ON -DLLVM_BUILD_LLVM_DYLIB=ON 
      -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles" ../llvm

Reference of building mlir with cuda runner enabled:
https://discourse.llvm.org/t/how-to-use-gpu-functions/62529
https://reviews.llvm.org/D135981
"""

import subprocess
import sys
import os

from fduai.common.lib import Library, libprinter, libfence, libtimer, cc, cxx

def _lib_cuda_runtime() -> str:
    """
    :return: path to libmlir_cuda_runtime.so
    """
    if 'LIBMLIR_CUDA_RUNTIME' in os.environ:
        return os.environ['LIBMLIR_CUDA_RUNTIME']
    return os.path.join(os.environ['LLVM_INSTALL_DIR'], 'lib', 'libmlir_cuda_runtime.so')

class CudaRunner():
    EXAMPLE_IR = """
func.func @main() -> i32 {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 2 : index

    %a1 = memref.alloc() : memref<2x2xf32>
    %a2 = memref.alloc() : memref<2x2xf32>
    %a3 = memref.alloc() : memref<2x2xf32>

    gpu.launch
        blocks(%0, %1, %2) in (%3 = %c1, %4 = %c1, %5 = %c1)
        threads(%6, %7, %8) in (%9 = %c2, %10 = %c2, %11 = %c2) {
        %s1 = arith.constant 1.0 : f32
        memref.store %s1, %a1[%0, %1] : memref<2x2xf32>
        memref.store %s1, %a2[%0, %1] : memref<2x2xf32>

        gpu.terminator
    }

    gpu.launch
        blocks(%0, %1, %2) in (%3 = %c1, %4 = %c1, %5 = %c1)
        threads(%6, %7, %8) in (%9 = %c2, %10 = %c2, %11 = %c2) {
        %s1 = memref.load %a1[%0, %1] : memref<2x2xf32>
        %s2 = memref.load %a2[%0, %1] : memref<2x2xf32>
        %s3 = arith.addf %s1, %s2 : f32
        memref.store %s3, %a3[%0, %1] : memref<2x2xf32>

        gpu.terminator
    }

    %main_ret = arith.constant 0 : i32
    return %main_ret : i32
}\n"""
    def __init__(self, ir: str):
        # FIXME
        self.ir = ir
        raise NotImplementedError("CudaRunner not implemented")

        # possible solution:
        # mlir-opt -gpu-lower-to-nvvm-pipeline --convert-gpu-to-nvvm \
        # --convert-to-llvm --gpu-module-to-binary --convert-to-llvm \
        # --reconcile-unrealized-casts ./gpu.mlir | \
        # mlir-runner --entry-point-result=i32 \
        # --shared-libs=/lib/libmlir_cuda_runtime.so.20.1 
