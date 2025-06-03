import json
from enum import Enum
import re
import numpy as np

from fduai.common import Operator
from fduai.common import Mlir

class CompilerContext:
    compiling: bool = False
    compiler = None

class Compiler:

    def __init__(self):
        self.vars = []
        self.shapes = {}
        self.instructions = []

        self.allocated = set()
        self.globl_var = []

        self.args = []
        self.ret = []

        self.prev = False
        self.prev_compiler = None

        self.n_const = 0

    def next_constant(self) -> int:
        ret = self.n_const 
        self.n_const += 1
        return ret

    def __enter__(self):

        self.prev = CompilerContext.compiling
        self.prev_compiler = CompilerContext.compiler
        # self.vars = []
        # self.shapes = {}
        # self.instructions = []

        CompilerContext.compiling = True
        CompilerContext.compiler = self # type: ignore

        return self

    def retshape(self):
        ret = []
        for var in self.ret:
            ret.append(self.shapes[var])

        return ret

    def add_insn(self, op: Operator, output, *inputs):
        if output not in self.vars:
            self.vars.append(output)
        for input in inputs:
            if input not in self.vars:
                self.vars.append(input)
        
        if op == Operator.RETURN:
            self.ret = [output]
        self.instructions.append((op.value, output, inputs))

    def add_ret_stmt(self, ret):
        if ret:
            self.instructions.append(('return', ret[0], []))
            self.ret = ret
        else:
            self.instructions.append(('return', None, []))
            self.ret = []

    def add_globl_var(self, name: str):
        if name not in self.shapes:
            raise RuntimeError(f'Global variable {name} not found')

        if name not in self.globl_var:
            self.globl_var.append(name)

    def add_arg(self, name: str):
        if name not in self.shapes:
            raise RuntimeError(f'Argument {name} not found')

        if name not in self.args:
            self.args.append(name)

    def __exit__(self, *args):
        CompilerContext.compiling = self.prev
        compiler = self.prev_compiler

    def __repr__(self):
        d = {
            "instructions": self.instructions,
            "symbols": self.shapes,
        }
        return json.dumps(d)

def _broadcast(shape1: list[int], shape2: list[int]) -> list[int]:
    ret = [1 for _ in range(max(len(shape1), len(shape2)))]
    i1, i2 = len(shape1) - 1, len(shape2) - 1

    while i1 >= 0 and i2 >= 0:
        i = len(ret) - len(shape1) + i1
        if shape1[i1] == shape2[i2]:
            ret[i] = shape1[i1]
        elif shape1[i1] == 1:
            ret[i] = shape2[i2]
        elif shape2[i2] == 1:
            ret[i] = shape1[i1]
        else:
            raise ValueError('Incompatible shapes: %s and %s' % (shape1, shape2))
        i1 -= 1
        i2 -= 1

    while i1 >= 0:
        i = len(ret) - len(shape1) + i1
        ret[i] = shape1[i1]
        i1 -= 1

    while i2 >= 0:
        i = len(ret) - len(shape2) + i2
        ret[i] = shape2[i2]
        i2 -= 1

    return ret

class DType(Enum):
    FLOAT32  = 'float32'

    @staticmethod
    def sizeof(dtype):
        # FIXME: only support float32
        return 4

class MemoryOp(Enum):
    ALLOC = 'alloc'
    FREE = 'free'

class Instruction():
    _mlir_bin_ops = {
        Operator.ADD: 'addf',
        Operator.SUB: 'subf',
        Operator.MUL: 'mulf',
    }

    _mlir_cmp_op = {
        Operator.GT: 'ogt',
        Operator.LT: 'olt',
    }

    def __init__(self, op: Operator, output: str, inputs: list[str], compiler = None):
        self.op = op
        self.output = output
        self.inputs = inputs
        self.compiler = CompilerContext.compiler if compiler is None else compiler

    @staticmethod 
    def _mlir_shape(shape) -> str:
        return '<' + 'x'.join([str(x) for x in shape]) + 'xf32>'

    @staticmethod
    def _mlir_index(output_shape, input_shape) -> str:
        assert len(output_shape) >= len(input_shape)
        d = len(output_shape) - len(input_shape)
        ret = []
        for i in range(len(output_shape) - 1, d - 1, -1):
            ret.append(f'%arg{i}' if input_shape[i - d] > 1 else '%zero')
        ret.reverse()
        return '[' + ', '.join(ret) + ']'

    def _mlir_init(self, ir: list, var, data, idx: list = [], indent: int = 2):
        #
        # FIXME: our implementation of `init` copies each entry
        # to the output. This is not efficient. TO OUR KNOWLEDGE,
        # there is not a simplier way to do this.
        #
        shape = self.compiler.shapes[var]
        if len(idx) == len(shape):
            ret = '\t' * indent
            ret += 'affine.for %tmp = 0 to 1 {\n'

            indent += 1
            for i in range(len(idx)):
                ret += '\t' * indent
                ret += f'%arg{i} = arith.constant {idx[i]} : index\n'

            ret += '\t' * indent
            ret += f'%s = arith.constant {float(data)} : f32\n'

            ret += '\t' * indent
            ret += f'memref.store %s, {var}{self._mlir_index(shape, shape)} : memref{self._mlir_shape(shape)}\n'

            indent -= 1
            ret += '\t' * indent
            ret += "}\n"
            ir.append(ret)
            return

        for i in range(len(data)):
            self._mlir_init(ir, var, data[i], idx + [i], indent)

    def _mlir_next_const(self):
        return '%const' + str(self.compiler.next_constant())

    def generate_mlir(self, indent: int = 2) -> str:
        if self.compiler is None:
            raise ValueError('Compiler not set')

        if self.op == Operator.ASM:
            return '\t' * indent + self.output + '\n'

        if self.op == Operator.RETURN and self.compiler.ret is None:
            return ('\t' * indent) + "return\n"

        if self.op == Operator.INIT:
            if self.output not in self.compiler.allocated:
                self.compiler.allocated.add(self.output)

                # add an instruction that allocates the memory
                ins = '\t' * indent
                shape = self.compiler.shapes[self.output]
                ins += f'{self.output} = memref.alloc() : memref{self._mlir_shape(shape)}\n'
                buf = [ins]
            else:
                buf = []
            self._mlir_init(buf, self.output, self.inputs[0], idx=[], indent=indent)
            return ''.join(buf)

        if self.op == Operator.FOR:
            ret = '\t' * indent
            ret += f'affine.for %tmp = {self.inputs[0]} to {self.output}' + ' {\n'
            return ret
        if self.op == Operator.END_FOR:
            ret = '\t' * indent
            ret += '}\n'
            return ret

        if self.op == Operator.MOV:
            src = self.inputs[0]
            src_shape = self.compiler.shapes[self.inputs[0]]
            tgt = self.output
            tgt_shape = self.compiler.shapes[self.output]

            ret = ''
            if tgt in self.compiler.allocated:
                # free this buffer
                ret += '\t' * indent
                ret += f'memref.dealloc {tgt} : memref{self._mlir_shape(tgt_shape)}\n'

            ret += '\t' * indent
            ret += f'memref.copy {src}, {tgt} : memref{self._mlir_shape(src_shape)} to memref{self._mlir_shape(tgt_shape)}\n'

            return ret

        
        if self.op == Operator.FILL:
            data: float = float(self.inputs[0])
            ret = ""
            shape = self.compiler.shapes[self.output]

            if self.output not in self.compiler.allocated:
                self.compiler.allocated.add(self.output)
                ret += '\t' * indent
                ret += f'{self.output} = memref.alloc() : memref{self._mlir_shape(shape)}\n'

            for i in range(len(shape)):
                ret += '\t' * indent
                ret += f'affine.for %arg{i} = 0 to {shape[i]}' + ' {\n'
                indent += 1

            ret += '\t' * indent
            ret += f'%s = arith.constant {data} : f32\n'
            ret += '\t' * indent
            ret += f'memref.store %s, {self.output}{self._mlir_index(shape, shape)} : memref{self._mlir_shape(shape)}\n'

            for i in range(len(shape)):
                indent -= 1
                ret += '\t' * indent
                ret += '}\n'
            
            return ret

        ret = ""
        for node in self.inputs:
            if node not in self.compiler.allocated:
                ret += '\t' * indent
                shape = self.compiler.shapes[node]
                ret += f'{node} = memref.alloc() : memref{self._mlir_shape(shape)}\n'
                self.compiler.allocated.add(node)

        output_shape = self.compiler.shapes[self.output]
        if self.output not in self.compiler.allocated:
            ret += '\t' * indent
            ret += f'{self.output} = memref.alloc() : memref{self._mlir_shape(output_shape)}\n'
            self.compiler.allocated.add(self.output)

        if self.op in self._mlir_bin_ops:
            # %output = memref.alloc() : output.shape

            lh_shape = self.compiler.shapes[self.inputs[0]]
            rh_shape = self.compiler.shapes[self.inputs[1]]

            for i in range(len(output_shape)):
                ret += '\t' * indent
                # affine.for %arg0 = 0 to 4 {
                #     affine.for %arg1 = 0 to 3 {
                # ...
                ret += f'affine.for %arg{i} = 0 to {output_shape[i]} ' + '{\n'
                indent += 1

            # %s0 = memref.load %a[%arg0, 0] : memref<4x1xf32>
            ret += '\t' * indent
            ret += f'%s0 = memref.load {self.inputs[0]}' + self._mlir_index(output_shape, lh_shape)
            ret += ' : memref' + self._mlir_shape(lh_shape) + '\n'

            # %s1 = memref.load %a[0, %arg1] : memref<1x3xf32>
            ret += '\t' * indent
            ret += f'%s1 = memref.load {self.inputs[1]}' + self._mlir_index(output_shape, rh_shape)
            ret += ' : memref' + self._mlir_shape(rh_shape) + '\n'

            # %s2 = arith.addf %s0, %s1 : f32
            ret += '\t' * indent
            ret += f'%s2 = arith.{self._mlir_bin_ops[self.op]} %s0, %s1 : f32 \n'

            # memref.store %s2, %c[%arg0, %arg1] : memref<4x3xf32>
            ret += '\t' * indent
            ret += f'memref.store %s2, {self.output}' + self._mlir_index(output_shape, output_shape)
            ret += ' : memref' + self._mlir_shape(output_shape) + '\n'

            for i in range(len(output_shape)):
                indent -= 1
                ret += '\t' * indent
                ret += '}\n'

            # the result is in self.output.
        elif self.op in self._mlir_cmp_op:
            lh, rh = self.inputs[0], self.inputs[1]
            output = self.output
            output_shape = self.compiler.shapes[output]

            lh_shape = self.compiler.shapes[lh]
            rh_shape = self.compiler.shapes[rh]

            for i in range(len(output_shape)):
                ret += '\t' * indent
                # affine.for %arg0 = 0 to 4 {
                #     affine.for %arg1 = 0 to 3 {
                # ...
                ret += f'affine.for %arg{i} = 0 to {output_shape[i]} ' + '{\n'
                indent += 1

            # %s_zero = arith.constant 0.0 : f32
            ret += '\t' * indent + '%s_zero = arith.constant 0.0 : f32\n'
            # %s_one = arith.constant 1.0 : f32
            ret += '\t' * indent + '%s_one = arith.constant 1.0 : f32\n'

            # %s0 = memref.load %a[%arg0, 0] : memref<4x1xf32>
            ret += '\t' * indent
            ret += f'%s0 = memref.load {lh}' + self._mlir_index(output_shape, lh_shape)
            ret += ' : memref' + self._mlir_shape(lh_shape) + '\n'

            # %s1 = memref.load %a[0, %arg1] : memref<1x3xf32>
            ret += '\t' * indent
            ret += f'%s1 = memref.load {rh}' + self._mlir_index(output_shape, rh_shape)
            ret += ' : memref' + self._mlir_shape(rh_shape) + '\n'

            # %s2 = arith.cmpf %s0, [ogt|olt] %s1 : f32
            ret += '\t' * indent
            ret += f'%s2 = arith.cmpf {self._mlir_cmp_op[self.op]}, %s0, %s1 : f32\n'

            # %s3 = arith.select %s2, %s_zero, %s_one : f32
            ret += '\t' * indent
            ret += '%s3 = arith.select %s2, %s_zero, %s_one : f32\n'

            # memref.store %s3, %output[...] : memref<...>
            ret += '\t' * indent
            ret += f'memref.store %s3, {output}' + self._mlir_index(output_shape, output_shape)
            ret += ' : memref' + self._mlir_shape(output_shape) + '\n'

            for i in range(len(output_shape)):
                indent -= 1
                ret += '\t' * indent
                ret += '}\n'

        elif self.op == Operator.TRANSPOSE:
            a, b = tuple(self.compiler.shapes[self.inputs[0]])
            output_shape = [b, a]

            # affine.for %arg0 = 0 to %a {
            #    affine.for %arg1 = 0 to %b {
            #        %s = memref.load %input[%arg0, %arg1] : memref<axbxf32>
            #        memref.store %s, %output[%arg1, %arg0] : memref<bxaxf32>
            #    }
            # }

            # affine.for %arg0 = 0 to %a {
            ret += '\t' * indent
            ret += f'affine.for %arg0 = 0 to {a} ' + '{\n'
            indent += 1

            #    affine.for %arg1 = 0 to %b {
            ret += '\t' * indent
            ret += f'affine.for %arg1 = 0 to {b} ' + '{\n'
            indent += 1

            #        %s = memref.load %input[%arg0, %arg1] : memref<axbxf32>
            ret += '\t' * indent
            ret += f'%s = memref.load {self.inputs[0]}[%arg0, %arg1] : memref<{a}x{b}xf32>\n'

            #        memref.store %s, %output[%arg1, %arg0] : memref<bxaxf32>
            ret += '\t' * indent
            ret += f'memref.store %s, {self.output}[%arg1, %arg0] : memref<{b}x{a}xf32>\n'

            indent -= 1
            ret += '\t' * indent
            ret += '}\n'
            indent -= 1
            ret += '\t' * indent
            ret += '}\n'
        elif self.op == Operator.MATMUL:
            lh_shape = self.compiler.shapes[self.inputs[0]]
            rh_shape = self.compiler.shapes[self.inputs[1]]
            output_shape = self.compiler.shapes[self.output]

            a, b, c = lh_shape[0], lh_shape[1], rh_shape[1]
            lh, rh = self.inputs[0], self.inputs[1]
            lines = [
                f"affine.for %arg0 = 0 to {a}" + " {\n",
                f"\taffine.for %arg1 = 0 to {c}" + " {\n",
                f"\t\t%s5 = arith.constant 0.0 : f32\n",
                f"\t\tmemref.store %s5, {self.output}[%arg0, %arg1] : memref<{a}x{c}xf32>\n",
                f"\t\taffine.for %arg2 = 0 to {b}" + " {\n",
                f"\t\t\t%s0 = memref.load {self.output}[%arg0, %arg1] : memref<{a}x{c}xf32>\n",
                f"\t\t\t%s1 = memref.load {lh}[%arg0, %arg2] : memref<{a}x{b}xf32>\n",
                f"\t\t\t%s2 = memref.load {rh}[%arg2, %arg1] : memref<{b}x{c}xf32>\n",
                f"\t\t\t%s3 = arith.mulf %s1, %s2 : f32\n",
                f"\t\t\t%s4 = arith.addf %s0, %s3 : f32\n",
                f"\t\t\tmemref.store %s4, {self.output}[%arg0, %arg1] : memref<{a}x{c}xf32>\n",
                "\t\t}\n",
                "\t}\n",
                "}\n",]
            
            for line in lines:
                ret += '\t' * indent 
                ret += line

            del lines
        elif self.op == Operator.NEG:
            in_shape = self.compiler.shapes[self.inputs[0]]
            output_shape = in_shape
            in_var = self.inputs[0]
            out_var = self.output

            for i in range(len(output_shape)):
                ret += '\t' * indent
                # affine.for %arg0 = 0 to 4 {
                #     affine.for %arg1 = 0 to 3 {
                # ...
                ret += f'affine.for %arg{i} = 0 to {output_shape[i]} ' + '{\n'
                indent += 1

            # %s_zero = arith.constant 0.0 : f32
            ret += '\t' * indent
            ret += '%s_zero = arith.constant 0.0 : f32\n'

            # %s0 = memref.load %in_var[%arg0, ... %argn] : memref<...xf32>
            ret += '\t' * indent
            ret += f'%s0 = memref.load {in_var}' + self._mlir_index(output_shape, in_shape)
            ret += ' : memref' + self._mlir_shape(in_shape) + '\n'

            # %s1 = arith.subf %s_zero, %s0 : f32
            ret += '\t' * indent
            ret +=  f'%s1 = arith.subf %s_zero, %s0 : f32\n'

            # memref.store %s1, %out_var[%arg0, ... %argn] : memref<...xf32>
            ret += '\t' * indent
            ret += f'memref.store %s1, {out_var}' + self._mlir_index(output_shape, in_shape)
            ret += f' : {Mlir.typename(output_shape)}\n'

            for i in range(len(output_shape)):
                indent -= 1
                ret += '\t' * indent
                ret += "}\n"
        elif self.op == Operator.RELU:
            in_shape = self.compiler.shapes[self.inputs[0]]
            output_shape = in_shape
            in_var = self.inputs[0]
            out_var = self.output

            for i in range(len(output_shape)):
                ret += '\t' * indent
                # affine.for %arg0 = 0 to 4 {
                #     affine.for %arg1 = 0 to 3 {
                # ...
                ret += f'affine.for %arg{i} = 0 to {output_shape[i]} ' + '{\n'
                indent += 1

            # %s_zero = arith.constant 0.0 : f32
            ret += '\t' * indent
            ret += '%s_zero = arith.constant 0.0 : f32\n'

            # %s0 = memref.load %in_var[%arg0, ... %argn] : memref<...xf32>
            ret += '\t' * indent
            ret += f'%s0 = memref.load {in_var}' + self._mlir_index(output_shape, in_shape)
            ret += ' : memref' + self._mlir_shape(in_shape) + '\n'

            # %s1 = arith.maxf %s_zero, %s0 : f32
            ret += '\t' * indent
            ret +=  f'%s1 = arith.maxf %s_zero, %s0 : f32\n'

            # memref.store %s1, %out_var[%arg0, ... %argn] : memref<...xf32>
            ret += '\t' * indent
            ret += f'memref.store %s1, {out_var}' + self._mlir_index(output_shape, in_shape)
            ret += f' : {Mlir.typename(output_shape)}\n'

            for i in range(len(output_shape)):
                indent -= 1
                ret += '\t' * indent
                ret += "}\n"
        elif self.op == Operator.RESHAPE:
            ret += '\t' * indent
            ret += '// FIXME: reshape is not implemented. Coming soon\n'

            in_var = self.inputs[0]
            out_var = self.output
            in_shape = self.compiler.shapes[in_var]
            out_shape = self.compiler.shapes[out_var]

            #
            # set all elements of `output` to 0.0
            #
            for i in range(len(out_shape)):
                ret += '\t' * indent
                ret += f'affine.for %arg{i} = 0 to {out_shape[i]}' + ' {\n'
                indent += 1
            
            # %s_zero = arithm.constant 0.0 : f32
            ret += '\t' * indent
            ret += f'%s_zero = arith.constant 0.0 : f32\n'
            # memref.store %s_zero, %output[%arg0, ..., %argN] : memref<*xf32>
            ret += '\t' * indent
            ret += f'memref.store %s_zero, {out_var}{self._mlir_index(out_shape, out_shape)}'
            ret += f" : {Mlir.typename(out_shape)}\n"

            for i in range(len(out_shape)):
                indent -= 1
                ret += '\t' * indent
                ret += '}\n'

            # 
            # add
            #
            for i in range(len(in_shape)):
                ret += '\t' * indent
                ret += f'affine.for %arg{i} = 0 to {in_shape[i]}' + ' {\n'
                indent += 1

            # %s0 = memref.load %output[%arg0, ..., %argn] : memref<...>
            ret += '\t' * indent
            ret += f'%s0 = memref.load {out_var}{self._mlir_index(in_shape, out_shape)} : {Mlir.typename(out_shape)}\n'
            # %s1 = memref.load %input[%arg0, ..., %argn] : memref<...>
            ret += '\t' * indent
            ret += f'%s1 = memref.load {in_var}{self._mlir_index(in_shape, in_shape)} : {Mlir.typename(in_shape)}\n'

            # %s2 = arith.addf %s0, %s1 : f32
            ret += '\t' * indent
            ret += '%s2 = arith.addf %s0, %s1 : f32\n'
            # memref.store %s2, %output[%arg0, ..., %argN] : memref<*xf32>
            ret += '\t' * indent
            ret += f'memref.store %s2, {out_var}{self._mlir_index(out_shape, out_shape)}'
            ret += f" : {Mlir.typename(out_shape)}\n"

            for i in range(len(in_shape)):
                indent -= 1
                ret += '\t' * indent
                ret += '}\n'
        elif self.op == Operator.RETURN:
            ret += '\t' * indent

            # return %ret : memref<2x2xf32>
            var = self.compiler.ret
            ret += f"return {Mlir.retstmt(var, self.compiler.retshape())}\n"
        elif self.op == Operator.PRINT:
            one = self._mlir_next_const()
            ret += '\t' * indent
            ret += f'{one} = arith.constant 1 : index\n'

            indices = []
            for i in range(len(output_shape)):
                # create a constant for each dimension
                ret += '\t' * indent
                constant = self._mlir_next_const()
                ret +=  f'{constant} = arith.constant {output_shape[i]} : index\n'
                indices.append(constant)

            # llvm.call @json_list_start() : () -> ()
            ret += '\t' * indent
            ret += f'llvm.call @json_list_start() : () -> ()' + '\n'

            for i in range(len(output_shape) - 1):
                ret += '\t' * indent
                # use scf.for instead of affine.for
                ret += f'scf.for %arg{i} = %zero to {indices[i]} step {one}' + ' {\n'
                indent += 1

                cond = self._mlir_next_const()
                ret += '\t' * indent
                ret += f"{cond} = arith.cmpi ne, %zero, %arg{i} : index\n"

                # if %cond { json_list_sep(); }
                ret += '\t' * indent
                ret += f'scf.if {cond} ' + " {\n"
                indent += 1
                ret += '\t' * indent
                ret += f'llvm.call @json_list_sep() : () -> ()' + '\n'
                indent -= 1
                ret += '}\n'

                # llvm.call @json_list_start() : () -> ()
                ret += '\t' * indent
                ret += f'llvm.call @json_list_start() : () -> ()' + '\n'

            i = len(output_shape) - 1

            # scf.for %argi = %zero to %one step %one { json_f32();}
            ret += '\t' * indent
            ret += f'scf.for %arg{i} = %zero to {one} step {one}' + ' {\n'
            indent += 1
            ret += '\t' * indent
            ret += f'%s0 = memref.load {self.output}{self._mlir_index(output_shape, output_shape)} : {Mlir.typename(output_shape)}\n'
            ret += '\t' * indent
            ret +=  f'llvm.call @json_f32(%s0) : (f32) -> ()' + '\n'
            indent -= 1
            ret += '\t' * indent
            ret += "}\n"

            # scf.for %argi = %zero to %one step %one { json_list_sep(); json_f32();}
            ret += '\t' * indent
            ret += f'scf.for %arg{i} = {one} to {indices[i]} step {one}' + ' {\n'
            indent += 1
            ret += '\t' * indent
            ret += 'llvm.call @json_list_sep() : () -> ()\n'
            ret += '\t' * indent
            ret += f'%s0 = memref.load {self.output}{self._mlir_index(output_shape, output_shape)} : {Mlir.typename(output_shape)}\n'
            ret += '\t' * indent
            ret +=  f'llvm.call @json_f32(%s0) : (f32) -> ()\n'
            indent -= 1
            ret += '\t' * indent
            ret += "}\n"

            for i in range(len(output_shape) - 1):
                # llvm.call @json_list_end() : () -> ()
                ret += '\t' * indent
                ret += f'llvm.call @json_list_end() : () -> ()' + '\n'

                indent -= 1
                ret += '\t' * indent
                ret += '}\n'

            # llvm.call @json_list_end() : () -> ()
            ret += '\t' * indent
            ret +=  "llvm.call @json_list_end() : () -> ()\n"

            # llvm.call @new_line() : () -> ()
            ret += '\t' * indent
            ret += 'llvm.call @new_line() : () -> ()' + '\n'
        else:
            raise NotImplementedError(f'Instruction {self.op.value} not implemented')

        return ret

class Variable():
    count = 0
    def __init__(self, shape, dtype: DType=DType.FLOAT32):
        self.shape: list[int] = shape
        self.dtype = dtype
        self.name = f'%v{Variable.count}'
        Variable.count += 1

        self.device = None
        # self.data: list | np.ndarray = None
        self.data: list = None

        if isinstance(CompilerContext.compiler, Compiler):
            CompilerContext.compiler.shapes[self.name] = self.shape

    @staticmethod
    def fill(shape: list[int], data: float, device=None):
        ret = Variable(shape)

        compiler = CompilerContext.compiler
        if isinstance(compiler, Compiler):
            compiler.add_insn(Operator.FILL, ret.name, data)

        return ret
        
    @staticmethod 
    def zeros(shape: list[int], device=None):
        return Variable.fill(shape, 0.0)

    @staticmethod 
    def ones(shape: list[int], device=None):
        return Variable.fill(shape, 1.0)

    @staticmethod
    def random_seed(seed: int = 42):
        """
        :param seed: The seed to use for random number generation.
        """
        np.random.seed(seed)    

    @staticmethod
    def uniform(shape: list[int], xmin: float = 0.0, 
                xmax: float = 1.0, device=None):
        data = np.random.uniform(xmin, xmax, tuple(shape)).tolist()
        return Variable.from_list(shape, data)

    @staticmethod 
    def _check_shape(shape: list[int], data: list, idx: int = 0):
        if idx >= len(shape):
            return
        if len(data) != shape[idx]:
            raise ValueError(f'Incompatible shape: {shape} and {data}')

        for i in range(len(data)):
            Variable._check_shape(shape, data[i], idx + 1)

    @staticmethod 
    def from_list(shape: list[int], data: list, device=None):
        Variable._check_shape(shape, data)
        ret = Variable(shape)
        ret.data = data

        compiler = CompilerContext.compiler
        if isinstance(compiler, Compiler):
            compiler.add_insn(Operator.INIT, ret.name, data)
        return ret

    def __repr__(self):
        if CompilerContext.compiler:
            CompilerContext.compiler.add_insn(Operator.PRINT, self.name)
            # does not print anything
            return ''

        d = {
            "shape":  self.shape,
            "dtype":  self.dtype.value}

        return json.dumps(d)

    def __str__(self):
        if CompilerContext.compiler:
            CompilerContext.compiler.add_insn(Operator.PRINT, self.name)
            # does not print anything
            return ''
        return self.name

    def __add__(self, other):
        assert isinstance(other, Variable), f'Incompatible types: {type(self)} and {type(other)}'
        ret = Variable(_broadcast(self.shape, other.shape))

        if isinstance(CompilerContext.compiler, Compiler):
            CompilerContext.compiler.add_insn(Operator.ADD, ret.name, self.name, other.name)

        return ret

    def __sub__(self, other):
        assert isinstance(other, Variable), f'Incompatible types: {type(self)} and {type(other)}'
        ret = Variable(_broadcast(self.shape, other.shape))

        if isinstance(CompilerContext.compiler, Compiler):
            CompilerContext.compiler.add_insn(Operator.SUB, ret.name, self.name, other.name)

        return ret

    def __mul__(self, other):
        assert isinstance(other, Variable), f'Incompatible types: {type(self)} and {type(other)}'
        ret = Variable(_broadcast(self.shape, other.shape))

        if isinstance(CompilerContext.compiler, Compiler):
            CompilerContext.compiler.add_insn(Operator.MUL, ret.name, self.name, other.name)

        return ret

    def __neg__(self):
        ret = Variable(self.shape)

        if isinstance(CompilerContext.compiler, Compiler):
            CompilerContext.compiler.add_insn(Operator.NEG, ret.name, self.name)

        return ret

    @staticmethod 
    def matmul(lh, rh):
        if len(lh.shape) != 2 or len(rh.shape) != 2:
            raise ValueError('Both arguments must be 2D')
        if lh.shape[1] != rh.shape[0]:
            raise ValueError('Incompatible shapes %s and %s for matmul' % (lh.shape, rh.shape))

        ret = Variable(lh.shape[:-1] + rh.shape[1:])
        if isinstance(CompilerContext.compiler, Compiler):
            CompilerContext.compiler.add_insn(Operator.MATMUL, ret.name, lh.name, rh.name)

        return ret

    def __lt__(self, other):
        assert isinstance(other, Variable), f'Incompatible types: {type(self)} and {type(other)}'
        ret = Variable(_broadcast(self.shape, other.shape))

        if isinstance(CompilerContext.compiler, Compiler):
            CompilerContext.compiler.add_insn(Operator.LT, ret.name, self.name, other.name)

        return ret

    def __gt__(self, other):
        assert isinstance(other, Variable), f'Incompatible types: {type(self)} and {type(other)}'
        ret = Variable(_broadcast(self.shape, other.shape))

        if isinstance(CompilerContext.compiler, Compiler):
            CompilerContext.compiler.add_insn(Operator.GT, ret.name, self.name, other.name)

        return ret

    @staticmethod
    def dot(lh, rh):
        return Variable.matmul(lh, rh)

    @staticmethod 
    def transpose(x):
        if len(x.shape) != 2:
            raise ValueError('Argument must be 2D')
        ret = Variable([x.shape[1], x.shape[0]])

        if isinstance(CompilerContext.compiler, Compiler):
            CompilerContext.compiler.add_insn(Operator.TRANSPOSE, ret.name, x.name)

        return ret

    @staticmethod
    def relu(x):
        ret = Variable(x.shape)
        if isinstance(CompilerContext.compiler, Compiler):
            CompilerContext.compiler.add_insn(Operator.RELU, ret.name, x.name)
        return ret

    @staticmethod
    def grad_reshape(x, shape):
        ret = Variable(shape)
        if isinstance(CompilerContext.compiler, Compiler):
            CompilerContext.compiler.add_insn(Operator.RESHAPE, ret.name, x.name)
        return ret

    def memory(self) -> int:
        """
        :return: the memory size of the variable in bytes
        """

        size = 1
        for i in self.shape:
            size *= i

        return size * DType.sizeof(self.dtype)
