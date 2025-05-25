import json
from enum import Enum
import re

from common import Operator

class CompilerContext:
    compiling: bool = False
    compiler = None

class Compiler:

    def __init__(self):
        self.vars = []
        self.shapes = {}
        self.instructions = []

        self.allocated = set()
        self.globl_var = set()

        self.args = set()
        self.ret = None

        self.prev = False
        self.prev_compiler = None

    def __enter__(self):

        self.prev = CompilerContext.compiling
        self.prev_compiler = CompilerContext.compiler
        # self.vars = []
        # self.shapes = {}
        # self.instructions = []

        CompilerContext.compiling = True
        CompilerContext.compiler = self # type: ignore

        return self

    def add_insn(self, op: Operator, output, *inputs):
        if output not in self.vars:
            self.vars.append(output)
        for input in inputs:
            if input not in self.vars:
                self.vars.append(input)
        
        if op == Operator.RETURN:
            self.ret = output
        self.instructions.append((op.value, output, inputs))

    def add_globl_var(self, name: str):
        if name not in self.shapes:
            raise RuntimeError(f'Global variable {name} not found')

        self.globl_var.add(name)

    def add_arg(self, name: str):
        if name not in self.shapes:
            raise RuntimeError(f'Argument {name} not found')

        self.args.add(name)

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
        indices = ['0' if input_shape[i] == 1 else f'%arg{i}' for i in range(len(output_shape))]
        return '[' + ', '.join(indices) + ']'

    def generate_mlir(self, indent: int = 2) -> str:
        if self.compiler is None:
            raise ValueError('Compiler not set')

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
            if len(lh_shape) < len(output_shape):
                lh_shape_ext = [1 for _ in range(len(output_shape) - len(lh_shape))] + lh_shape
            else:
                lh_shape_ext = lh_shape

            rh_shape = self.compiler.shapes[self.inputs[1]]
            if len(rh_shape) < len(output_shape):
                rh_shape_ext = [1 for _ in range(len(output_shape) - len(rh_shape))] + rh_shape
            else:
                rh_shape_ext = rh_shape

            for i in range(len(output_shape)):
                ret += '\t' * indent
                # affine.for %arg0 = 0 to 4 {
                #     affine.for %arg1 = 0 to 3 {
                # ...
                ret += f'affine.for arg{i} = 0 to {output_shape[i]} ' + '{\n'
                indent += 1

            # %s0 = memref.load %a[%arg0, 0] : memref<4x1xf32>
            ret += '\t' * indent
            ret += f'%s0 = memref.load {self.inputs[0]}' + self._mlir_index(output_shape, lh_shape_ext)
            ret += ' : memref' + self._mlir_shape(lh_shape) + '\n'

            # %s1 = memref.load %a[0, %arg1] : memref<1x3xf32>
            ret += '\t' * indent
            ret += f'%s1 = memref.load {self.inputs[1]}' + self._mlir_index(output_shape, rh_shape_ext)
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
            ret += f'%s = memref.load %input[%arg0, %arg1] : memref<{a}x{b}f32>\n'

            #        memref.store %s, %output[%arg1, %arg0] : memref<bxaxf32>
            ret += '\t' * indent
            ret += f'memref.store %s, %output[%arg1, %arg0] : memref<{b}x{a}f32>\n'

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
        elif self.op == Operator.RETURN:
            ret += '\t' * indent

            # return %ret : memref<2x2xf32>
            var = self.compiler.ret
            shape = self.compiler.shapes[var]
            ret += f"return {var} : memref{self._mlir_shape(shape)}\n"
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

        if isinstance(CompilerContext.compiler, Compiler):
            CompilerContext.compiler.shapes[self.name] = self.shape
        
    @staticmethod 
    def zeros(shape: list[int], device=None):
        return Variable(shape)

    @staticmethod 
    def ones(shape: list[int], device=None):
        return Variable(shape)

    def __str__(self):
        return self.name

    def __repr__(self):
        d = {
            "shape":  self.shape,
            "dtype":  self.dtype.value}

        return json.dumps(d)

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

    @staticmethod
    def grad_reshape(x, shape):
        ret = Variable(shape)
        if isinstance(CompilerContext.compiler, Compiler):
            CompilerContext.compiler.add_insn(Operator.RESHAPE, ret.name, x.name)

    def memory(self) -> int:
        """
        :return: the memory size of the variable in bytes
        """

        size = 1
        for i in self.shape:
            size *= i

        return size * DType.sizeof(self.dtype)
