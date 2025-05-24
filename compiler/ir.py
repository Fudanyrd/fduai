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

        self.instructions.append((op.value, output, inputs))

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
    def __init__(self, op: Operator, output: str, inputs: list[str]):
        self.op = op
        self.output = output
        self.inputs = inputs

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
