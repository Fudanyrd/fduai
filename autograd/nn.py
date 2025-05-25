from .node import DataNode

from compiler import Compiler, Variable, CompilerContext
from common import Operator

class nn():
    def __init__(self):
        self.params: list[DataNode] = []

    def parameters(self) -> list[DataNode]:
        return self.params

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, dout: DataNode):
        raise NotImplementedError

class linear(nn):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.w = DataNode.tensor([in_features, out_features])
        self.b = DataNode.tensor([1, out_features])
        self.params = [self.w, self.b]

    def parameters(self) -> list[DataNode]:
        return self.params

    def forward(self, *inputs):
        x = inputs[0]
        return DataNode.matmul(x, self.w) + self.b

def compile_nn(nn: nn, input_dims) -> Compiler:
    if CompilerContext.compiler:
        compiler = CompilerContext.compiler
        inputs = []
        for dim in input_dims:
            inputs.append(DataNode.tensor(dim))
        
        for param in nn.parameters():
            compiler.add_globl_var(param.tensor.name)
            compiler.allocated.add(param.tensor.name)

        for ar in inputs:
            compiler.add_arg(ar.tensor.name)
            compiler.allocated.add(ar.tensor.name)

        ret = nn.forward(*inputs)

        compiler.add_insn(Operator.RETURN, ret.tensor.name)
        return compiler

    else:
        raise NotImplementedError
