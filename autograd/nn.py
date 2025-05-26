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

    def predict(self, *inputs):
        return self.forward(*inputs)

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

class relu(nn):
    def __init__(self):
        super().__init__()

    def forward(self, *inputs):
        assert len(inputs) == 1, 'Incorrect number of inputs'
        x = inputs[0]
        return DataNode.relu(x)

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

def compile_backward(nn: nn, loss_fn, y_shape, input_dims) -> Compiler:
    if CompilerContext.compiler:
        compiler = CompilerContext.compiler

        inputs = []
        for dim in input_dims:
            inputs.append(DataNode.tensor(dim))

        for param in nn.parameters():
            compiler.add_globl_var(param.tensor.name)
            compiler.allocated.add(param.tensor.name)

        y = DataNode.tensor(y_shape)
        compiler.add_arg(y.tensor.name)
        compiler.allocated.add(y.tensor.name)

        for arg in inputs:
            compiler.add_arg(arg.tensor.name)
            compiler.allocated.add(arg.tensor.name)

        y_pred = nn.forward(*inputs)
        loss = loss_fn(y_pred, y)
        loss.backward()

        ret = []
        for param in nn.parameters():
            grad = param.grad
            assert isinstance(grad, Variable)
            ret.append(param.grad.name)

        compiler.add_ret_stmt(ret)
        return compiler
    else:
        raise NotImplementedError  
