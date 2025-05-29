from fduai.tensor import Tensor, Device
from fduai.common import Operator
from fduai.compiler import CompilerContext, Variable

class DataNode():
    topological_order = []
    def __init__(self, tensor: Tensor, requires_grad: bool = True):
        self.tensor = tensor
        self.requires_grad = requires_grad
        self.grad = None

        self.inputs = []
        self.op = Operator.NONE
        DataNode.topological_order.append(self)

    @staticmethod
    def tensor(shape: list[int], requires_grad: bool = True, device=Device.CPU):
        t = Variable(shape) if CompilerContext.compiling else Tensor.zeros(shape, device)
        return DataNode(t, requires_grad)

    @staticmethod 
    def zeros(shape: list[int], requires_grad: bool = True, device=Device.CPU):
        t = Variable.zeros(shape) if CompilerContext.compiling else Tensor.zeros(shape, device)
        return DataNode(t, requires_grad)

    @staticmethod 
    def ones(shape: list[int], requires_grad: bool = True, device=Device.CPU):
        t = Variable.ones(shape) if CompilerContext.compiling else Tensor.ones(shape, device)
        return DataNode(t, requires_grad)

    @staticmethod
    def uniform(shape: list[int], xmin: float = 0.0, xmax: float = 1.0,
                requires_grad: bool = True):
        if not CompilerContext.compiling:
            raise ValueError("uniform is not implemented for Tensor.")
        t = Variable.uniform(shape, xmin, xmax)
        return DataNode(t, requires_grad)

    @staticmethod 
    def from_list(shape: list[int], data: list, requires_grad: bool = True,
                  device=Device.CPU):
        t = Variable.from_list(shape, data) if CompilerContext.compiling \
            else Tensor.from_list(data)
        return DataNode(t, requires_grad)

    def shape(self):
        return self.tensor.shape

    def __add__(self, other):
        t = self.tensor + other.tensor
        ret = DataNode(t, requires_grad=False)

        ret.op = Operator.ADD
        ret.inputs = [self, other]
        ret.requires_grad = self.requires_grad or other.requires_grad

        return ret
    
    def __sub__(self, other):
        t = self.tensor - other.tensor
        ret = DataNode(t, requires_grad=False)

        ret.op = Operator.SUB
        ret.inputs = [self, other]
        ret.requires_grad = self.requires_grad or other.requires_grad

        return ret
    
    def __mul__(self, other):
        t = self.tensor * other.tensor
        ret = DataNode(t, requires_grad=False)

        ret.op = Operator.MUL
        ret.inputs = [self, other]
        ret.requires_grad = self.requires_grad or other.requires_grad

        return ret

    def __neg__(self):
        ret = DataNode(-self.tensor, requires_grad=self.requires_grad)

        ret.op = Operator.NEG
        ret.inputs = [self]

        return ret
    
    @staticmethod 
    def matmul(lh, rh):
        t = type(lh.tensor).dot(lh.tensor, rh.tensor)
        ret = DataNode(t, requires_grad=False)

        ret.op = Operator.MATMUL
        ret.inputs = [lh, rh]
        ret.requires_grad = lh.requires_grad or rh.requires_grad

        return ret

    @staticmethod
    def zero_grad():
        for node in DataNode.topological_order:
            node.grad = None

    def _add_grad(self, grad):
        TT = type(self.tensor)
        if grad.shape != self.tensor.shape:
            grad = TT.grad_reshape(grad, self.shape())
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

    def backward(self, grad=None):
        no_grad = grad is None
        if CompilerContext.compiling:
            grad = Variable.ones(self.shape()) if grad is None else grad
        else:
            grad = Tensor.ones(self.shape(), self.tensor.device) if grad is None else grad
        
        TT = type(self.tensor)

        if self.op == Operator.NONE:
            self.grad = grad
        elif self.op == Operator.ADD:
            self.inputs[0]._add_grad(grad)
            self.inputs[1]._add_grad(grad)

            lh, rh = self.inputs[0], self.inputs[1]
            if DataNode.topological_order.index(lh) < DataNode.topological_order.index(rh):
                rh.backward(rh.grad)
                lh.backward(lh.grad)
            else:
                lh.backward(lh.grad)
                rh.backward(rh.grad)
        elif self.op == Operator.SUB:
            self.inputs[0]._add_grad(grad)
            self.inputs[1]._add_grad(-grad)

            lh, rh = self.inputs[0], self.inputs[1]
            if DataNode.topological_order.index(lh) < DataNode.topological_order.index(rh):
                rh.backward(rh.grad)
                lh.backward(lh.grad)
            else:
                lh.backward(lh.grad)
                rh.backward(rh.grad)
        elif self.op == Operator.MUL:
            self.inputs[0]._add_grad(grad * self.inputs[1].tensor)
            self.inputs[1]._add_grad(grad * self.inputs[0].tensor)

            lh, rh = self.inputs[0], self.inputs[1]
            if DataNode.topological_order.index(lh) < DataNode.topological_order.index(rh):
                rh.backward(rh.grad)
                lh.backward(lh.grad)
            else:
                lh.backward(lh.grad)
                rh.backward(rh.grad)
        elif self.op == Operator.MATMUL:
            self.inputs[0]._add_grad(TT.dot(grad, TT.transpose(self.inputs[1].tensor)))
            self.inputs[1]._add_grad(TT.dot(TT.transpose(self.inputs[0].tensor), grad))

            lh, rh = self.inputs[0], self.inputs[1]
            if DataNode.topological_order.index(lh) < DataNode.topological_order.index(rh):
                rh.backward(rh.grad)
                lh.backward(lh.grad)
            else:
                lh.backward(lh.grad)
                rh.backward(rh.grad)
        elif self.op == Operator.NEG:
            src = self.inputs[0]
            src._add_grad(-grad)
            src.backward(src.grad)
        elif self.op == Operator.RELU:
            src = self.inputs[0]
            zero_scalar = TT.zeros([1, ], src.tensor.device)
            if not no_grad:
                grad *= (zero_scalar < src.tensor)
                src._add_grad(grad)
            else:
                src._add_grad(zero_scalar < src.tensor)
            src.backward(src.grad)

            del zero_scalar
        else:
            raise NotImplementedError()

    def compile(self):
        raise NotImplementedError()

    @staticmethod 
    def relu(x):
        TT = type(x.tensor)
        ret = DataNode(TT.relu(x.tensor), requires_grad=x.requires_grad)

        ret.op = Operator.RELU
        ret.inputs = [x]

        return ret


if __name__ == '__main__':
    # should print [2], [2], [2]
    a = DataNode(Tensor.from_list([1]))
    b = DataNode(Tensor.from_list([1]))
    c = DataNode(Tensor.from_list([1]))
    x = a + b
    y = b + c 
    z = c + a
    ret = x + y + z
    ret.backward()
    print(c.grad.to_list(), a.grad.to_list(), b.grad.to_list())
    DataNode.zero_grad()

    # should print [1], [2], [3]
    ret = a + b + b + c + c + c
    ret.backward()
    print(c.grad.to_list(), a.grad.to_list(), b.grad.to_list())
    DataNode.zero_grad()

    # should print None, [1], None
    a.backward()
    print(c.grad, a.grad.to_list(), b.grad)
    DataNode.zero_grad()

    x = a - b
    x.backward()
    print(a.grad.to_list(), b.grad.to_list())
    DataNode.zero_grad()

    x = a - b 
    y = x * x
    y.backward()
    print(a.grad.to_list(), b.grad.to_list())
    DataNode.zero_grad()
