from tensor import Tensor, Device
from .op import Operator

class DataNode():
    topological_order = []
    def __init__(self, tensor: Tensor, requires_grad: bool = True):
        self.tensor = tensor
        self.requires_grad = requires_grad
        self.grad = None

        self.inputs = []
        self.op = Operator.NONE
        DataNode.topological_order.append(self)

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
        t = Tensor.dot(lh.tensor, rh.tensor)
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
        if grad.shape != self.tensor.shape:
            grad = Tensor.grad_reshape(grad, self.shape())
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

    def backward(self, grad=None):
        grad = Tensor.ones(self.shape(), self.tensor.device) if grad is None else grad

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
            self.inputs[0]._add_grad(Tensor.dot(grad, Tensor.transpose(self.inputs[1].tensor)))
            self.inputs[1]._add_grad(Tensor.dot(Tensor.transpose(self.inputs[0].tensor), grad))

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
        else:
            raise NotImplementedError()

    def compile(self):
        raise NotImplementedError()


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
