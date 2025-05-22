from tensor_module import Tensor, Device
from op import Operator

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

        return ret

    @staticmethod
    def zero_grad():
        for node in DataNode.topological_order:
            node.grad = None

    def _add_grad(self, grad):
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
