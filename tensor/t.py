import tensor_module
from tensor_module import Tensor, Device
import cffi

_: int = 0;

print(int(Device.CPU))
print(Device.CPU == Device.CUDA)

t = Tensor((2, 2), Device.CPU)
print(t.shape)

# del t
t = Tensor.ones((2, 2), Device.CPU)
print(t[0], t.device)
print(t.to_list())

t1 = t + t
print(t1[0])

t1 += Tensor.zeros((2, 2), Device.CPU)
print(t1[0])

t1.to(Device.CUDA)

import numpy as np

#FIXME: will crash
a = np.array(t1)
