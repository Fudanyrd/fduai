from tensor_module import Tensor, Device
import numpy as np
import sys

# ===========================================
# relu test
# ===========================================
a = Tensor.from_list([1,-2,3,-4])
a.to(Device.CUDA)
b = Tensor.relu(a)
b.to(Device.CPU)
l = np.array(b)
assert np.allclose(b, np.array([1., 0., 3., 0.]))
del l 
del a 
del b

# ===========================================
# lt test
# ===========================================
a = Tensor.from_list([1,-2,3,-4])
a.to(Device.CUDA)
b = Tensor.from_list([0,2,1,4])
b.to(Device.CUDA)
c = (a < b)
c.to(Device.CPU)
l = np.array(c)
assert l.shape == (4, )
assert np.allclose(c, np.array([0.0, 1.0, 0.0, 1.0]))
del l 
del a 
del b
del c

# ===========================================
# tensor-scalar op test
# ===========================================
a = Tensor.from_list([1, 2, 3])
a.to(Device.CUDA)
b = -1.0
c = a + b
c.to(Device.CPU)
l = np.array(c)
assert l.shape == (3, )
assert np.allclose(l, np.array([0., 1., 2.]))
del l 
del a 
del b
del c

# ===========================================
# broadcast test
# ===========================================
a = Tensor.from_list([[1, 2], [3, 4]])
b = Tensor.from_list([-1, 1])
a.to(Device.CUDA)
b.to(Device.CUDA)
c = a + b
c.to(Device.CPU)
l = np.array(c)
assert l.shape == (2, 2)
assert np.allclose(l, np.array([[0., 3.], [2., 5.]]))
del a
del b 
del c
del l

print('exit 0', file=sys.stderr)
