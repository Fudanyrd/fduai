#!/usr/bin/python3
import numpy as np
import tensor_module
from tensor_module import Tensor, Device
import cffi


def separator(title):
    """Print a separator with title"""
    print("\n" + "=" * 50)
    print(f"  {title}")
    print("=" * 50)


# ===========================================
# Basic tensor creation and properties tests
# ===========================================
separator("DEVICE ENUMERATION")
print(f"Device.CPU value: {int(Device.CPU)}")
print(f"Device.CUDA value: {int(Device.CUDA)}")
print(f"CPU == CUDA: {Device.CPU == Device.CUDA}")

separator("BASIC TENSOR CREATION")
# Empty tensor
t_empty = Tensor((2, 2), Device.CPU)
print(f"Empty tensor shape: {t_empty.shape}")
print(f"Empty tensor num_elements: {t_empty.num_elements}")
print(f"Empty tensor device: {t_empty.device}")

# Ones tensor
t_ones = Tensor.ones((2, 3), Device.CPU)
print(f"Ones tensor shape: {t_ones.shape}")
print(f"Ones tensor device: {t_ones.device}")
print(f"Ones tensor values: {t_ones.to_list()}")

# Zeros tensor
t_zeros = Tensor.zeros((3, 2), Device.CPU)
print(f"Zeros tensor shape: {t_zeros.shape}")
print(f"Zeros tensor values: {t_zeros.to_list()}")

# ===========================================
# Element access tests
# ===========================================
separator("ELEMENT ACCESS")
t = Tensor.ones((2, 2), Device.CPU)
print(f"Original tensor: {t.to_list()}")
print(f"Element at index 0: {t[0]}")
print(f"Element at index 3: {t[3]}")

# Set elements
t[0] = 5.0
t[3] = 10.0
print(f"After setting elements: {t.to_list()}")

# ===========================================
# Tensor operations tests
# ===========================================
separator("TENSOR ADDITION")
a = Tensor.ones((2, 2), Device.CPU)
b = Tensor.ones((2, 2), Device.CPU)
c = a + b
print(f"a: {a.to_list()}")
print(f"b: {b.to_list()}")
print(f"a + b: {c.to_list()}")

# ===========================================
# NumPy Conversion tests
# ===========================================
separator("NUMPY CONVERSION")

# Test NumPy to Tensor conversion
try:
    # Create a NumPy array
    np_a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    np_b = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=np.float32)

    print(f"NumPy array A:\n{np_a}")
    print(f"NumPy array B:\n{np_b}")

    # Convert to Tensor
    tensor_a = Tensor.from_numpy(np_a)
    tensor_b = Tensor.from_numpy(np_b)

    print(f"Tensor A from NumPy: {tensor_a.to_list()}")
    print(f"Tensor B from NumPy: {tensor_b.to_list()}")

    # Convert back to NumPy
    np_a_back = np.array(tensor_a)
    np_b_back = np.array(tensor_b)

    print(f"NumPy array A (round-trip):\n{np_a_back}")
    print(f"NumPy array B (round-trip):\n{np_b_back}")

    # Test with assertions
    assert np.array_equal(
        np_a, np_a_back), "Round-trip conversion failed for array A"
    assert np.array_equal(
        np_b, np_b_back), "Round-trip conversion failed for array B"
    print("✓ Round-trip NumPy conversion tests passed!")

except Exception as e:
    print(f"NumPy conversion error: {e}")

# ===========================================
# Dot Product with NumPy verification - CPU
# ===========================================
separator("DOT PRODUCT WITH NUMPY VERIFICATION - CPU")

try:
    # Create NumPy arrays
    np_a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    np_b = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=np.float32)

    # Compute dot product in NumPy
    np_result = np.matmul(np_a, np_b)
    print(f"NumPy dot product result:\n{np_result}")

    # Convert to Tensors
    tensor_a = Tensor.from_numpy(np_a)
    tensor_b = Tensor.from_numpy(np_b)

    # Compute dot product using our library
    tensor_result = Tensor.dot(tensor_a, tensor_b)
    print(f"Tensor dot product result: {tensor_result.to_list()}")

    # Convert result back to NumPy
    result_np = np.array(tensor_result)
    print(f"Tensor result converted to NumPy:\n{result_np}")

    # Verify the results match
    assert np.allclose(np_result, result_np,
                       atol=1e-5), "Dot product results don't match"
    print("✓ Dot product test passed!")

    # Test with another set of matrices
    np_x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    np_y = np.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)

    np_z_expected = np.matmul(np_x, np_y)

    tensor_x = Tensor.from_numpy(np_x)
    tensor_y = Tensor.from_numpy(np_y)
    tensor_z = Tensor.dot(tensor_x, tensor_y)

    np_z_actual = np.array(tensor_z)

    print(f"Second test - Expected:\n{np_z_expected}")
    print(f"Second test - Actual:\n{np_z_actual}")

    assert np.allclose(np_z_expected, np_z_actual,
                       atol=1e-5), "Second dot product test failed"
    print("✓ Second dot product test passed!")

except Exception as e:
    print(f"Dot product verification error: {e}")

# ===========================================
# Dot Product with NumPy verification - CUDA
# ===========================================
separator("DOT PRODUCT WITH NUMPY VERIFICATION - CUDA")

try:
    # Create NumPy arrays
    np_a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    np_b = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=np.float32)

    # Compute dot product in NumPy
    np_result = np.matmul(np_a, np_b)
    print(f"NumPy dot product result:\n{np_result}")

    # Convert to Tensors on CUDA
    tensor_a = Tensor.from_numpy(np_a)
    tensor_b = Tensor.from_numpy(np_b)

    # Transfer to CUDA
    tensor_a.to(Device.CUDA)
    tensor_b.to(Device.CUDA)

    print(f"Tensor A device: {tensor_a.device}")
    print(f"Tensor B device: {tensor_b.device}")

    # Compute dot product using our library on CUDA
    tensor_result = Tensor.dot(tensor_a, tensor_b)
    print(f"CUDA tensor result device: {tensor_result.device}")

    # Transfer result back to CPU for verification
    tensor_result.to(Device.CPU)

    # Convert result back to NumPy
    result_np = np.array(tensor_result)
    print(f"CUDA tensor result converted to NumPy:\n{result_np}")

    # Verify the results match
    assert np.allclose(np_result, result_np,
                       atol=1e-5), "CUDA dot product results don't match"
    print("✓ CUDA dot product test passed!")

    # Test with larger matrices
    np_x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    np_y = np.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)

    np_z_expected = np.matmul(np_x, np_y)

    tensor_x = Tensor.from_numpy(np_x)
    tensor_y = Tensor.from_numpy(np_y)

    tensor_x.to(Device.CUDA)
    tensor_y.to(Device.CUDA)

    tensor_z = Tensor.dot(tensor_x, tensor_y)
    tensor_z.to(Device.CPU)

    np_z_actual = np.array(tensor_z)

    print(f"Second CUDA test - Expected:\n{np_z_expected}")
    print(f"Second CUDA test - Actual:\n{np_z_actual}")

    assert np.allclose(np_z_expected, np_z_actual,
                       atol=1e-5), "Second CUDA dot product test failed"
    print("✓ Second CUDA dot product test passed!")

    # Benchmark CPU vs CUDA
    separator("BENCHMARK: CPU vs CUDA DOT PRODUCT")
    import time

    # Create larger matrices for benchmarking
    large_m, large_n, large_p = 1000, 1000, 1000
    np_large_a = np.random.rand(large_m, large_n).astype(np.float32)
    np_large_b = np.random.rand(large_n, large_p).astype(np.float32)

    # CPU test
    tensor_large_a_cpu = Tensor.from_numpy(np_large_a)
    tensor_large_b_cpu = Tensor.from_numpy(np_large_b)

    start_time = time.time()
    tensor_result_cpu = Tensor.dot(tensor_large_a_cpu, tensor_large_b_cpu)
    cpu_time = time.time() - start_time
    print(f"CPU dot product time: {cpu_time:.6f} seconds")

    # CUDA test
    tensor_large_a_cuda = Tensor.from_numpy(np_large_a)
    tensor_large_b_cuda = Tensor.from_numpy(np_large_b)
    tensor_large_a_cuda.to(Device.CUDA)
    tensor_large_b_cuda.to(Device.CUDA)

    # Warm-up CUDA
    tensor_result_cuda_warmup = Tensor.dot(tensor_large_a_cuda, tensor_large_b_cuda)

    start_time = time.time()
    tensor_result_cuda = Tensor.dot(tensor_large_a_cuda, tensor_large_b_cuda)
    cuda_time = time.time() - start_time
    print(f"CUDA dot product time: {cuda_time:.6f} seconds")

    # Compare results
    tensor_result_cuda.to(Device.CPU)
    result_cpu = np.array(tensor_result_cpu)
    result_cuda = np.array(tensor_result_cuda)

    # Check if results are close enough
    if np.allclose(result_cpu, result_cuda, atol=1e-3):
        print("✓ CPU and CUDA results match!")
    else:
        print("✗ CPU and CUDA results differ!")

    # Calculate speedup
    speedup = cpu_time / cuda_time
    print(f"CUDA speedup over CPU: {speedup:.2f}x")

except Exception as e:
    print(f"CUDA dot product verification error: {e}")

# ===========================================
# Device transfer tests
# ===========================================
separator("DEVICE TRANSFER")
t_cpu = Tensor.ones((2, 2), Device.CPU)
print(f"Original tensor device: {t_cpu.device}")

try:
    # Transfer to CUDA
    t_cpu.to(Device.CUDA)
    print(f"After transfer to CUDA, device: {t_cpu.device}")

    # Transfer back to CPU
    t_cpu.to(Device.CPU)
    print(f"After transfer back to CPU, device: {t_cpu.device}")
except Exception as e:
    print(f"Device transfer error: {e}")

print("\nAll tests completed!")
