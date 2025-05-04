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
# Transpose with NumPy verification - CPU
# ===========================================
separator("TRANSPOSE WITH NUMPY VERIFICATION - CPU")

try:
    # Create test matrices with numpy
    np_matrices = [
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),  # 2x3
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                 dtype=np.float32),  # 3x2
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)  # 2x2
    ]

    for i, np_matrix in enumerate(np_matrices):
        print(f"\nTest case {i+1}: Matrix shape {np_matrix.shape}")
        print(f"Original numpy matrix:\n{np_matrix}")

        # Compute transpose in NumPy
        np_transposed = np_matrix.T
        print(f"NumPy transposed result:\n{np_transposed}")

        # Convert to Tensor
        tensor = Tensor.from_numpy(np_matrix)

        # Compute transpose using our library
        tensor_transposed = Tensor.transpose(tensor)
        print(f"Tensor shape after transpose: {tensor_transposed.shape}")

        # Convert result back to NumPy
        result_np = np.array(tensor_transposed)
        print(f"Tensor transposed result converted to NumPy:\n{result_np}")

        # Verify the results match
        assert np.array_equal(
            np_transposed, result_np), f"Transpose results don't match for case {i+1}"
        print(f"✓ Transpose test {i+1} passed!")

        # Verify double transpose equals original
        tensor_double_transposed = Tensor.transpose(tensor_transposed)
        result_double_np = np.array(tensor_double_transposed)
        assert np.array_equal(
            np_matrix, result_double_np), f"Double transpose test failed for case {i+1}"
        print(f"✓ Double transpose test {i+1} passed!")

except Exception as e:
    print(f"Transpose verification error: {e}")

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
    tensor_result_cuda_warmup = Tensor.dot(
        tensor_large_a_cuda, tensor_large_b_cuda)

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
# CUDA Transpose Test
# ===========================================
separator("TRANSPOSE WITH CUDA")

try:
    # Create test matrices with numpy
    np_matrices = [
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),  # 2x3
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                 dtype=np.float32),  # 3x2
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)  # 2x2
    ]
    
    for i, np_matrix in enumerate(np_matrices):
        print(f"\nTest case {i+1}: Matrix shape {np_matrix.shape}")
        print(f"Original numpy matrix:\n{np_matrix}")
        
        # Compute transpose in NumPy
        np_transposed = np_matrix.T
        print(f"NumPy transposed result:\n{np_transposed}")
        
        # Convert to Tensor and move to CUDA
        tensor = Tensor.from_numpy(np_matrix)
        tensor.to(Device.CUDA)
        
        # Compute transpose using our CUDA implementation
        tensor_transposed = Tensor.transpose(tensor)
        print(f"CUDA tensor result device: {tensor_transposed.device}")
        
        # Transfer result back to CPU for verification
        tensor_transposed.to(Device.CPU)
        
        # Convert to NumPy
        result_np = np.array(tensor_transposed)
        print(f"CUDA transposed result:\n{result_np}")
        
        # Verify the results match
        assert np.array_equal(
            np_transposed, result_np), f"CUDA transpose results don't match for case {i+1}"
        print(f"✓ CUDA transpose test {i+1} passed!")
        
        # Verify double transpose equals original
        tensor = Tensor.from_numpy(np_matrix)
        tensor.to(Device.CUDA)
        tensor_transposed = Tensor.transpose(tensor)
        tensor_double_transposed = Tensor.transpose(tensor_transposed)
        tensor_double_transposed.to(Device.CPU)
        
        result_double_np = np.array(tensor_double_transposed)
        assert np.array_equal(
            np_matrix, result_double_np), f"CUDA double transpose test failed for case {i+1}"
        print(f"✓ CUDA double transpose test {i+1} passed!")
    
    # Benchmark CPU vs CUDA Transpose
    separator("BENCHMARK: CPU vs CUDA TRANSPOSE")
    import time
    
    # Create larger matrices for benchmarking
    large_m, large_n = 1000, 1000
    np_large = np.random.rand(large_m, large_n).astype(np.float32)
    
    # CPU test
    tensor_large_cpu = Tensor.from_numpy(np_large)
    
    start_time = time.time()
    tensor_result_cpu = Tensor.transpose(tensor_large_cpu)
    cpu_time = time.time() - start_time
    print(f"CPU transpose time: {cpu_time:.6f} seconds")
    
    # CUDA test
    tensor_large_cuda = Tensor.from_numpy(np_large)
    tensor_large_cuda.to(Device.CUDA)
    
    # Warm-up CUDA
    tensor_result_cuda_warmup = Tensor.transpose(tensor_large_cuda)
    
    start_time = time.time()
    tensor_result_cuda = Tensor.transpose(tensor_large_cuda)
    cuda_time = time.time() - start_time
    print(f"CUDA transpose time: {cuda_time:.6f} seconds")
    
    # Compare results
    tensor_result_cuda.to(Device.CPU)
    result_cpu = np.array(tensor_result_cpu)
    result_cuda = np.array(tensor_result_cuda)
    
    # Check if results are close enough
    if np.allclose(result_cpu, result_cuda, atol=1e-3):
        print("✓ CPU and CUDA transpose results match!")
    else:
        print("✗ CPU and CUDA transpose results differ!")
    
    # Calculate speedup
    speedup = cpu_time / cuda_time
    print(f"CUDA transpose speedup over CPU: {speedup:.2f}x")
    
except Exception as e:
    print(f"CUDA transpose test error: {e}")

# ===========================================
# CPU vs CUDA Transpose Comparison
# ===========================================
separator("CPU vs CUDA TRANSPOSE COMPARISON")

try:
    # Create test matrices with different characteristics
    test_matrices = [
        # Small matrix (2x3)
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        # Larger square matrix (10x10)
        np.random.rand(10, 10).astype(np.float32),
        # Non-square matrix with many rows (30x5)
        np.random.rand(30, 5).astype(np.float32),
        # Non-square matrix with many columns (5x30)
        np.random.rand(5, 30).astype(np.float32)
    ]
    
    for i, np_matrix in enumerate(test_matrices):
        print(f"\nTest Matrix {i+1}: Shape {np_matrix.shape}")
        
        # CPU transpose
        tensor_cpu = Tensor.from_numpy(np_matrix)
        
        start_time = time.time()
        cpu_result = Tensor.transpose(tensor_cpu)
        cpu_time = time.time() - start_time
        
        cpu_np_result = np.array(cpu_result)
        
        # CUDA transpose
        tensor_cuda = Tensor.from_numpy(np_matrix)
        tensor_cuda.to(Device.CUDA)
        
        start_time = time.time()
        cuda_result = Tensor.transpose(tensor_cuda)
        cuda_time = time.time() - start_time
        
        cuda_result.to(Device.CPU)
        cuda_np_result = np.array(cuda_result)
        
        # Compare results
        results_match = np.array_equal(cpu_np_result, cuda_np_result)
        
        # Print comparison
        print(f"Matrix shape: {np_matrix.shape} → Transposed: {cpu_result.shape}")
        print(f"CPU transpose time: {cpu_time:.6f} seconds")
        print(f"CUDA transpose time: {cuda_time:.6f} seconds")
        if np_matrix.shape[0] * np_matrix.shape[1] <= 36:  # Only print small matrices
            print(f"Original matrix:\n{np_matrix}")
            print(f"CPU result:\n{cpu_np_result}")
            print(f"CUDA result:\n{cuda_np_result}")
        
        if results_match:
            print(f"✓ CPU and CUDA results match!")
            speedup = cpu_time / cuda_time if cuda_time > 0 else float('inf')
            print(f"CUDA speedup: {speedup:.2f}x")
        else:
            print(f"✗ CPU and CUDA results differ!")
            # Print differences for debugging
            if np_matrix.shape[0] * np_matrix.shape[1] <= 100:  # Only for reasonably sized matrices
                diff = np.abs(cpu_np_result - cuda_np_result)
                print(f"Max difference: {np.max(diff)}")
                print(f"Average difference: {np.mean(diff)}")
    
    # Visual separator for the results
    print("\n" + "-" * 50)
    print("  Performance Summary")
    print("-" * 50)
    
    # Test with different matrix sizes to see how performance scales
    sizes = [10, 100, 500, 1000, 2000]
    cpu_times = []
    cuda_times = []
    speedups = []
    
    for size in sizes:
        # Square matrix of given size
        np_matrix = np.random.rand(size, size).astype(np.float32)
        
        # CPU transpose
        tensor_cpu = Tensor.from_numpy(np_matrix)
        start_time = time.time()
        Tensor.transpose(tensor_cpu)
        cpu_time = time.time() - start_time
        cpu_times.append(cpu_time)
        
        # CUDA transpose
        tensor_cuda = Tensor.from_numpy(np_matrix)
        tensor_cuda.to(Device.CUDA)
        
        # Warm-up
        Tensor.transpose(tensor_cuda)
        
        start_time = time.time()
        Tensor.transpose(tensor_cuda)
        cuda_time = time.time() - start_time
        cuda_times.append(cuda_time)
        
        speedup = cpu_time / cuda_time if cuda_time > 0 else float('inf')
        speedups.append(speedup)
        
        print(f"Matrix size {size}x{size}: CPU {cpu_time:.6f}s, CUDA {cuda_time:.6f}s, Speedup {speedup:.2f}x")
    
except Exception as e:
    print(f"CPU vs CUDA comparison error: {e}")

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
