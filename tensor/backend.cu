#include "tensor.h" // Include the header file where Tensor is defined
#include <string>
#include <stdexcept>
#include <cuda_runtime.h>

// CUDA kernel for element-wise addition
__global__ void addKernel(const float* a, const float* b, float* result, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        result[idx] = a[idx] + b[idx];
    }
}

__global__ void negKernel(const float *a, float *result, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        result[idx] = -a[idx];
    }
}

Tensor Tensor::cuda_add(const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape) {
        throw std::invalid_argument("Tensors must have the same shape for element-wise addition");
    }

    Tensor result(a.shape, Device::CUDA);
    int num_elements = a.num_elements;

    // Kernel launch parameters
    int threads_per_block = 256;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    addKernel<<<blocks_per_grid, threads_per_block>>>(a.data, b.data, result.data, num_elements);

    // Check for any kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    return result;
}

Tensor Tensor::cuda_neg(const Tensor& a) {
    Tensor result(a.shape, Device::CUDA);
    int num_elements = a.num_elements;

    // Kernel launch parameters
    int threads_per_block = 256;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    negKernel<<<blocks_per_grid, threads_per_block>>>(a.data, result.data, num_elements);

    // Check for any kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    return result;
}
