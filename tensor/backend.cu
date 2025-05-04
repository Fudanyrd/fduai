#include "tensor.h" // Include the header file where Tensor is defined
#include <string>
#include <stdexcept>
#include <cuda_runtime.h>

// CUDA kernel for element-wise addition
__global__ void addKernel(const float *a, const float *b, float *result, int num_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements)
    {
        result[idx] = a[idx] + b[idx];
    }
}

__global__ void negKernel(const float *a, float *result, int num_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements)
    {
        result[idx] = -a[idx];
    }
}

// CUDA kernel for matrix multiplication
__global__ void matmulKernel(const float *a, const float *b, float *c, int m, int n, int p)
{
    // a is [m, n] matrix, b is [n, p] matrix, c is [m, p] matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p)
    {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
        {
            sum += a[row * n + k] * b[k * p + col];
        }
        c[row * p + col] = sum;
    }
}

Tensor Tensor::cuda_add(const Tensor &a, const Tensor &b)
{
    if (a.shape != b.shape)
    {
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
    if (err != cudaSuccess)
    {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    return result;
}

Tensor Tensor::cuda_neg(const Tensor &a)
{
    Tensor result(a.shape, Device::CUDA);
    int num_elements = a.num_elements;

    // Kernel launch parameters
    int threads_per_block = 256;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    negKernel<<<blocks_per_grid, threads_per_block>>>(a.data, result.data, num_elements);

    // Check for any kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    return result;
}

Tensor Tensor::cuda_dot(const Tensor &a, const Tensor &b)
{
    // Verify input tensors are 2D
    if (a.shape.size() != 2 || b.shape.size() != 2)
    {
        throw std::invalid_argument("Matrix multiplication requires 2D tensors");
    }

    // Verify matrix dimensions are compatible
    if (a.shape[1] != b.shape[0])
    {
        throw std::invalid_argument("Matrix dimensions mismatch for multiplication: " +
                                    std::to_string(a.shape[0]) + "x" + std::to_string(a.shape[1]) +
                                    " and " + std::to_string(b.shape[0]) + "x" + std::to_string(b.shape[1]));
    }

    // Get matrix dimensions
    int m = a.shape[0]; // rows of output
    int n = a.shape[1]; // inner dimension
    int p = b.shape[1]; // columns of output

    // Create result tensor with appropriate shape [m, p]
    std::vector<int> result_shape = {m, p};
    Tensor result(result_shape, Device::CUDA);

    // Define block and grid dimensions for 2D kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((p + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the matrix multiplication kernel
    matmulKernel<<<blocksPerGrid, threadsPerBlock>>>(a.data, b.data, result.data, m, n, p);

    // Check for any kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error("CUDA matrix multiplication kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    return result;
}
