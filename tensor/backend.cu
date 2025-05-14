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

__global__ void subKernel(const float *a, const float *b, float *result, int num_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements)
    {
        result[idx] = a[idx] - b[idx];
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

// CUDA kernel for matrix transpose
__global__ void transposeKernel(const float *input, float *output, int rows, int cols)
{
    // input is [rows, cols], output is [cols, rows]
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // column index
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // row index

    if (ix < cols && iy < rows)
    {
        // Read from input[iy, ix], write to output[ix, iy]
        output[ix * rows + iy] = input[iy * cols + ix];
    }
}

// CUDA kernel for element-wise operation with broadcasting
template <typename Op>
__global__ void broadcastOpKernel(const float *a, const float *b, float *result,
                                  int *a_shape, int a_ndim,
                                  int *b_shape, int b_ndim,
                                  int *result_shape, int result_ndim,
                                  int num_elements, Op op)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements)
    {
        // Convert flat index to multi-dimensional indices for result tensor
        int indices[8]; // Support up to 8 dimensions
        int remaining = idx;
        for (int i = result_ndim - 1; i >= 0; i--)
        {
            indices[i] = remaining % result_shape[i];
            remaining /= result_shape[i];
        }

        // Map to a and b indices with broadcasting
        int a_idx = 0;
        int b_idx = 0;
        int a_stride = 1;
        int b_stride = 1;

        for (int i = a_ndim - 1; i >= 0; i--)
        {
            int result_dim = i + (result_ndim - a_ndim);
            if (result_dim >= 0)
            {
                int a_i = (a_shape[i] == 1) ? 0 : indices[result_dim];
                a_idx += a_i * a_stride;
            }
            a_stride *= a_shape[i];
        }

        for (int i = b_ndim - 1; i >= 0; i--)
        {
            int result_dim = i + (result_ndim - b_ndim);
            if (result_dim >= 0)
            {
                int b_i = (b_shape[i] == 1) ? 0 : indices[result_dim];
                b_idx += b_i * b_stride;
            }
            b_stride *= b_shape[i];
        }

        // Apply the operation
        result[idx] = op(a[a_idx], b[b_idx]);
    }
}

// Operations for the kernels
struct AddOp
{
    __device__ float operator()(float a, float b) const { return a + b; }
};

struct SubOp
{
    __device__ float operator()(float a, float b) const { return a - b; }
};

struct MulOp
{
    __device__ float operator()(float a, float b) const { return a * b; }
};

struct DivOp
{
    __device__ float operator()(float a, float b) const { return a / b; }
};

// Helper function to check if two shapes can be broadcast and get the result shape
bool cuda_can_broadcast(const std::vector<int> &a_shape, const std::vector<int> &b_shape, std::vector<int> &result_shape)
{
    result_shape.resize(std::max(a_shape.size(), b_shape.size()), 1);

    int i1 = a_shape.size() - 1;
    int i2 = b_shape.size() - 1;
    int i3 = result_shape.size() - 1;

    while (i1 >= 0 && i2 >= 0)
    {
        if (a_shape[i1] != b_shape[i2])
        {
            if (a_shape[i1] != 1 && b_shape[i2] != 1)
            {
                return false;
            }
        }
        result_shape[i3] = std::max(a_shape[i1], b_shape[i2]);
        i1--;
        i2--;
        i3--;
    }

    while (i1 >= 0)
    {
        result_shape[i3] = a_shape[i1];
        i1--;
        i3--;
    }

    while (i2 >= 0)
    {
        result_shape[i3] = b_shape[i2];
        i2--;
        i3--;
    }

    return true;
}

Tensor Tensor::cuda_add(const Tensor &a, const Tensor &b)
{
    std::vector<int> result_shape;
    if (!cuda_can_broadcast(a.shape, b.shape, result_shape))
    {
        throw std::invalid_argument("Shapes of tensors cannot be broadcast together for addition");
    }

    Tensor result(result_shape, Device::CUDA);
    int num_elements = result.num_elements;

    // Prepare shape arrays for CUDA
    int *d_a_shape, *d_b_shape, *d_result_shape;
    cudaMalloc(&d_a_shape, a.shape.size() * sizeof(int));
    cudaMalloc(&d_b_shape, b.shape.size() * sizeof(int));
    cudaMalloc(&d_result_shape, result_shape.size() * sizeof(int));

    cudaMemcpy(d_a_shape, a.shape.data(), a.shape.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_shape, b.shape.data(), b.shape.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result_shape, result_shape.data(), result_shape.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel launch parameters
    int threads_per_block = 256;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    broadcastOpKernel<<<blocks_per_grid, threads_per_block>>>(
        a.data, b.data, result.data,
        d_a_shape, a.shape.size(),
        d_b_shape, b.shape.size(),
        d_result_shape, result_shape.size(),
        num_elements, AddOp());

    // Check for any kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    // Free device memory
    cudaFree(d_a_shape);
    cudaFree(d_b_shape);
    cudaFree(d_result_shape);

    return result;
}

Tensor Tensor::cuda_sub(const Tensor &a, const Tensor &b)
{
    std::vector<int> result_shape;
    if (!cuda_can_broadcast(a.shape, b.shape, result_shape))
    {
        throw std::invalid_argument("Shapes of tensors cannot be broadcast together for subtraction");
    }

    Tensor result(result_shape, Device::CUDA);
    int num_elements = result.num_elements;

    // Prepare shape arrays for CUDA
    int *d_a_shape, *d_b_shape, *d_result_shape;
    cudaMalloc(&d_a_shape, a.shape.size() * sizeof(int));
    cudaMalloc(&d_b_shape, b.shape.size() * sizeof(int));
    cudaMalloc(&d_result_shape, result_shape.size() * sizeof(int));

    cudaMemcpy(d_a_shape, a.shape.data(), a.shape.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_shape, b.shape.data(), b.shape.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result_shape, result_shape.data(), result_shape.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel launch parameters
    int threads_per_block = 256;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    broadcastOpKernel<<<blocks_per_grid, threads_per_block>>>(
        a.data, b.data, result.data,
        d_a_shape, a.shape.size(),
        d_b_shape, b.shape.size(),
        d_result_shape, result_shape.size(),
        num_elements, SubOp());

    // Check for any kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    // Free device memory
    cudaFree(d_a_shape);
    cudaFree(d_b_shape);
    cudaFree(d_result_shape);

    return result;
}

Tensor Tensor::cuda_mul(const Tensor &a, const Tensor &b)
{
    std::vector<int> result_shape;
    if (!cuda_can_broadcast(a.shape, b.shape, result_shape))
    {
        throw std::invalid_argument("Shapes of tensors cannot be broadcast together for multiplication");
    }

    Tensor result(result_shape, Device::CUDA);
    int num_elements = result.num_elements;

    // Prepare shape arrays for CUDA
    int *d_a_shape, *d_b_shape, *d_result_shape;
    cudaMalloc(&d_a_shape, a.shape.size() * sizeof(int));
    cudaMalloc(&d_b_shape, b.shape.size() * sizeof(int));
    cudaMalloc(&d_result_shape, result_shape.size() * sizeof(int));

    cudaMemcpy(d_a_shape, a.shape.data(), a.shape.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_shape, b.shape.data(), b.shape.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result_shape, result_shape.data(), result_shape.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel launch parameters
    int threads_per_block = 256;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    broadcastOpKernel<<<blocks_per_grid, threads_per_block>>>(
        a.data, b.data, result.data,
        d_a_shape, a.shape.size(),
        d_b_shape, b.shape.size(),
        d_result_shape, result_shape.size(),
        num_elements, MulOp());

    // Check for any kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    // Free device memory
    cudaFree(d_a_shape);
    cudaFree(d_b_shape);
    cudaFree(d_result_shape);

    return result;
}

Tensor Tensor::cuda_div(const Tensor &a, const Tensor &b)
{
    std::vector<int> result_shape;
    if (!cuda_can_broadcast(a.shape, b.shape, result_shape))
    {
        throw std::invalid_argument("Shapes of tensors cannot be broadcast together for division");
    }

    Tensor result(result_shape, Device::CUDA);
    int num_elements = result.num_elements;

    // Prepare shape arrays for CUDA
    int *d_a_shape, *d_b_shape, *d_result_shape;
    cudaMalloc(&d_a_shape, a.shape.size() * sizeof(int));
    cudaMalloc(&d_b_shape, b.shape.size() * sizeof(int));
    cudaMalloc(&d_result_shape, result_shape.size() * sizeof(int));

    cudaMemcpy(d_a_shape, a.shape.data(), a.shape.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_shape, b.shape.data(), b.shape.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result_shape, result_shape.data(), result_shape.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel launch parameters
    int threads_per_block = 256;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    broadcastOpKernel<<<blocks_per_grid, threads_per_block>>>(
        a.data, b.data, result.data,
        d_a_shape, a.shape.size(),
        d_b_shape, b.shape.size(),
        d_result_shape, result_shape.size(),
        num_elements, DivOp());

    // Check for any kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    // Free device memory
    cudaFree(d_a_shape);
    cudaFree(d_b_shape);
    cudaFree(d_result_shape);

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

Tensor Tensor::cuda_transpose(const Tensor &a)
{
    // Verify input tensor is 2D
    if (a.shape.size() != 2)
    {
        throw std::invalid_argument("Transpose requires a 2D tensor");
    }

    // Get matrix dimensions
    int rows = a.shape[0];
    int cols = a.shape[1];

    // Create result tensor with transposed shape [cols, rows]
    std::vector<int> result_shape = {cols, rows};
    Tensor result(result_shape, Device::CUDA);

    // Define block and grid dimensions for 2D kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the transpose kernel
    transposeKernel<<<blocksPerGrid, threadsPerBlock>>>(a.data, result.data, rows, cols);

    // Check for any kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error("CUDA transpose kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    return result;
}

float Tensor::cuda_sum_all(const Tensor &a)
{
    throw std::runtime_error("CUDA sum_all not implemented yet");
}

float Tensor::cuda_max_all(const Tensor &a)
{
    throw std::runtime_error("CUDA max_all not implemented yet");
}
