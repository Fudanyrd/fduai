#include "tensor.h"

#include <thread>
#include <cmath>


typedef float (*unary_op)(float a);
typedef float (*binary_op)(float a, float b);

static void unaryOpKernel(const float* a, float* result, int num_elements, unary_op op) {
    if (num_elements <= 8) {
        // If the number of elements is small, use a single thread
        for (int k = 0; k < num_elements; ++k) {
            result[k] = op(a[k]);
        }
        return;
    }

    std::vector<std::thread> threads;
    for (int t = 0; t < 7; t++) {
        int i = num_elements / 8 * t;
        int j = num_elements / 8 * (t + 1);

        std::thread thread([=]() {
            for (int k = i; k < j; ++k) {
                result[k] = op(a[k]);
            }
        });

        threads.push_back(std::move(thread));
    }

    {
        int i = num_elements / 8 * 7;
        int j = num_elements;

        std::thread thread([=]() {
            for (int k = i; k < j; ++k) {
                result[k] = op(a[k]);
            }
        });

        threads.push_back(std::move(thread));
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

static void binOpKernel(const float* a, const float* b, float* result, int num_elements, binary_op op) {
    if (num_elements <= 8) {
        // If the number of elements is small, use a single thread
        for (int k = 0; k < num_elements; ++k) {
            result[k] = a[k] + b[k];
        }
        return;
    }

    std::vector<std::thread> threads;
    for (int t = 0; t < 7; t++) {
        int i = num_elements / 8 * t;
        int j = num_elements / 8 * (t + 1);

        std::thread thread([=]() {
            for (int k = i; k < j; ++k) {
                result[k] = op(a[k], b[k]);
            }
        });

        threads.push_back(std::move(thread));
    }

    {
        int i = num_elements / 8 * 7;
        int j = num_elements;

        std::thread thread([=]() {
            for (int k = i; k < j; ++k) {
                result[k] = op(a[k], b[k]);
            }
        });

        threads.push_back(std::move(thread));
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

static void addKernel(const float* a, const float* b, float* result, int num_elements) {
    binOpKernel(a, b, result, num_elements, [](float a, float b) { return a + b; });
}

Tensor Tensor::cpu_add(const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape) {
        throw std::invalid_argument("Shapes of tensors must match for addition");
    }

    Tensor result(a.shape, Device::CPU);
    addKernel(a.data, b.data, result.data, a.num_elements);
    return result;
}

Tensor Tensor::cpu_neg(const Tensor &a) {
    Tensor result(a.shape, Device::CPU);
    unaryOpKernel(a.data, result.data, a.num_elements, [](float a) { return -a; });
    return result;
}



template <typename T>
void Tensor::cpu_memset(T *dst, T value, size_t len) {
    for (size_t i = 0; i < len; i++) {
        dst[i] = value;
    }
}

template void Tensor::cpu_memset<float>(float *dst, float value, size_t len);

Tensor Tensor::cpu_dot(const Tensor &a, const Tensor &b) {
    // a is [m, n] and b is [n, p], result is [m, p]
    int m = a.shape[0];
    int n = a.shape[1];
    int p = b.shape[1];
    
    // Create result tensor with shape [m, p]
    std::vector<int> result_shape = {m, p};
    Tensor result(result_shape, Device::CPU);
    
    // Perform matrix multiplication
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                // a[i,k] * b[k,j]
                sum += a.data[i * n + k] * b.data[k * p + j];
            }
            result.data[i * p + j] = sum;
        }
    }
    
    return result;
}
