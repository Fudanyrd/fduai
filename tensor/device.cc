#include "tensor.h"


Tensor Tensor::zeros(const std::vector<int>& shape, Device dev) {
    if (dev == Device::CPU) {
        Tensor tensor(shape, Device::CPU);
        float val = 0.0f;
        Tensor::cpu_memset<float>(tensor.data, val, tensor.num_elements);
        return tensor;
    }
    Tensor tensor(shape, Device::CUDA);
    std::vector<float> zeros(tensor.num_elements, 0.0f);
    cudaMemcpy(tensor.data, zeros.data(), tensor.num_elements * sizeof(float), cudaMemcpyHostToDevice);
    return tensor;
}

Tensor Tensor::ones(const std::vector<int>& shape, Device dev) {
    if (dev == Device::CPU) {
        Tensor tensor(shape, Device::CPU);
        float val = 1.0f;
        Tensor::cpu_memset<float>(tensor.data, val, tensor.num_elements);
        return tensor;
    }
    Tensor tensor(shape, Device::CUDA);
    std::vector<float> ones(tensor.num_elements, 1.0f);
    cudaMemcpy(tensor.data, ones.data(), tensor.num_elements * sizeof(float), cudaMemcpyHostToDevice);
    return tensor;
}

void Tensor::to(Device device) {
    if (this->device == device) {
        return; // No need to transfer if already on the same device
    }
    if (device == Device::CPU) {
        size_t buf_size = num_elements * sizeof(float);
        float * host_data = static_cast<float*>(malloc(buf_size));
        cudaMemcpy(host_data, data, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(data);
        data = host_data;
        // Allocate CPU memory and copy data back to CPU (not shown here)
    } else if (device == Device::CUDA) {
        // Allocate GPU memory and copy data to GPU (not shown here)
        float * gpu_data;
        cudaMalloc(&gpu_data, num_elements * sizeof(float));
        cudaMemcpy(gpu_data, data, num_elements * sizeof(float), cudaMemcpyHostToDevice);
        free(data); // Free the old CPU data
        data = gpu_data;
    }
    this->device = device;
}

Tensor Tensor::clone() const {
    Tensor ret(shape, device);
    if (device == Device::CPU) {
        std::copy(data, data + num_elements, ret.data);
    } else if (device == Device::CUDA) {
        cudaMemcpy(ret.data, data, num_elements * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    return ret;
}
