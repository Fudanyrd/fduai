#include <vector>
#include <cuda_runtime.h>
#include <string>
#include <iostream>
#include <cstdlib>

#include <Python.h>
#include <pybind11/buffer_info.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

enum class Device
{
    CPU = 0,
    CUDA = 1,
};

struct Tensor
{
    // Shape of the tensor (e.g., [2, 3, 4] for a 3D tensor)
    std::vector<int> shape;

    // Total number of elements
    int num_elements;

    // Pointer to the data on the GPU
    float *data;

    // Device type (CPU or CUDA)
    Device device;

    // Constructor
    Tensor(const std::vector<int> &shape, Device dev) : shape(shape), device(dev)
    {
        num_elements = 1;
        for (int dim : shape)
        {
            num_elements *= dim;
        }

        size_t buf_size = num_elements * sizeof(float);

        if (device == Device::CUDA)
        {
            auto ret = cudaMalloc(&data, buf_size);

            if (ret != cudaSuccess)
            {
                throw std::runtime_error("Failed to allocate memory on GPU");
            }
        }
        else if (device == Device::CPU)
        {
            data = static_cast<float *>(malloc(buf_size));
            if (!data)
            {
                throw std::runtime_error("Failed to allocate memory on CPU");
            }
        }
        else
        {
            throw std::invalid_argument("Invalid device type");
        }
    }

    // Destructor
    ~Tensor()
    {
        if (data)
        {

            if (device == Device::CUDA)
                cudaFree(data);
            else
                free(data);
        }
    }

    // Copy constructor and assignment operator should be deleted to prevent accidental copying
    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;

    // Move constructor and assignment operator
    Tensor(Tensor &&other) noexcept : shape(std::move(other.shape)), num_elements(other.num_elements), data(other.data),
                                      device(other.device)
    {
        other.data = nullptr;
    }

    Tensor &operator=(Tensor &&other) noexcept
    {
        if (this != &other)
        {
            if (data)
            {
                cudaFree(data);
            }
            shape = std::move(other.shape);
            device = other.device;
            num_elements = other.num_elements;
            data = other.data;
            other.data = nullptr;
        }
        return *this;
    }

    std::string __repr__() const
    {
        std::string repr = "Tensor (";
        for (size_t i = 0; i < shape.size(); ++i)
        {
            repr += std::to_string(shape[i]);
            if (i < shape.size() - 1)
            {
                repr += ", ";
            }
        }
        repr += ")";
        return repr;
    }

    // Element-wise addition
    Tensor __add__(const Tensor &other) const
    {
        return add(*this, other);
    }
    Tensor operator+(const Tensor &other) const {
        return add(*this, other);
    }
    Tensor operator+(float scalar) const {
        return add_scalar(*this, scalar);
    }

    Tensor __sub__(const Tensor &other) const
    {
        return sub(*this, other);
    }
    Tensor operator-(const Tensor &other) const {
        return sub(*this, other);
    }
    Tensor operator-(float scalar) const {
        return sub_scalar(*this, scalar);
    }

    Tensor operator*(const Tensor &other) const {
        return mul(*this, other);
    }
    Tensor operator*(float scalar) const {
        return mul_scalar(*this, scalar);
    }
    Tensor operator/(const Tensor &other) const {
        return div(*this, other);
    }
    Tensor operator/(float scalar) const {
        return div_scalar(*this, scalar);
    }

    Tensor __neg__() const
    {
        return neg(*this);
    }

    float __getitem__(int index) const
    {
        if (index < 0 || index >= num_elements)
        {
            throw std::out_of_range("Index out of range");
        }

        if (device == Device::CPU)
        {
            return data[index]; // Direct access for CPU
        }

        float value;
        cudaMemcpy(&value, data + index, sizeof(float), cudaMemcpyDeviceToHost);
        return value;
    }

    void __setitem__(int index, float value)
    {
        if (index < 0 || index >= num_elements)
        {
            throw std::out_of_range("Index out of range");
        }

        if (device == Device::CPU)
        {
            data[index] = value; // Direct access for CPU
        }
        else
        {
            cudaMemcpy(data + index, &value, sizeof(float), cudaMemcpyHostToDevice);
        }
    }

    size_t __len__() const
    {
        return this->shape[0];
    }

    static Tensor exp(const Tensor &a) {
        if (a.device == Device::CPU) {
            return cpu_exp(a);
        }
        throw std::invalid_argument("CUDA exp is not implemented yet");
    }

    static bool allclose(const Tensor &a, const Tensor &b, float atol);

    static Tensor zeros(const std::vector<int> &shape, Device dev = Device::CUDA);
    static Tensor ones(const std::vector<int> &shape, Device dev);
    void to(Device device);
    Tensor clone() const;

    void save(const std::string &filename) const;

    PyObject *to_numpy(void) const;
    pybind11::list to_list(void) const;
    static Tensor from_list(const pybind11::list &list);

    pybind11::buffer_info get_buffer_info();
    static Tensor from_numpy(pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> &arr)
    {
        //
        // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html#arrays
        // restrict the function to only accept numpy, c_style, float arrays
        //

        auto ndim = arr.ndim();
        std::vector<int> shape(arr.ndim());

        for (auto i = 0; i < ndim; i++)
        {
            shape[i] = arr.shape(i);
        }

        Tensor tensor(shape, Device::CPU);
        std::memcpy(tensor.data, arr.data(), tensor.num_elements * sizeof(float));
        return tensor;
    }

    static Tensor dot(const Tensor &a, const Tensor &b)
    {
        if (a.shape.size() != 2 || b.shape.size() != 2)
        {
            throw std::invalid_argument("Dot product requires 2D tensors");
        }

        if (a.shape[1] != b.shape[0])
        {
            throw std::invalid_argument("Incompatible dimensions for dot product: " +
                                        std::to_string(a.shape[0]) + "x" + std::to_string(a.shape[1]) + " and " +
                                        std::to_string(b.shape[0]) + "x" + std::to_string(b.shape[1]));
        }

        if (a.device == Device::CUDA && b.device == Device::CUDA)
        {
            return cuda_dot(a, b);
        }
        else if (a.device == Device::CPU && b.device == Device::CPU)
        {
            return cpu_dot(a, b);
        }
        else
        {
            throw std::invalid_argument("Cannot perform dot product on tensors with different devices");
        }
    }

    static Tensor transpose(const Tensor &a)
    {
        if (a.shape.size() != 2)
        {
            throw std::invalid_argument("Transpose requires a 2D tensor");
        }

        std::vector<int> transposed_shape = {a.shape[1], a.shape[0]};

        if (a.device == Device::CUDA)
        {
            return cuda_transpose(a);
        }
        else if (a.device == Device::CPU)
        {
            return cpu_transpose(a);
        }
        else
        {
            throw std::invalid_argument("Cannot transpose tensor on unknown device");
        }
    }

    static float sum_all(const Tensor &a) {
        if (a.device == Device::CUDA)
        {
            return cuda_sum_all(a);
        }
        else if (a.device == Device::CPU)
        {
            return cpu_sum_all(a);
        }
        else
        {
            throw std::invalid_argument("Cannot sum tensor on unknown device");
        }
    }
    static Tensor sum(const Tensor &a, int start_dim = 0) {
        if (a.device == Device::CPU) {
            return cpu_sum(a, start_dim);
        }
        throw std::invalid_argument("CUDA sum is not implemented yet");
    }

    static float max_all(const Tensor &a) {
        if (a.device == Device::CUDA)
        {
            return cuda_max_all(a);
        }
        else if (a.device == Device::CPU)
        {
            return cpu_max_all(a);
        }
        else
        {
            throw std::invalid_argument("Cannot max tensor on unknown device");
        }
    }
    static Tensor max(const Tensor &a, bool keep_dim = false, int start_dim = 0) {
        if (a.device == Device::CPU) {
            return cpu_max(a, keep_dim, start_dim);
        }
        throw std::invalid_argument("CUDA max is not implemented yet");
    }
private:
    static Tensor add(const Tensor &a, const Tensor &b)
    {
        if (a.device == Device::CUDA && b.device == Device::CUDA)
        {
            return cuda_add(a, b);
        }
        else if (a.device == Device::CPU && b.device == Device::CPU)
        {
            return cpu_add(a, b);
        }
        else
        {
            throw std::invalid_argument("Cannot add tensors on different devices");
        }
    }
    static Tensor add_scalar(const Tensor &a, float &b) {
        if (a.device == Device::CUDA)
        {
            return cuda_add_scalar(a, b);
        }
        else if (a.device == Device::CPU)
        {
            return cpu_add_scalar(a, b);
        }
        else
        {
            throw std::invalid_argument("Cannot add scalar to tensor on unknown device");
        }
    }
    static Tensor cpu_add(const Tensor &a, const Tensor &b);
    static Tensor cpu_add_scalar(const Tensor &a, float &b);
    static Tensor cuda_add(const Tensor &a, const Tensor &b);
    static Tensor cuda_add_scalar(const Tensor &a, float &b) {
        //
        // FIXME: add implementation
        //
        throw std::invalid_argument("CUDA add scalar not implemented yet");
    }


    static Tensor sub(const Tensor &a, const Tensor &b)
    {
        if (a.device == Device::CUDA && b.device == Device::CUDA)
        {
            return cuda_sub(a, b);
        }
        else if (a.device == Device::CPU && b.device == Device::CPU)
        {
            return cpu_sub(a, b);
        }
        else
        {
            throw std::invalid_argument("Cannot sub tensors on different devices");
        }
    }
    static Tensor sub_scalar(const Tensor &a, float b) {
        if (a.device == Device::CUDA)
        {
            return cuda_sub_scalar(a, b);
        }
        else if (a.device == Device::CPU)
        {
            return cpu_sub_scalar(a, b);
        }
        else
        {
            throw std::invalid_argument("Cannot sub scalar to tensor on unknown device");
        }
    }
    static Tensor cpu_sub(const Tensor &a, const Tensor &b);
    static Tensor cuda_sub(const Tensor &a, const Tensor &b);
    static Tensor cpu_sub_scalar(const Tensor &a, float &b);
    static Tensor cuda_sub_scalar(const Tensor &a, float &b) {
        //
        // FIXME: CUDA sub scalar is not implemented yet
        //
        throw std::invalid_argument("CUDA sub scalar not implemented yet");
    }

    static Tensor mul(const Tensor &a, const Tensor &b)
    {
        if (a.device == Device::CUDA && b.device == Device::CUDA)
        {
            return cuda_mul(a, b);
        }
        else if (a.device == Device::CPU && b.device == Device::CPU)
        {
            return cpu_mul(a, b);
        }
        else
        {
            throw std::invalid_argument("Cannot mul tensors on different devices");
        }
    }
    static Tensor mul_scalar(const Tensor &a, float b) {
        if (a.device == Device::CUDA)
        {
            return cuda_mul_scalar(a, b);
        }
        else if (a.device == Device::CPU)
        {
            return cpu_mul_scalar(a, b);
        }
        else
        {
            throw std::invalid_argument("Cannot mul scalar to tensor on unknown device");
        }
    }
    static Tensor cpu_mul(const Tensor &a, const Tensor &b);
    static Tensor cuda_mul(const Tensor &a, const Tensor &b) {
        //
        // FIXME: CUDA mul is not implemented yet
        //
        throw std::invalid_argument("CUDA mul not implemented yet");
    }
    static Tensor cpu_mul_scalar(const Tensor &a, float &b);
    static Tensor cuda_mul_scalar(const Tensor &a, float &b) {
        //
        // FIXME: CUDA mul scalar is not implemented yet
        //
        throw std::invalid_argument("CUDA mul scalar not implemented yet");
    }


    static Tensor div(const Tensor &a, const Tensor &b)
    {
        if (a.device == Device::CUDA && b.device == Device::CUDA)
        {
            return cuda_div(a, b);
        }
        else if (a.device == Device::CPU && b.device == Device::CPU)
        {
            return cpu_div(a, b);
        }
        else
        {
            throw std::invalid_argument("Cannot divide tensors on different devices");
        }
    }
    static Tensor div_scalar(const Tensor &a, float b) {
        if (a.device == Device::CUDA)
        {
            return cuda_div_scalar(a, b);
        }
        else if (a.device == Device::CPU)
        {
            return cpu_div_scalar(a, b);
        }
        else
        {
            throw std::invalid_argument("Cannot div scalar to tensor on unknown device");
        }
    }
    static Tensor cpu_div(const Tensor &a, const Tensor &b);
    static Tensor cuda_div(const Tensor &a, const Tensor &b) {
        //
        // FIXME: CUDA div is not implemented yet
        //
        throw std::invalid_argument("CUDA div not implemented yet");
    }
    static Tensor cpu_div_scalar(const Tensor &a, float &b);
    static Tensor cuda_div_scalar(const Tensor &a, float &b) {
        //
        // FIXME: CUDA div scalar is not implemented yet
        //
        throw std::invalid_argument("CUDA div scalar not implemented yet");
    }

    static Tensor neg(const Tensor &a)
    {
        if (a.device == Device::CPU)
        {
            return cpu_neg(a);
        }
        else if (a.device == Device::CUDA)
        {
            return cuda_neg(a);
        }
        else
        {
            throw std::invalid_argument("Cannot negate tensors on different devices");
        }
    }
    static Tensor cpu_neg(const Tensor &a);
    static Tensor cuda_neg(const Tensor &a);

    static Tensor cpu_dot(const Tensor &a, const Tensor &b);
    static Tensor cuda_dot(const Tensor &a, const Tensor &b);

    static Tensor cpu_transpose(const Tensor&a);
    static Tensor cuda_transpose(const Tensor&a);

    static Tensor cpu_exp(const Tensor &a);

    template <typename T>
    static void cpu_memset(T *dst, T value, size_t len);

    std::vector<size_t> get_strides() const;

    static float cpu_sum_all(const Tensor &a);
    static Tensor cpu_sum(const Tensor &a, int start_dim = 0);
    static float cuda_sum_all(const Tensor &a);


    static float cpu_max_all(const Tensor &a);
    static Tensor cpu_max(const Tensor &a, bool keep_dim, int start_dim = 0);
    // static Tensor cpu_max(const Tensor &a, int start_dim = 0);
    static float cuda_max_all(const Tensor &a);
};
