#include "tensor.h"

#include <thread>
#include <cmath>

#include <omp.h>

#define N_THREADS 8

//
// If a and b can broadcast, set `shape` to the broadcasted shape.
// Else `shape` is undefined.
//
static bool can_broadcast(const std::vector<int> &a, const std::vector<int> &b, std::vector<int> &shape)
{
    //
    // use the same broadcasting rules as numpy
    // https://numpy.org/doc/stable/user/basics.broadcasting.html#broadcasting
    // eg. (3,2) + (3,1) ok
    // (3,2) + (2,) ok
    // (3,2) + (1,) ok
    //

    //
    // When operating on two arrays, NumPy compares their shapes element-wise. 
    // It starts with the trailing (i.e. rightmost) dimension and works its way left. 
    // Two dimensions are compatible when:
    // 1. they are equal, or
    // 2. one of them is 1
    //

    int i1 = a.size() - 1;
    int i2 = b.size() - 1;
    int i3 = shape.size() - 1;
    while (i1 >= 0 && i2 >= 0)
    {
        if (a[i1] != b[i2])
        {
            if (a[i1] != 1 && b[i2] != 1)
            {
                return false;
            }
        }
        shape[i3] = std::max(a[i1], b[i2]);
        i1--;
        i2--;
        i3--;
    }

    while (i1 >= 0)
    {
        shape[i3] = a[i1];
        i1--;
        i3--;
    }

    while (i2 >= 0)
    {
        shape[i3] = b[i2];
        i2--;
        i3--;
    }

    return true;
}

typedef float (*unary_op)(float a);
typedef float (*binary_op)(float a, float b);
typedef float (*map_fn)(float a, float b);
typedef float (*reduce_fn)(const float *a, size_t len);

static const std::vector<int> scalar_shape = {1};

/*+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
 * Kernel functions
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-*/

static void unaryOpKernel(const float *a, float *result, int num_elements, unary_op op)
{
    if (num_elements <= 8)
    {
        // If the number of elements is small, use a single thread
        for (int k = 0; k < num_elements; ++k)
        {
            result[k] = op(a[k]);
        }
        return;
    }

    std::vector<std::thread> threads;
    for (int t = 0; t < 7; t++)
    {
        int i = num_elements / 8 * t;
        int j = num_elements / 8 * (t + 1);

        std::thread thread([=]()
                           {
            for (int k = i; k < j; ++k) {
                result[k] = op(a[k]);
            } });

        threads.push_back(std::move(thread));
    }

    {
        int i = num_elements / 8 * 7;
        int j = num_elements;

        std::thread thread([=]()
                           {
            for (int k = i; k < j; ++k) {
                result[k] = op(a[k]);
            } });

        threads.push_back(std::move(thread));
    }

    for (auto &thread : threads)
    {
        thread.join();
    }
}

static void tensorScalarOpKernel(const float *a, const float b, float *result, int num_elements, binary_op op)
{
    if (num_elements <= 8)
    {
        // If the number of elements is small, use a single thread
        for (int k = 0; k < num_elements; ++k)
        {
            result[k] = op(a[k], b);
        }
        return;
    }

    std::vector<std::thread> threads;
    for (int t = 0; t < 7; t++)
    {
        int i = num_elements / 8 * t;
        int j = num_elements / 8 * (t + 1);

        std::thread thread([=]()
                           {
            for (int k = i; k < j; ++k) {
                result[k] = op(a[k], b);
            } });

        threads.push_back(std::move(thread));
    }

    {
        int i = num_elements / 8 * 7;
        int j = num_elements;

        std::thread thread([=]()
                           {
            for (int k = i; k < j; ++k) {
                result[k] = op(a[k], b);
            } });

        threads.push_back(std::move(thread));
    }

    for (auto &thread : threads)
    {
        thread.join();
    }
}

/*+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
 * Operators
  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-*/

Tensor Tensor::cpu_add_scalar(const Tensor &a, float &b) {
    Tensor result(a.shape, Device::CPU);
    tensorScalarOpKernel(a.data, b, result.data, a.num_elements, [](float a, float b)
                         { return a + b; });
    return result;
}

Tensor Tensor::cpu_add(const Tensor &a, const Tensor &b)
{

    std::vector<int> shape(std::max(a.shape.size(), b.shape.size()), 1);

    auto broadcast = can_broadcast(a.shape, b.shape, shape);
    if (!broadcast) {
        throw std::invalid_argument("Shapes of tensors must match for addition");
    }

    Tensor result(shape, Device::CPU);

    //
    // accelerated with openmp
    //
    #pragma omp parallel for
    for (int i = 0; i < result.num_elements; i++) {
        result.data[i] = (*a.view(shape, i)) + (*b.view(shape, i));
    }

    return result;
}

Tensor Tensor::cpu_sub_scalar(const Tensor &a, float &b) {
    Tensor result(a.shape, Device::CPU);
    tensorScalarOpKernel(a.data, b, result.data, a.num_elements, [](float a, float b)
                         { return a - b; });
    return result;
}
Tensor Tensor::cpu_sub(const Tensor &a, const Tensor &b)
{
    std::vector<int> shape(std::max(a.shape.size(), b.shape.size()), 1);

    auto broadcast = can_broadcast(a.shape, b.shape, shape);
    if (!broadcast) {
        throw std::invalid_argument("Shapes of tensors must match for subtraction");
    }

    Tensor result(shape, Device::CPU);
    #pragma omp parallel for
    for (int i = 0; i < result.num_elements; i++) {
        result.data[i] = (*a.view(shape, i)) - (*b.view(shape, i));
    }
    return result;
}

Tensor Tensor::cpu_mul_scalar(const Tensor &a, float &b) {
    Tensor result(a.shape, Device::CPU);
    tensorScalarOpKernel(a.data, b, result.data, a.num_elements, [](float a, float b)
                         { return a * b; });
    return result;
}
Tensor Tensor::cpu_mul(const Tensor &a, const Tensor &b)
{
    std::vector<int> shape(std::max(a.shape.size(), b.shape.size()), 1);

    auto broadcast = can_broadcast(a.shape, b.shape, shape);
    if (!broadcast) {
        throw std::invalid_argument("Shapes of tensors must match for multiply operation");
    }

    Tensor result(shape, Device::CPU);
    #pragma omp parallel for
    for (int i = 0; i < result.num_elements; i++) {
        result.data[i] = (*a.view(shape, i)) * (*b.view(shape, i));
    }

    return result;
}

Tensor Tensor::cpu_div_scalar(const Tensor &a, float &b) {
    Tensor result(a.shape, Device::CPU);
    tensorScalarOpKernel(a.data, b, result.data, a.num_elements, [](float a, float b)
                         { return a / b; });
    return result;
}
Tensor Tensor::cpu_div(const Tensor &a, const Tensor &b)
{
    std::vector<int> shape(std::max(a.shape.size(), b.shape.size()), 1);

    auto broadcast = can_broadcast(a.shape, b.shape, shape);
    if (!broadcast) {
        throw std::invalid_argument("Shapes of tensors must match for division");
    }

    Tensor result(shape, Device::CPU);
    #pragma omp parallel for
    for (int i = 0; i < result.num_elements; i++) {
        result.data[i] = (*a.view(shape, i)) / (*b.view(shape, i));
    }

    return result;
}

Tensor Tensor::cpu_lt(const Tensor &a, const Tensor &b)
{
    std::vector<int> shape(std::max(a.shape.size(), b.shape.size()), 1);

    auto broadcast = can_broadcast(a.shape, b.shape, shape);
    if (!broadcast) {
        throw std::invalid_argument("Shapes of tensors must match for comparison \"<\".");
    }

    Tensor result(shape, Device::CPU);
    #pragma omp parallel for
    for (int i = 0; i < result.num_elements; i++) {
        const float ea = (*a.view(shape, i));
        const float eb = (*b.view(shape, i));
        result.data[i] = (ea < eb) ? 1.0f : 0.0f;
    }

    return result;
}

Tensor Tensor::cpu_neg(const Tensor &a)
{
    Tensor result(a.shape, Device::CPU);
    unaryOpKernel(a.data, result.data, a.num_elements, [](float a)
                  { return -a; });
    return result;
}

template <typename T>
void Tensor::cpu_memset(T *dst, T value, size_t len)
{
    #pragma omp parallel for
    for (size_t i = 0; i < len; i++)
    {
        dst[i] = value;
    }
}

template void Tensor::cpu_memset<float>(float *dst, float value, size_t len);

Tensor Tensor::cpu_dot(const Tensor &a, const Tensor &b)
{
    // a is [m, n] and b is [n, p], result is [m, p]
    int m = a.shape[0];
    int n = a.shape[1];
    int p = b.shape[1];

    // Create result tensor with shape [m, p]
    std::vector<int> result_shape = {m, p};
    Tensor result(result_shape, Device::CPU);

    // Perform matrix multiplication
    #pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < p; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < n; k++)
            {
                // a[i,k] * b[k,j]
                sum += a.data[i * n + k] * b.data[k * p + j];
            }
            result.data[i * p + j] = sum;
        }
    }

    return result;
}

Tensor Tensor::cpu_transpose(const Tensor &a)
{
    // a is [m, n]
    int m = a.shape[0];
    int n = a.shape[1];

    std::vector<int> result_shape = {n, m};
    Tensor result(result_shape, Device::CPU);

    // Perform Transpose
    for (int ix = 0; ix < n; ix++)
    {
        for (int jx = 0; jx < m; jx++)
        {
            result.data[ix * m + jx] = a.data[jx * n + ix];
        }
    }

    return result;
}

Tensor Tensor::cpu_exp(const Tensor &a) {
    Tensor result(a.shape, Device::CPU);
    unaryOpKernel(a.data, result.data, a.num_elements, [](float a)
                  { return std::exp(a); });
    return result;
}

static float map_reduce(const float *a, float default_val, size_t n, map_fn map, reduce_fn reduce)
{
    float *buf = static_cast<float *>(malloc(N_THREADS * sizeof(float)));
    size_t chunk_size = n / N_THREADS;
    
    /* Begin map */
    std::vector<std::thread> threads;
    for (int t = 0; t < N_THREADS; t++)
    {
        int i = chunk_size * t;
        int j = (t + 1 == N_THREADS) ? n : chunk_size * (t + 1);

        std::thread thread([=]()
                           {
            buf[t] = default_val;
            for (int k = i; k < j; ++k)
            {
                buf[t] = map(a[k], buf[t]);
            } });
        threads.push_back(std::move(thread));
    }

    for (auto &thread : threads)
    {
        thread.join();
    }
    /* End map */

    float ret = reduce(buf, N_THREADS);

    free(buf);
    return ret;
}

static float naive_sum(const float *a, size_t n) {
    float ret = 0.0f;
    for (size_t i = 0; i < n; i++) {
        ret += a[i];
    }
    return ret;
}
static float kernel_sum(const float *a, size_t n) {
    if (n < 2 * N_THREADS) {
        return naive_sum(a, n);
    }

    map_fn map = [](float a, float b) { return a + b; };
    reduce_fn reduce = naive_sum;
    float ret = map_reduce(a, 0.0f, n, map, reduce);
    return ret;
}

static float naive_max(const float *a, size_t n) {
    float ret = a[0];
    for (size_t i = 1; i < n; i++) {
        ret = std::max(ret, a[i]);
    }
    return ret;
}

static float kernel_max(const float *a, size_t n) {
    if (n < 2 * N_THREADS) {
        return naive_max(a, n);
    }

    map_fn map = [](float a, float b) { return std::max(a, b); };
    reduce_fn reduce = naive_max;
    float ret = map_reduce(a, a[0], n, map, reduce);

    return ret;
}

float Tensor::cpu_sum_all(const Tensor &a) {
    return kernel_sum(a.data, a.num_elements);
}
Tensor Tensor::cpu_sum(const Tensor &a, int start_dim) {
    if (start_dim < 0 || (size_t)start_dim >= a.shape.size()) {
        throw std::invalid_argument("start_dim must be 1 or greater");
    }

    if (start_dim == 0) {
        //
        // the same as sum_all, but returns a tensor.
        //
        Tensor result({1}, Device::CPU);
        result.data[0] = cpu_sum_all(a);
        return result;
    }

    std::vector<int> shape;
    size_t n = 1;
    for (int i = 0; i < start_dim; i++) {
        shape.push_back(a.shape[i]);
        n *= a.shape[i];
    }
    n = a.num_elements / n;

    Tensor result(shape, Device::CPU);
    for (int i = 0; i < result.num_elements; i++) {
        result.data[i] = kernel_sum(a.data + i * n, n);
    }

    return result;
}

float Tensor::cpu_max_all(const Tensor &a) {
    return kernel_max(a.data, a.num_elements);
}

Tensor Tensor::cpu_max(const Tensor &a, bool keep_dim, int start_dim) {
    if (start_dim < 0 || (size_t)start_dim >= a.shape.size()) {
        throw std::invalid_argument("start_dim must be 1 or greater");
    }

    if (start_dim == 0) {
        //
        // the same as max_all, but returns a tensor.
        //
        Tensor result({1}, Device::CPU);
        result.data[0] = cpu_max_all(a);
        return result;
    }

    std::vector<int> shape;
    size_t n = 1;
    for (int i = 0; i < start_dim; i++) {
        shape.push_back(a.shape[i]);
        n *= a.shape[i];
    }
    n = a.num_elements / n;

    Tensor result(shape, Device::CPU);
    for (int i = 0; i < result.num_elements; i++) {
        result.data[i] = kernel_max(a.data + i * n, n);
    }

    if (keep_dim) {
        //
        // pad 1 dim util the shape is same as a.shape
        //
        for (size_t k = result.shape.size(); k < a.shape.size(); k++) {
            result.shape.push_back(1);
        }
    }

    return result;
}

Tensor Tensor::cpu_relu(const Tensor &a) {
    Tensor result(a.shape, Device::CPU);

    #pragma omp parallel for
    for (int i = 0; i < result.num_elements; i++) {
        result.data[i] = std::max(0.0f, a.data[i]);
    }

    return result;
}

Tensor Tensor::cpu_grad_reshape(const Tensor &a, const std::vector<int> &shape) {
    Tensor ret(shape, Device::CPU);
    Tensor::cpu_memset<float>(ret.data, 0.0f, ret.num_elements);

    #pragma omp parallel for
    for (int i = 0; i < a.num_elements; i++) {
        float *dst = ret.view_mut(a.shape, i);
        *dst = *dst + a.data[i];
    }

    return ret;
}
