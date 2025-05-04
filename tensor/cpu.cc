#include "tensor.h"

#include <thread>
#include <cmath>

#define N_THREADS 8

static bool can_broadcast(const std::vector<int> &a, const std::vector<int> &b)
{
    //
    // if a is (t1, t2) and b is (t2,) or (1..1, t2, 1..1)
    // then a can broadcast to b.
    // eg. (2, 3), (3) => ok
    // eg. (2, 3), (1, 3) => ok
    // eg. (2, 3), (2) => NO
    //
    if (a.size() < b.size())
    {
        return false;
    }

    size_t i2 = 0;
    while (i2 < b.size() && b[i2] == 1)
    {
        i2++;
    }
    size_t e2 = b.size() - 1;
    while (e2 >= i2 && b[e2] == 1) {
        e2--;
    }

    size_t i1 = a.size() - b.size() + i2;

    while (i2 < e2)
    {
        if (a[i1] != b[i2])
        {
            return false;
        }
        i1++;
        i2++;
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

static void binOpKernel(const float *a, const float *b, float *result, int num_elements, binary_op op)
{
    if (num_elements <= 8)
    {
        // If the number of elements is small, use a single thread
        for (int k = 0; k < num_elements; ++k)
        {
            result[k] = op(a[k], b[k]);
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
                result[k] = op(a[k], b[k]);
            } });

        threads.push_back(std::move(thread));
    }

    {
        int i = num_elements / 8 * 7;
        int j = num_elements;

        std::thread thread([=]()
                           {
            for (int k = i; k < j; ++k) {
                result[k] = op(a[k], b[k]);
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

static void addKernel(const float *a, const float *b, float *result, int num_elements)
{
    binOpKernel(a, b, result, num_elements, [](float a, float b)
                { return a + b; });
}

static void subKernel(const float *a, const float *b, float *result, int num_elements)
{ 
    binOpKernel(a, b, result, num_elements, [](float a, float b)
                { return a - b; });
}
static void mulKernel(const float *a, const float *b, float *result, int num_elements)
{ 
    binOpKernel(a, b, result, num_elements, [](float a, float b)
                { return a * b; });
}

static void divKernel(const float *a, const float *b, float *result, int num_elements)
{ 
    binOpKernel(a, b, result, num_elements, [](float a, float b)
                { return a / b; });
}

Tensor Tensor::cpu_add_scalar(const Tensor &a, float &b) {
    Tensor result(a.shape, Device::CPU);
    tensorScalarOpKernel(a.data, b, result.data, a.num_elements, [](float a, float b)
                         { return a + b; });
    return result;
}

Tensor Tensor::cpu_add(const Tensor &a, const Tensor &b)
{
    if (b.num_elements == 1) {
        Tensor result(a.shape, Device::CPU);
        float item = b.data[0];

        tensorScalarOpKernel(a.data, item, result.data, a.num_elements, [](float a, float b)
                             { return a + b; });

        return result;
    }

    bool ab = can_broadcast(a.shape, b.shape), ba = can_broadcast(b.shape, a.shape);
    if (!ab && !ba)
    {
        throw std::invalid_argument("Shapes of tensors must match for addition");
    }

    if (ab) {
        //
        // perform a + b
        //
        float *pa = a.data;
        float *pb = b.data;

        Tensor result(a.shape, Device::CPU);
        float *pr = result.data;
        size_t stride = b.num_elements;
        int n = a.num_elements / b.num_elements;

        //
        // FIXME: if n is very large, we should use multi-thread
        //
        for (int i = 0; i < n; i++) {
            addKernel(pa, pb, pr, b.num_elements);
            pa += stride;
            pr += stride;
        }

        return result;
    }

    Tensor result(b.shape, Device::CPU);
    float *pa = a.data, *pb = b.data, *pr = result.data;
    size_t stride = a.num_elements;
    int n = b.num_elements / a.num_elements;

    for (int i = 0; i < n; i++) {
        //
        // FIXME: if n is very large, we should use multi-thread
        //
        addKernel(pa, pb, pr, a.num_elements);
        pb += stride;
        pr += stride;
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
    if (b.shape == scalar_shape) {
        Tensor result(a.shape, Device::CPU);
        float item = b.data[0];

        tensorScalarOpKernel(a.data, item, result.data, a.num_elements, [](float a, float b)
                             { return a - b; });

        return result;
    }

    // FIXME: broadcasting operation is not supported yet
    if (a.shape != b.shape)
    {
        throw std::invalid_argument("Shapes of tensors must match for subtraction");
    }

    Tensor result(a.shape, Device::CPU);
    subKernel(a.data, b.data, result.data, a.num_elements);
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
    if (b.shape == scalar_shape) {
        return cpu_mul_scalar(a, b.data[0]);
    }

    // FIXME: broadcasting operation is not supported yet
    if (a.shape != b.shape)
    {
        throw std::invalid_argument("Shapes of tensors must match for subtraction");
    }

    Tensor result(a.shape, Device::CPU);
    mulKernel(a.data, b.data, result.data, a.num_elements);
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
    if (b.shape == scalar_shape) {
        return cpu_div_scalar(a, b.data[0]);
    }

    bool ab = can_broadcast(a.shape, b.shape), ba = can_broadcast(b.shape, a.shape);
    if (!ab && !ba)
    {
        throw std::invalid_argument("Shapes of tensors must match for division");
    }

    if (ab) {
        //
        // perform a / b
        //
        float *pa = a.data;
        float *pb = b.data;

        Tensor result(a.shape, Device::CPU);
        float *pr = result.data;
        size_t stride = b.num_elements;
        int n = a.num_elements / b.num_elements;

        //
        // FIXME: if n is very large, we should use multi-thread
        //
        for (int i = 0; i < n; i++) {
            divKernel(pa, pb, pr, b.num_elements);
            pa += stride;
            pr += stride;
        }

        return result;
    }

    Tensor result(b.shape, Device::CPU);
    float *pa = a.data, *pb = b.data, *pr = result.data;
    size_t stride = a.num_elements;
    int n = b.num_elements / a.num_elements;

    for (int i = 0; i < n; i++) {
        //
        // FIXME: if n is very large, we should use multi-thread
        //
        divKernel(pa, pb, pr, a.num_elements);
        pb += stride;
        pr += stride;
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

float Tensor::cpu_sum_all(const Tensor &a) {
    return kernel_sum(a.data, a.num_elements);
}
Tensor Tensor::cpu_sum(const Tensor &a, int start_dim) {
    if (start_dim < 0 || start_dim >= a.shape.size()) {
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
    if (a.num_elements < 2 * N_THREADS) {
        return naive_max(a.data, a.num_elements);
    }

    map_fn map = [](float a, float b) { return std::max(a, b); };
    reduce_fn reduce = naive_max;

    //
    // a.data[0] is an element of a.data.
    // this will guarantee the correct result
    //
    float ret = map_reduce(a.data, a.data[0], a.num_elements, map, reduce);
    return ret;
}
