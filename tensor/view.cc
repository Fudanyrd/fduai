#include "tensor.h"

//
// to implement broadcasting, Tensor must be able to be viewed as a tensor of another shape.
//

#ifndef __TEST__
#warning "Sanity checks are disabled. Please enable them for production code."
#endif // __TEST__

size_t indices_to_offset(const std::vector<int> &shape, const std::vector<int> &indices) {
    size_t offset = 0;
    size_t n_elem = 1;

    for (size_t i = 0; i < shape.size(); i++) {
        if (indices[i] < 0 || indices[i] >= shape[i]) {
            throw std::out_of_range("Index out of range");
        }

        offset += n_elem * indices[i];
        n_elem *= shape[i];
    }

    #ifdef __TEST__
    if (offset >= n_elem) {
        throw std::out_of_range("Index out of range");
    }
    #endif // __TEST__

    return offset;
}

const float *Tensor::view(const std::vector<int> &asshape, const std::vector<int> &indices) const {
    int n_elem = this->num_elements;

    if (asshape.size() < shape.size() || asshape.size() != indices.size()) {
        throw std::invalid_argument("Shape and indices size mismatch");
    }
    const float *ret = data;
    const size_t diff = asshape.size() - shape.size();

    for (size_t i = 0; i < shape.size(); i++) {
        #ifdef __TEST__
        if (shape[i] != asshape[i + diff] && shape[i] != 1) {
            throw std::invalid_argument("Shape and indices mismatch");
        }
        #endif // __TEST__

        if (shape[i] == 1) {
            continue;
        }

        const int idx = indices[i + diff];
        #ifdef __TEST__
        if (idx < 0 || idx >= shape[i]) {
            throw std::out_of_range("Index out of range");
        }
        #endif // __TEST__

        n_elem /= shape[i];
        ret += n_elem * idx;
    }

    #ifdef __TEST__
    if (ret < data || ret >= data + num_elements) {
        throw std::out_of_range("Index out of range");
    }
    #endif // __TEST__

    return ret;
}

const float *Tensor::view(const std::vector<int> &asshape, int index) const {
    int view_elem = 1;
    const size_t diff = asshape.size() - shape.size();

#ifdef __TEST__
    for (const auto i : asshape) {
        view_elem *= i;
    }
    if (index >= view_elem) {
        throw std::out_of_range("Index out of range");
    }
    for (size_t i = 0; i < diff; i++) {
        view_elem /= asshape[i];
    }
#else 
    for (size_t i = 0; i < shape.size(); i++) {
        view_elem *= asshape[i + diff];
    }
#endif // __TEST__
    index = index % view_elem;

    int n_elem = this->num_elements;
    const float *ret = data;

    for (size_t i = 0; i < shape.size(); i++) {
        view_elem /= asshape[i + diff];
    #ifdef __TEST__
        if (shape[i] != asshape[i + diff] && shape[i] != 1) {
            throw std::invalid_argument("Shape and indices mismatch");
        }
    #endif // __TEST__

        if (shape[i] == 1) {
            continue;
        }

        const int idx = index / view_elem; 
        #ifdef __TEST__
        if (idx < 0 || idx >= shape[i]) {
            throw std::out_of_range("Index out of range");
        }
        #endif // __TEST__

        n_elem /= shape[i];
        /* ret += n_elem * indices[i + diff]; */
        ret += n_elem * idx;
        index /= asshape[i + diff];
    }

#ifdef __TEST__
    if (ret < data || ret >= data + num_elements) {
        throw std::out_of_range("Index out of range");
    }
#endif // __TEST__

    return ret;
}
