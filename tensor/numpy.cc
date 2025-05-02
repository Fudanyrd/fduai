// convertion of Tensor to python list and numpy array
#include "tensor.h"
#include <cassert>

static PyObject *recursive_to_list(float *data, const std::vector<int> &shape, size_t dim, size_t num_elem) {
	if (dim == shape.size() - 1) {
		PyObject *list = PyList_New(shape[dim]);
		for (int i = 0; i < shape[dim]; ++i) {
			PyList_SetItem(list, i, PyFloat_FromDouble(data[i]));
		}
		return list;
	} else {
		PyObject *list = PyList_New(shape[dim]);
		num_elem /= shape[dim];
		for (int i = 0; i < shape[dim]; ++i) {
			PyObject *sublist = recursive_to_list(data + i * num_elem, shape, dim + 1, num_elem);
			PyList_SetItem(list, i, sublist);
		}
		return list;
	}
}

static void recursive_from_list(float *dst, PyObject *list, const std::vector<int> &shape, 
	size_t dim, size_t num_elem) {
	if (dim == shape.size() - 1) {
		for (int i = 0; i < shape[dim]; ++i) {
			PyObject *item = PyList_GetItem(list, i);
			if (PyFloat_Check(item)) {
				dst[i] = static_cast<float>(PyFloat_AsDouble(item));
			} else if (PyLong_Check(item)) {
				dst[i] = static_cast<float>(PyLong_AsLong(item));
			} else {
				throw std::runtime_error("Invalid item type in list");
			}
		}
	} else {
		num_elem /= shape[dim];
		for (int i = 0; i < shape[dim]; ++i) {
			PyObject *sublist = PyList_GetItem(list, i);
			recursive_from_list(dst + i * num_elem, sublist, shape, dim + 1, num_elem);
		}
	}
}

static void recursive_get_size(PyObject *list, std::vector<int> &shape) {
	if (PyList_Check(list)) {
		int size = PyList_Size(list);

		if (size == 0) {
			throw std::runtime_error("Empty list");
		}

		shape.push_back(size);
		for (int i = 0; i < size; ++i) {
			PyObject *item = PyList_GetItem(list, i);
			recursive_get_size(item, shape);
			break;
		}
	}
}

static void recursive_check_size(PyObject *list, const std::vector<int> &shape, size_t dim) {
	if (dim == shape.size() - 1) {
		int size = PyList_Size(list);
		if (size != shape[dim]) {
			throw std::runtime_error("List size does not match tensor shape");
		}
	} else {
		int size = PyList_Size(list);
		if (size != shape[dim]) {
			throw std::runtime_error("List size does not match tensor shape");
		}
		for (int i = 0; i < size; ++i) {
			PyObject *item = PyList_GetItem(list, i);
			recursive_check_size(item, shape, dim + 1);
		}
	}
}

pybind11::list Tensor::to_list() const {

    if (device != Device::CPU) {
        throw std::runtime_error("Tensor is not on CPU, cannot convert to numpy");
    }
    // FIXME: not implemented

	PyObject *ret; // = nullptr;
	ret = recursive_to_list(this->data, this->shape, 0, static_cast<size_t>(this->num_elements));

    return pybind11::reinterpret_borrow<pybind11::list>(ret);
}

Tensor Tensor::from_list(const pybind11::list &list) {
	// FIXME: not implemented
	PyObject *list_obj = list.ptr();
	if (!PyList_Check(list_obj)) {
		throw std::runtime_error("Input is not a list");
	}

	std::vector<int> shape;
	recursive_get_size(list_obj, shape);
	recursive_check_size(list_obj, shape, 0);

	Tensor tensor(shape, Device::CPU);
	recursive_from_list(tensor.data, list_obj, shape, 0, static_cast<size_t>(tensor.num_elements));

	return tensor;
}

PyObject *Tensor::to_numpy() const { 
	throw std::runtime_error("to_numpy() is not implemented");
}

pybind11::buffer_info Tensor::get_buffer_info() {
	//
	// https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html#buffer-protocol
	// Implementing the buffer protocol
	//

	return pybind11::buffer_info(
		this->data,                               /* Pointer to buffer */
		sizeof(float),                            /* Size of one scalar */
		pybind11::format_descriptor<float>::format(),   /* Python struct-style format descriptor */
		this->shape.size(),                       /* Number of dimensions */
		this->shape,                              /* Buffer dimensions */
		get_strides()                             /* Strides (in bytes) for each index */
	);
}

std::vector<size_t> Tensor::get_strides() const {
	//
	// row-major order
	//
	size_t stride = static_cast<size_t>(this->num_elements) * sizeof(float);
	std::vector<size_t> strides(this->shape.size(), 1);

	for (size_t i = 0; i < this->shape.size(); ++i) {
		stride /= this->shape[i];
		strides[i] = stride;
	}

	return strides;
}
