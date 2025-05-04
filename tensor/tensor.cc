#include <pybind11/pybind11.h>
#include <pybind11/buffer_info.h>
#include <pybind11/stl.h>
#include "tensor.h" // Include the header file where Tensor is defined

namespace py = pybind11;
PYBIND11_MODULE(tensor_module, m) {
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("CUDA", Device::CUDA)
        .export_values();
    py::class_<Tensor>(m, "Tensor", pybind11::buffer_protocol())
        .def(py::init<std::vector<int>, Device>())
        .def_buffer(&Tensor::get_buffer_info)
        .def_readonly("shape", &Tensor::shape)
        .def_readonly("device", &Tensor::device)
        .def_readonly("num_elements", &Tensor::num_elements)
        .def_readonly("data", &Tensor::data)
        .def("__repr__", &Tensor::__repr__)
        .def("__add__", &Tensor::__add__, py::arg("other"))
        .def("__getitem__", &Tensor::__getitem__, py::arg("index"))
        .def("__setitem__", &Tensor::__setitem__, py::arg("index"), py::arg("value"))
        .def("__len__", &Tensor::__len__)
        .def("get_data", [](const Tensor& self) {
            // Return a pointer to the data for read-only access
            return self.data;
        })
        .def("to", &Tensor::to, py::arg("dev"))
		.def("to_list", &Tensor::to_list)
        .def_static("from_list", &Tensor::from_list, py::arg("list"))
		.def("to_numpy", &Tensor::to_numpy)
        .def_static("zeros", &Tensor::zeros, py::arg("shape"), py::arg("dev") = Device::CUDA)
        .def_static("from_numpy", &Tensor::from_numpy, py::arg("array"))
        .def_static("ones", &Tensor::ones, py::arg("shape"), py::arg("dev"))
        .def_static("dot", &Tensor::dot, py::arg("a"), py::arg("b"));
}
