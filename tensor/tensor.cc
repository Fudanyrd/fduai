#include <pybind11/pybind11.h>
#include <pybind11/buffer_info.h>
#include <pybind11/stl.h>
#include "tensor.h" // Include the header file where Tensor is defined

//
// see 
// https://pybind11.readthedocs.io/en/stable/advanced/classes.html#operator-overloading
// for recommended way to overload operators
//
#include <pybind11/operators.h>

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
        .def(py::self + py::self)
        .def(py::self + float())
        .def(py::self - py::self)
        .def(py::self - float())
        .def(py::self * py::self)
        .def(py::self * float())
        .def(py::self / py::self)
        .def(py::self / float())
        .def("__neg__", &Tensor::__neg__)
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
        .def_static("dot", &Tensor::dot, py::arg("a"), py::arg("b"))
        .def_static("exp", &Tensor::exp, py::arg("a"))
        .def_static("transpose", &Tensor::transpose, py::arg("a"))
        .def_static("sum_all", &Tensor::sum_all, py::arg("a"))
        .def_static("sum", &Tensor::sum, py::arg("a"), py::arg("start_dim"))
        .def_static("max_all", &Tensor::sum_all, py::arg("a"))
        .def_static("max", &Tensor::max, py::arg("a"), py::arg("keep_dim"), py::arg("start_dim"));
}
