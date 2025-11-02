#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include "../header/LSHSearch.hpp"
#include "../header/VectorRecord.hpp"

namespace py = pybind11;
void bind_search(py::module_ &m)
{
    py::class_<LSHSearch>(m, "LSHSearch")
        .def(py::init<>())
        .def_static("jarcardDistance", &LSHSearch::jarcardDistance)
        .def_static("hammingDistance", &LSHSearch::hammingDistance)
        .def("classifyByBand", &LSHSearch::classifyByBand)
        .def("classify", &LSHSearch::classify)
        .def("setDisFunc", &LSHSearch::setDisFunc, py::arg("nameDisFunc"))

        .def_readwrite("disFunc", &LSHSearch::disFunc)
        .def_readwrite("num_bands", &LSHSearch::num_bands)
        .def_readwrite("threshold", &LSHSearch::threshold);
}