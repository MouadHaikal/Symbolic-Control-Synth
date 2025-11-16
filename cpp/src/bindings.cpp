#include "automaton.hpp"


PYBIND11_MODULE(bindings, m) {
    py::class_<Automaton>(m, "Automaton")
        .def(py::init<py::object, py::object, py::object, const char*>());
}
