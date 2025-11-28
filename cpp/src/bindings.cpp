#include "automaton.hpp"
#include <pybind11/stl.h> 


PYBIND11_MODULE(bindings, m) {
    py::class_<Automaton>(m, "Automaton")
        .def(py::init<py::object, py::object, py::object, bool, py::tuple, const char*>())

        .def("applySecuritySpec", &Automaton::applySecuritySpec, py::arg("pyObstacleLowerBoundCoords"), py::arg("pyObstacleUpperBoundCoords"))

        .def("getController", &Automaton::getController, py::arg("startState"), py::arg("pyTargetLowerBoundCoords"), py::arg("pyTargetUpperBoundCoords"));

}
