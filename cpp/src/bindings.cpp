#include "automaton.hpp"


PYBIND11_MODULE(bindings, m) {
    py::class_<Automaton>(m, "Automaton")
        .def(py::init<py::object, py::object, py::object, bool, py::tuple, const char*>())

        .def("applySecuritySpec", &Automaton::applySecuritySpec, py::arg("pyObstacleLowerBound"), py::arg("pyObstacleUpperBound"))

        .def("applyReachabilitySpec", &Automaton::applyReachabilitySpec, py::arg("targetBounds"));
}
