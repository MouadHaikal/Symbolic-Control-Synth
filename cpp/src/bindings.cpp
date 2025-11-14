#include "automaton.hpp"
#include "floodFill.hpp"


PYBIND11_MODULE(bindings, m) {
    py::class_<Automaton>(m, "Automaton")
        .def(py::init<py::object, py::object, py::object, const char*>());


    m.def(
        "floodFill",
        &floodFill,
        R"doc(
            Floods the available space to detect cells to be used.

            Takes the lowest corner of the space and performs a BFS until it reaches
            the upper corner, recording all traversed cells.

            N.B:
            Point is an alias to std::vector<double>.

            Args:
            lowerBound (Point): The starting point of the flood fill.
            upperBound (Point): The bounding corner of the flood fill.
            stepSize (Point): The step size along each dimension.

            Returns:
            set[Point]: The set of cell coordinates within the bounds.
        )doc",
        py::arg("lowerBound"),
        py::arg("upperBound")
    );
}
