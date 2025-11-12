#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <model_bindings.hpp>

namespace py = pybind11;

// "floodfill" is the variable representing the module object in C++
PYBIND11_MODULE(ExternalModules, floodfill) {
    floodfill.doc() = "Flood fill bindings";

    floodfill.def(
        "floodFill",
        &floodFill,  // this must match your C++ function signature
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
