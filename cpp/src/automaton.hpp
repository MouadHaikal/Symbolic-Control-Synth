#pragma once

#include <pybind11/embed.h>

namespace py = pybind11;


class Automaton{
public:
    Automaton(py::object stateSpace, py::object controlSpace, py::object disturbanceSpace, const char* fAtPointCode);
};
