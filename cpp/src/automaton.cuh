#pragma once

#include <pybind11/embed.h>
#include <cuda.h>
// #include <cuco/static_map.cuh>

namespace py = pybind11;


class Automaton{

public:
    Automaton(py::object stateSpace, py::object controlSpace, py::object disturbanceSpace, const char* fAtPointCode);

    ~Automaton();

private:
    CUdevice cuDevice;
    CUcontext context;
    CUmodule module;
    CUfunction testKernel;
};
