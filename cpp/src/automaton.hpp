#pragma once

#include <pybind11/embed.h>
#include <cuda.h>

#include "utilsHost.hpp"

namespace py = pybind11;


class Automaton{

public:
    Automaton(py::object stateSpace,        // DiscreteSpace  
              py::object inputSpace,        // DiscreteSpace
              py::object disturbanceSpace,  // ContinuousSpace
              bool isCooperative,
              py::tuple maxDisturbJac,
              const char* buildAutomatonCode);


    ~Automaton();

private:
    TransitionTableHost table;    
};
