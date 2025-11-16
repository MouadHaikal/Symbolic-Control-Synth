#pragma once

#include <pybind11/embed.h>
#include <cuda.h>

#include "utilsHost.hpp"

namespace py = pybind11;


class Automaton{

public:
    Automaton(py::object stateSpace,        // DiscreteSpace  
              py::object inputSpace,        // DiscreteSpace
              py::object disturbanceSpace,  // ContinuousSpac
              const char* fAtPointCode);


    ~Automaton();

private:
    
    TransitionTableHost table;    
};
