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


    void resolveSecuritySpec(bool* processed, int* hData, int* hRevData, int* roots, int size);

private:
    
    TransitionTableHost table;    
    void preProcessSecuritySpec(int* hData, int* hRevData, int* roots, int size);
};
