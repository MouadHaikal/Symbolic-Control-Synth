#include "automaton.hpp"
#include <iostream>


// py::scoped_interpreter guard{};
py::object DiscreteSpace = py::module_::import("symControl.space.discreteSpace").attr("DiscreteSpace");
py::object ContinuousSpace = py::module_::import("symControl.space.discreteSpace").attr("ContinuousSpace"); 



Automaton::Automaton(py::object stateSpace,       // DiscreteSpace
                     py::object controlSpace,     // DiscreteSpace
                     py::object disturbanceSpace, // ContinuousSpace
                     const char* fAtPointCode)
{
    printf("Initialisation succesful\n");
    std::cout << stateSpace.attr("dimensions").cast<int>() << std::endl;
}
