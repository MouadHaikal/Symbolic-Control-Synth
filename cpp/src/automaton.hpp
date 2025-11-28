#pragma once

#include <pybind11/embed.h>
#include <cuda.h>

#include "utilsHost.hpp"

namespace py = pybind11;


class Automaton{

public:
    Automaton(py::object  stateSpace,        // DiscreteSpace  
              py::object  inputSpace,        // DiscreteSpace
              py::object  disturbanceSpace,  // ContinuousSpace
              bool        isCooperative,
              py::tuple   maxDisturbJac,
              const char* buildAutomatonCode);


    void applySecuritySpec(py::tuple pyObstacleLowerBoundCoords, py::tuple pyObstacleUpperBoundCoords);
    std::vector<int> getController(int startState, py::tuple pyTargetLowerBoundCoords, py::tuple pyTargetUpperBoundCoords);

    
private:
    TransitionTableHost table;
    const int           stateDim;
    std::vector<int>    resolutionStride;
    
private:
    std::vector<int> applyReachabilitySpec(py::tuple pyTargetLowerBoundCoords, py::tuple pyTargetUpperBoundCoords);
    std::vector<int> floodFill(const std::vector<int>&  lowerBoundCoords, const std::vector<int>&  upperBoundCoords);

    inline void validateDimension(const py::object& space);
};
