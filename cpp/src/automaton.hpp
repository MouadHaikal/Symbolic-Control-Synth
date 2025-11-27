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


    void applySecuritySpec(py::tuple pyObstacleLowerBound, py::tuple pyObstacleUpperBound);
    void applyReachabilitySpec(py::object targetBounds);

    
private:
    TransitionTableHost table;
    const int stateDim;
    std::vector<int> resolutionStride;
    
private:
    void removeUnsafeStates(const std::vector<int>& obstacleCells);

    std::vector<int> getController(int* hData, int* hRevData, int startState, int dimensions, int* targets, int target_size);
    float getDistance(int state, int otherState, int dimension);

    std::vector<int> floodFill(const std::vector<int>&  lowerBoundCoords, const std::vector<int>&  upperBoundCoords);
    inline void validateDimension(const py::object& space);

};
