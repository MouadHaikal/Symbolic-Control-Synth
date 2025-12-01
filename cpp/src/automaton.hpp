#pragma once

#include <pybind11/embed.h>
#include <cuda.h>

#include "utilsHost.hpp"

namespace py = pybind11;

struct XORTransitionTable;

class Automaton
{

public:
    std::vector<int> resolutionStride;
    TransitionTableHost table;
    Automaton(py::object stateSpace,       // DiscreteSpace
              py::object inputSpace,       // DiscreteSpace
              py::object disturbanceSpace, // ContinuousSpace
              bool isCooperative,
              py::tuple maxDisturbJac,
              const char *buildAutomatonCode);

    void applySecuritySpec(py::tuple pyObstacleLowerBoundCoords, py::tuple pyObstacleUpperBoundCoords);
    std::vector<std::vector<int>> getController(py::tuple pyStartStateCoords,
                                                py::tuple pyTargetLowerBoundCoords,
                                                py::tuple pyTargetUpperBoundCoords,
                                                int pathCount);
    std::vector<std::vector<int>> getXORController(py::tuple pyStartStateCoords,
                                                   py::tuple pyFirstLowerBound,
                                                   py::tuple pyFirstUpperBound,
                                                   py::tuple pySecondLowerBound,
                                                   py::tuple pySecondUpperBound,
                                                   py::tuple pyTargetLowerBound,
                                                   py::tuple pyTargetUpperBound,
                                                   int pathCount);

private:
    const int stateDim;
    std::vector<int> controller;

private:
    std::vector<int> getRandomPath(int startStateIdx) const;
    std::vector<int> getRandomXORPath(int startStateIdx, const XORTransitionTable &xorTable) const;

    void applyReachabilitySpec(py::tuple pyTargetLowerBoundCoords, py::tuple pyTargetUpperBoundCoords);
    std::vector<int> floodFill(const std::vector<int> &lowerBoundCoords, const std::vector<int> &upperBoundCoords);

    inline void validateDimension(const py::object &space);
};
