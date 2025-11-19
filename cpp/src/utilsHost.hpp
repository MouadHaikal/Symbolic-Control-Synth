#include "utilsDevice.hpp"



struct SpaceBoundsHost{
    int    dimensions;
    float* dLowerBound;
    float* dUpperBound;

    SpaceBoundsHost(int dimensions, float* hLowerBound, float* hUpperBound)
        : dimensions(dimensions) 
    {
        cudaMalloc(&dLowerBound, dimensions * sizeof(float));
        cudaMalloc(&dUpperBound, dimensions * sizeof(float));

        cudaMemcpy(dLowerBound, hLowerBound, dimensions * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dUpperBound, hUpperBound, dimensions * sizeof(float), cudaMemcpyHostToDevice);
    }

    ~SpaceBoundsHost() {
        cudaFree(dLowerBound);
        cudaFree(dUpperBound);
    }
};


struct SpaceInfoHost{
    int    dimensions;
    float* dLowerBound;
    float* dUpperBound;
    float* dCellSize;
    int*   dResolutions;
    int    cellCount;
    
    SpaceInfoHost(int dimensions, float* hLowerBound, float* hUpperBound, int* hResolutions, float* hCellSize, int cellCount)
        : dimensions(dimensions), cellCount(cellCount) 
    {
        cudaMalloc(&dResolutions, dimensions * sizeof(int));
        cudaMalloc(&dCellSize, dimensions * sizeof(float));
        cudaMalloc(&dLowerBound, dimensions * sizeof(float));
        cudaMalloc(&dUpperBound, dimensions * sizeof(float));

        cudaMemcpy(dResolutions, hResolutions, dimensions * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dCellSize, hCellSize, dimensions * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dLowerBound, hLowerBound, dimensions * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dUpperBound, hUpperBound, dimensions * sizeof(float), cudaMemcpyHostToDevice);
    }

    ~SpaceInfoHost() {
        cudaFree(dResolutions);
        cudaFree(dCellSize);
        cudaFree(dLowerBound);
        cudaFree(dUpperBound);
    }
};


struct TransitionTableHost{
    int*         dData;
    int*         dRevData;
    int*         dTransitionLocks;
    const size_t stateCount;
    const size_t inputCount;

    
    TransitionTableHost(size_t stateCount, size_t inputCount) 
        : stateCount(stateCount), inputCount(inputCount) 
    {

        size_t size = stateCount * inputCount * MAX_TRANSITIONS * sizeof(int);
        cudaMalloc(&dData, size);
        cudaMemset(dData, EMPTY_CELL, size);

        size_t sizeRev = stateCount * inputCount * MAX_PREDECESSORS * sizeof(int);
        cudaMalloc(&dRevData, sizeRev);
        cudaMemset(dRevData, EMPTY_CELL, sizeRev);

        cudaMalloc(&dTransitionLocks, stateCount * inputCount * sizeof(int));
        cudaMemset(dTransitionLocks, 0, stateCount * inputCount * sizeof(int));
    }
    
    ~TransitionTableHost() {
        cudaFree(dData);
        cudaFree(dRevData);
    }

    int getOffset(int stateIdx, int inputIdx, int transition = 0) const {
        return stateIdx * (inputCount * MAX_TRANSITIONS) + 
               inputIdx * MAX_TRANSITIONS + 
               transition;
    }
    int getRevOffset(int stateIdx, int inputIdx, int predecessor = 0) const {
        return stateIdx * (inputCount * MAX_PREDECESSORS) + 
               inputIdx * MAX_PREDECESSORS + 
               predecessor;
    }
    void set(int stateIdx, int inputIdx, int transition, int val) {
        dData[getOffset(stateIdx, inputIdx, transition)] = val;
    }
    void setRev(int stateIdx, int inputIdx, int predecessor, int val) {
        dRevData[getRevOffset(stateIdx, inputIdx, predecessor)] = val;
    }
    int get(int stateIdx, int inputIdx, int transition) const {
        return dData[getOffset(stateIdx, inputIdx, transition)];
    }
    int getRev(int stateIdx, int inputIdx, int predecessor) const {
        return dRevData[getRevOffset(stateIdx, inputIdx, predecessor)];
    }
};
