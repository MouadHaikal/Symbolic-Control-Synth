#include "constants.hpp"
#include <cuda_runtime.h>


struct SpaceBoundsDevice{
    int    dimensions;
    float* dLowerBound;
    float* dUpperBound;
    float* dCenter;
};

struct SpaceInfoDevice{
    int    dimensions;
    int    cellCount;
    float* dLowerBound;
    float* dUpperBound;
    float* dCellSize;
    int*   dResolutions;
    
    __device__ __forceinline__
    void getCellBounds(int cellIdx, float* cellLowerBound, float* cellUpperBound) const {
        int resProd = cellCount;

        for(int i = dimensions - 1; i >= 0; i--) {
            resProd /= dResolutions[i];

            int idx = cellIdx / resProd;
            cellIdx -= idx * resProd;

            cellLowerBound[i] = dLowerBound[i] + dCellSize[i] * idx;
            cellUpperBound[i] = dLowerBound[i] + dCellSize[i] * (idx + 1);
        }
    } 


    __device__ __forceinline__
    void getCellCenter(int cellIdx, float* cellCenter) const {
        int resProd = cellCount;

        // i start with higher dimensions, meaning i need to decrement 
        for(int i = dimensions - 1; i >= 0; i--) {
            resProd /= dResolutions[i];

            int idx = cellIdx / resProd;
            cellIdx -= idx * resProd;

            cellCenter[i] = dLowerBound[i] + dCellSize[i] * (.5f + idx);
        }
    } 

    __device__ __forceinline__
    void getCellCoords(const float* point, int* coords) const {
        for(int i = 0; i < dimensions; i++) {
            if (point[i] < dLowerBound[i]) 
                coords[i] = 0;
            else{
                float normalized = (point[i] - dLowerBound[i]) / (dUpperBound[i] - dLowerBound[i]);

                int cand1 = normalized * dResolutions[i];
                int cand2 = dResolutions[i] - 1;
                coords[i] = (cand1 < cand2) ? cand1 : cand2; 
            }
        }
    }
};


struct TransitionTableDevice{
    int*         dData;
    int*         dRevData;
    int*         dTransitionLocks;
    const size_t stateCount;
    const size_t inputCount;
    

    __device__ __forceinline__ 
    int getOffset(int stateIdx, int inputIdx, int transition = 0) const {
        return stateIdx * (inputCount * MAX_TRANSITIONS) + 
               inputIdx * MAX_TRANSITIONS + 
               transition;
    }
    __device__ __forceinline__ 
    int getRevOffset(int stateIdx, int inputIdx, int predecessor = 0) const {
        return stateIdx * (inputCount * MAX_PREDECESSORS) + 
               inputIdx * MAX_PREDECESSORS + 
               predecessor;
    }
    __device__ __forceinline__ 
    void set(int stateIdx, int inputIdx, int transition, int val) {
        dData[getOffset(stateIdx, inputIdx, transition)] = val;
    }
    __device__ __forceinline__ 
    void setRev(int stateIdx, int inputIdx, int predecessor, int val) {
        dRevData[getRevOffset(stateIdx, inputIdx, predecessor)] = val;
    }
    __device__ __forceinline__ 
    int get(int stateIdx, int inputIdx, int transition) const {
        return dData[getOffset(stateIdx, inputIdx, transition)];
    }
    __device__ __forceinline__ 
    int getRev(int stateIdx, int inputIdx, int predecessor) const {
        return dRevData[getRevOffset(stateIdx, inputIdx, predecessor)];
    }
};

