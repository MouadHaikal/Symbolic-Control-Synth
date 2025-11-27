#include "automaton.hpp"
#include "utilsHost.hpp"

#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <nvrtc.h>
#include <cstddef>
#include <queue>


Automaton::Automaton(py::object stateSpace,       // DiscreteSpace
                     py::object inputSpace,       // DiscreteSpace
                     py::object disturbanceSpace, // ContinuousSpace
                     bool isCooperative,
                     py::tuple maxDisturbJac,
                     const char* buildAutomatonCode)
    : table(stateSpace.attr("cellCount").cast<size_t>(), inputSpace.attr("cellCount").cast<size_t>()),
        stateDim(stateSpace.attr("dimensions").cast<int>())
{ 
    validateDimension(stateSpace);
    validateDimension(inputSpace);
    validateDimension(disturbanceSpace);

    CudaCompilationContext compilationContext{buildAutomatonCode, isCooperative};

    compilationContext.launchKernel(stateSpace, 
                                    inputSpace, 
                                    disturbanceSpace, 
                                    maxDisturbJac, 
                                    table
    );

    compilationContext.cleanup();

    table.syncDeviceData();

    resolutionStride.reserve(stateDim);
    resolutionStride[0] = 1;
    py::tuple pyResolutions = stateSpace.attr("resolutions").cast<py::tuple>();
    for (int i = 1; i < stateDim; ++i)
        resolutionStride[i] = pyResolutions[i-1].cast<int>() * resolutionStride[i-1];

}

Automaton::~Automaton() {

}

void Automaton::applySecuritySpec(py::tuple pyObstacleLowerBound, py::tuple pyObstacleUpperBound){
    std::vector<int> obstacleLowerBound = std::vector<int>(stateDim);
    std::vector<int> obstacleUpperBound = std::vector<int>(stateDim);

    for(int dim = 0; dim < stateDim; dim++){
        obstacleLowerBound[dim] = pyObstacleLowerBound[dim].cast<int>();
        obstacleUpperBound[dim] = pyObstacleUpperBound[dim].cast<int>();
    }
    std::vector<int> obstacleCells = floodFill(obstacleLowerBound, obstacleUpperBound);
    removeUnsafeStates(obstacleCells);


}


void Automaton::removeUnsafeStates(const std::vector<int>& obstacleCells){
    std::queue<int> toRemove;
    for(const int idx : obstacleCells) {
        toRemove.push(idx);
    }
    while(!toRemove.empty()) {
        int stateIdx = toRemove.front();
        toRemove.pop();
        for(int inputIdx = 0; inputIdx < table.inputCount; inputIdx++) {
            for(int revOff = table.getOffset(stateIdx, inputIdx), _ = 0; _ < MAX_PREDECESSORS; _++, revOff++) {
                int parIdx = table.hRevData[revOff]; 
                if(parIdx == -1) continue;
                table.removeTransitions(parIdx, inputIdx);
                
                bool toRemoveFlag = true;
                for(int i = table.getOffset(parIdx, 0), _ = 0; _ < table.inputCount*MAX_TRANSITIONS && toRemoveFlag; _++, i++) {
                    if(table.hData[i] != -1) {
                        toRemoveFlag = false;
                    }
                }

                if(toRemoveFlag) {
                    toRemove.push(parIdx);
                }
            }
        }
    }
    
}


/* resolveReachabilitySpec --> from a startState, how can we reach the targets?
 * Always call this function after pruning all undesirable states
 * this takes as input the direct graph and its reverse, the target states we would like to reach.
 * out represent the series of inputs to provide for the automaton
 * the algorithm will use Djkstra's algorithm to now what is the minimal distance from a start state and the targets
 * from that we will consider the target with the least cumulative weight and find the path leading to it.
 * the weights are represented by default as 1 between nodes with the helper function float table::getDistance(int state, int otherState) but can be changed to any other formula as long as the weights are positive.
 * */
std::vector<int> Automaton::getController(int* hData,
                                        int* hRevData,
                                        int startState,
                                        int dimensions, // dimension of state space
                                        int* targets,
                                        int target_size)
{
    const float INF = 1e30f;
    // total distance array
    float *dist = new float[table.stateCount];
    for(int i = 0; i < table.stateCount; i++) dist[i] = INF;

    // backtracking purposes.
    int *prevState = new int[table.stateCount];
    int *prevInput = new int[table.stateCount];

    using P = std::pair<float, int>;
    std::priority_queue<P, std::vector<P>, std::greater<P>> pq;

    dist[startState] = 0.0f;
    prevState[startState] = -1; // means we don't have any route to take
    prevInput[startState] = -1;
    pq.push({0.0f, startState});

    while(!pq.empty()) {
        auto [d, currentState] = pq.top();
        pq.pop();
        if(d > dist[currentState]) continue;

        // checking the neighbors
        for(int inputIdx = 0; inputIdx < table.inputCount; inputIdx++) {
            for(int offset = 0; offset < MAX_TRANSITIONS; offset++) {
              
                int nextState = hData[table.getOffset(currentState, inputIdx, offset)];
                if(nextState == -1) continue;

                int newDistance = d + getDistance(currentState, nextState, dimensions);

                if(newDistance < dist[nextState]) {
                    dist[nextState] = newDistance;
                    prevState[nextState] = currentState;
                    prevInput[nextState] = inputIdx;
                    pq.push({dist[nextState], nextState});
                }
            }
        }
    }
    

    // get the least expensive path
    int bestTarget = -1;
    float bestDistance = INF;
    for(int i = 0; i < target_size; i++) {
        int target = targets[i];
        if(dist[target] < bestDistance) bestDistance = dist[target], bestTarget = target;
    }

    // get path from bestTarget to the startState
    std::vector<int> inputsRev;
    if (bestTarget == -1) {
        printf("[ERR] No target reachable\n");
        return inputsRev;
    }

    // Step 4: Reconstruct path (backwards)
    int cur = bestTarget;
    while (cur != startState && cur != -1) {
        inputsRev.push_back(prevInput[cur]);
        cur = prevState[cur];
    }
    reverse(inputsRev.begin(), inputsRev.end());
    return inputsRev;

}

// TODO: andirouh db
void Automaton::applyReachabilitySpec(py::object targetBounds) {
    return;
}

float Automaton::getDistance(int state, int otherState, int dimensions) {
    return 1.0f;
}


inline void Automaton::validateDimension(const py::object& space) {
    if (space.attr("dimensions").cast<int>() > MAX_DIMENSIONS) {
        printf("Error: Space dimensions (%d) > MAX_DIMENSIONS (%d) (at %s)\n", 
               space.attr("dimensions").cast<int>(), 
               MAX_DIMENSIONS,
               __FILE__
        );

        exit(EXIT_FAILURE);
    }
}

std::vector<int> Automaton::floodFill(const std::vector<int>&  lowerBoundCoords, const std::vector<int>&  upperBoundCoords){
    std::vector<int> curCoords{stateDim};
    for (int i = 0; i < stateDim; ++i) 
        curCoords[i] = lowerBoundCoords[i];

    std::vector<int> out;

    while (true) {
        int cellIdx = 0;
        for (int i = 0; i < stateDim; ++i) 
            cellIdx += curCoords[i] * resolutionStride[i];
        
        out.push_back(cellIdx);
        
        int d = 0;
        while (d <= stateDim - 1) {
            curCoords[d]++;

            if (curCoords[d] <= upperBoundCoords[d]) break;

            curCoords[d] = lowerBoundCoords[d];
            d++;
        }

        if (d >= stateDim) break;
    }

    return out;
}
