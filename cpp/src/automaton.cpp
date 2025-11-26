#include "automaton.hpp"
#include "utilsHost.hpp"

#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <nvrtc.h>
#include <cstddef>
#include <chrono>
#include <queue>


Automaton::Automaton(py::object stateSpace,       // DiscreteSpace
                     py::object inputSpace,       // DiscreteSpace
                     py::object disturbanceSpace, // ContinuousSpace
                     bool isCooperative,
                     py::tuple maxDisturbJac,
                     const char* buildAutomatonCode)
    : table(stateSpace.attr("cellCount").cast<size_t>(), inputSpace.attr("cellCount").cast<size_t>())
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

    printf("Applying security specification\n");
    auto start = std::chrono::high_resolution_clock::now();
    int* hData = new int[table.stateCount * table.inputCount * MAX_TRANSITIONS];
    int* hRevData = new int[table.stateCount * table.inputCount * MAX_PREDECESSORS];
    bool* processed = new bool[table.stateCount];
    // processed[i] = false means it isn't processed yet.
    // processed[i] = true means it is processed
    // at the end of the algorithm all the nodes (marked true) are considered removed from the transition table of the automaton.

    cudaMemcpy(hData, 
               table.dData, 
               table.stateCount * table.inputCount * MAX_TRANSITIONS * sizeof(int), 
               cudaMemcpyDeviceToHost);

    cudaMemcpy(hRevData, 
               table.dRevData, 
               table.stateCount * table.inputCount * MAX_PREDECESSORS * sizeof(int), 
               cudaMemcpyDeviceToHost);

    for(int i = 0; i < table.stateCount; i++) processed[i] = false;

    int roots[2] = {5, 6};
    int size = 2;
    resolveSecuritySpec(processed, hData, hRevData, roots, size);



    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("Time to apply security specification: %.5f ms.\n", ms);

    printf("Applying reachibility specification\n");
    int startState = 0;
    int dimensions = 2;
    int targets[3] = {0, 1, 2};
    int target_size = 3;

    std::vector<int> controller = getController(hData, hRevData, startState, dimensions, targets, target_size);
    printf("controller is: \n");
    for(int &x : controller) printf("%d ", x);
    printf("\n");




    // =============== Testing ===============
    printf("=== Combined Direct + Reverse Table ===\n");


    for (int s = 0; s < table.stateCount; s++) {
        for (int u = 0; u < table.inputCount; u++) {

            printf("\n==============================\n");
            printf("     State %d | Input %d\n", s, u);
            printf("==============================\n");

            int baseDir = (s * table.inputCount + u) * MAX_TRANSITIONS;
            int baseRev = (s * table.inputCount + u) * MAX_PREDECESSORS;

            // ---- DIRECT ----
            printf("Direct : ");
            bool dirEmpty = true;
            for (int t = 0; t < MAX_TRANSITIONS; t++) {
                int v = hData[baseDir + t];
                if (v != -1) dirEmpty = false;
            }
            if (dirEmpty) {
                printf("[ none ]\n");
            } else {
                for (int t = 0; t < MAX_TRANSITIONS; t++) {
                    int v = hData[baseDir + t];
                    if (v != -1) printf("%d ", v);
                }
                printf("\n");
            }

            // ---- REVERSE ----
            printf("Reverse: ");
            bool revEmpty = true;
            for (int p = 0; p < MAX_PREDECESSORS; p++) {
                int v = hRevData[baseRev + p];
                if (v != -1) revEmpty = false;
            }
            if (revEmpty) {
                printf("[ none ]\n");
            } else {
                for (int p = 0; p < MAX_PREDECESSORS; p++) {
                    int v = hRevData[baseRev + p];
                    if (v != -1) printf("%d ", v);
                }
                printf("\n");
            }

            printf("-----------------------------------\n");
        }
    }

    printf("\n=== End Combined Table ===\n");

    // ============== Cleanup ================
    delete[] hData;
    delete[] hRevData;
    delete[] processed;
}

Automaton::~Automaton() {

}

void Automaton::preProcessSecuritySpec(int* hData, int* hRevData, int* roots, int size) {
    for(int i = 0; i < size; i++) {
        int stateIdx = roots[i];
        // remove the root from the graph
        for(int inputIdx = 0; inputIdx < table.inputCount; inputIdx++) {
            // printf("[Preprocess] Removing outgoing transitions from root %d on input %d\n",
               // stateIdx, inputIdx);    

            for(int offset = 0; offset < MAX_TRANSITIONS; offset++) {

                int globalOffset = table.getOffset(stateIdx, inputIdx, offset);

                // printf("    [Preprocess]   setting hData[%d] = -1\n", off);

                int otherSide = hData[globalOffset];
                if(otherSide != -1) {
                    // removing the reverse(otherSide, inputIdx) from the Reverse
                    for(int revOffset = 0; revOffset < MAX_PREDECESSORS; revOffset++) {
                        int globalRevOffset = table.getRevOffset(otherSide, inputIdx, revOffset);
                        if(hRevData[globalRevOffset] == stateIdx) {hRevData[globalRevOffset] = -1; break;}

                    }
                }

                hData[globalOffset] = -1;
            }
        }
    }
    return;
}

void Automaton::resolveSecuritySpec(bool* processed,
                                    int* hData,
                                    int* hRevData,
                                    int* roots, // vector of the roots array to be removed
                                    int size) // size of the roots array
{
    if(size <= 0) return; // nothing to remove

    preProcessSecuritySpec(hData, hRevData, roots, size);

    for(int i = 0; i < size; i++) {
        int stateIdx = roots[i];
        if(processed[stateIdx]) continue;
        processed[stateIdx] = true;

        // printf("\n[Resolve] Processing state to remove: %d\n", stateIdx);

        // Initialize the to_check array.
        int* toRemove = new int[table.stateCount + 1];
        int toRemoveCount = 0;

        for(int inputIdx = 0; inputIdx < table.inputCount; inputIdx++) {
            for(int revOffset = 0; revOffset < MAX_PREDECESSORS; revOffset++) {

                int parent = hRevData[table.getRevOffset(stateIdx, inputIdx, revOffset)];

                // printf("[Resolve]   Checking reverse edge: parent=%d via input=%d revSlot=%d\n",
                       // parent, inputIdx, revOffset);
                if(parent == -1 || (parent != -1 && processed[parent])) continue;

                // delete all the transitions (parent, inputIdx)
                for(int transition = 0; transition < MAX_TRANSITIONS; transition++) {
                    // printf("    [Delete]   Removing (parent=%d, input=%d) transition slot %d\n",
                           // parent, inputIdx, transition);
                    hData[table.getOffset(parent, inputIdx, transition)] = -1;
                    // TODO: check if the transitions removed should also be removed from hRevData
                }
                hRevData[table.getRevOffset(stateIdx, inputIdx, revOffset)] = -1;

                // we need to check if the parent still has any outgoing edges
                bool toRemoveFlag = true;
                for(int parentInputIdx = 0; parentInputIdx < table.inputCount && toRemoveFlag; parentInputIdx++) {
                    for(int parentOffset = 0; parentOffset < MAX_TRANSITIONS && toRemoveFlag; parentOffset++) {
                        if(hData[table.getOffset(parent, parentInputIdx, parentOffset)] != -1) {
                            toRemoveFlag = false;
                        }
                    }
                }
                if(toRemoveFlag)
                    toRemove[toRemoveCount++] = parent; 
                // the parent is should be removed and the state is removed in itself;
            }
        }

        resolveSecuritySpec(processed, hData, hRevData, toRemove, toRemoveCount);
        delete[] toRemove;
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

