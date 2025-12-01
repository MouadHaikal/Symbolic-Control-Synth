#include "automaton.hpp"
#include "utilsDevice.hpp"
#include "utilsHost.hpp"
#include "utilsMisc.hpp"

#include <cstdio>
#include <cstdlib>
#include <nvrtc.h>
#include <cstddef>
#include <queue>
#include <fstream>
#include <iomanip>
#include <unordered_set>
#include <vector>
#include <random>

Automaton::Automaton(py::object stateSpace,       // DiscreteSpace
                     py::object inputSpace,       // DiscreteSpace
                     py::object disturbanceSpace, // ContinuousSpace
                     bool isCooperative,
                     py::tuple maxDisturbJac,
                     const char *buildAutomatonCode)
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
                                    table);

    compilationContext.cleanup();

    table.syncDeviceData();

    resolutionStride.reserve(stateDim);
    resolutionStride[0] = 1;
    py::tuple pyResolutions = stateSpace.attr("resolutions").cast<py::tuple>();
    for (int i = 1; i < stateDim; ++i)
        resolutionStride[i] = pyResolutions[i - 1].cast<int>() * resolutionStride[i - 1];

    table.precomputeTransitions();

    std::string logFilePath = "table.log";
    std::ofstream logFile(logFilePath);

    if (!logFile.is_open())
    {
        std::cerr << "Error: Could not open log file at " << logFilePath << std::endl;
        return;
    }

    logFile << "\n"
            << std::string(80, '=') << "\n";
    logFile << "TRANSITION TABLES SUMMARY\n";
    logFile << std::string(80, '=') << "\n\n";

    // ============== FORWARD TRANSITIONS (hData) ==============
    logFile << "FORWARD TRANSITIONS (hData)\n";
    logFile << "- Represents: For each (state, input) pair, which states can be reached\n";
    logFile << std::string(80, '-') << "\n\n";

    for (int state = 0; state < table.stateCount; ++state)
    {
        for (int input = 0; input < table.inputCount; ++input)
        {
            // Calculate the block index in the flattened array
            // Block structure: [state0-input0 | state0-input1 | ... | stateN-inputM]
            int blockIndex = (state * table.inputCount + input) * MAX_TRANSITIONS;

            logFile << "State " << std::setw(3) << state
                    << " | Input " << std::setw(3) << input
                    << "   ==>   : ";

            // Iterate through the block and collect non-negative values
            std::vector<int> destinations;
            for (int i = 0; i < MAX_TRANSITIONS; ++i)
            {
                int destState = table.hData[blockIndex + i];
                if (destState >= 0)
                { // Skip -1 entries (no transition)
                    destinations.push_back(destState);
                }
            }

            // Print destinations or indicate no transitions
            if (destinations.empty())
            {
                logFile << "[ None ]\n";
            }
            else
            {
                logFile << "[ ";
                for (size_t i = 0; i < destinations.size(); ++i)
                {
                    logFile << destinations[i];
                    if (i < destinations.size() - 1)
                    {
                        logFile << ", ";
                    }
                }
                logFile << " ]\n";
            }
        }
    }

    logFile << "\n";

    // ============== REVERSE TRANSITIONS (hRevData) ==============
    logFile << "REVERSE TRANSITIONS (hRevData)\n";
    logFile << "- Represents: For each (state, input) pair, which states can reach it\n";
    logFile << std::string(80, '-') << "\n\n";

    for (int state = 0; state < table.stateCount; ++state)
    {
        for (int input = 0; input < table.inputCount; ++input)
        {
            // Calculate the block index in the flattened array
            // Same structure as hData but with predecessor counts
            int blockIndex = (state * table.inputCount + input) * MAX_PREDECESSORS;

            logFile << "State " << std::setw(3) << state
                    << " | Input " << std::setw(3) << input
                    << "   <==   : ";

            // Iterate through the block and collect non-negative values
            std::vector<int> predecessors;
            for (int i = 0; i < MAX_PREDECESSORS; ++i)
            {
                int srcState = table.hRevData[blockIndex + i];
                if (srcState >= 0)
                { // Skip -1 entries (no transition)
                    predecessors.push_back(srcState);
                }
            }

            // Print predecessors or indicate no transitions
            if (predecessors.empty())
            {
                logFile << "[ None ]\n";
            }
            else
            {
                logFile << "[ ";
                for (size_t i = 0; i < predecessors.size(); ++i)
                {
                    logFile << predecessors[i];
                    if (i < predecessors.size() - 1)
                    {
                        logFile << ", ";
                    }
                }
                logFile << " ]\n";
            }
        }
    }

    logFile << "\n"
            << std::string(80, '=') << "\n\n";
    logFile.flush();
    logFile.close();

    std::cout << "Transition tables printed to: " << logFilePath << std::endl;
}

void Automaton::applySecuritySpec(py::tuple pyObstacleLowerBoundCoords, py::tuple pyObstacleUpperBoundCoords)
{
    std::vector<int> obstacleLowerBoundCoords = std::vector<int>(stateDim);
    std::vector<int> obstacleUpperBoundCoords = std::vector<int>(stateDim);

    for (int dim = 0; dim < stateDim; dim++) {
        obstacleLowerBoundCoords[dim] = pyObstacleLowerBoundCoords[dim].cast<int>();
        obstacleUpperBoundCoords[dim] = pyObstacleUpperBoundCoords[dim].cast<int>();
    }
    std::vector<int> obstacleCells = floodFill(obstacleLowerBoundCoords, obstacleUpperBoundCoords);

    // Remove unsafe states
    std::queue<int> toRemove;
    std::unordered_set<int> removed;
    for (const int idx : obstacleCells)
        toRemove.push(idx);

    while (!toRemove.empty())
    {
        int stateIdx = toRemove.front();
        toRemove.pop();
        if (removed.count(stateIdx))
            continue;
        removed.insert(stateIdx);

        for (int inputIdx = 0; inputIdx < table.inputCount; inputIdx++)
        {
            table.removeTransitions(stateIdx, inputIdx);

            for (int pred = 0; pred < MAX_PREDECESSORS; pred++)
            {
                int parIdx = table.getRev(stateIdx, inputIdx, pred);
                if (parIdx == -1)
                    continue;
                table.removeTransitions(parIdx, inputIdx);

                bool toRemoveFlag = true;
                for (int i = table.getOffset(parIdx, 0), _ = 0; _ < table.inputCount * MAX_TRANSITIONS && toRemoveFlag; _++, i++)
                {
                    if (table.hData[i] != -1)
                        toRemoveFlag = false;
                }

                if (toRemoveFlag)
                    toRemove.push(parIdx);
            }
        }
    }

    // it is better to precompute transitions at the end of each security spec application
    table.precomputeTransitions();
}

std::vector<std::vector<int>> Automaton::getController(py::tuple pyStartStateCoords,
                                                       py::tuple pyTargetLowerBoundCoords,
                                                       py::tuple pyTargetUpperBoundCoords,
                                                       int pathCount)
{
    applyReachabilitySpec(pyTargetLowerBoundCoords, pyTargetUpperBoundCoords);

    std::ofstream logFile("controller.log");

    for (int i = 0; i < table.stateCount; i++)
    {
        logFile << i << " " << table.safeStates[i] << " " << controller[i] << std::endl;
    }

    logFile.flush();
    logFile.close();

    int startStateIdx = 0;
    for (int i = 0; i < stateDim; ++i)
        startStateIdx += pyStartStateCoords[i].cast<int>() * resolutionStride[i];

    if (table.safeStates[startStateIdx] == -1)
    {
        printf("No path found\n");
        return std::vector({std::vector({-1})});
    }

    std::vector<std::vector<int>> statePaths;

    for (int _ = 0; _ < pathCount; _++)
    {
        statePaths.push_back(getRandomPath(startStateIdx));
    }

    return statePaths;
}

void Automaton::applyReachabilitySpec(py::tuple pyTargetLowerBoundCoords, py::tuple pyTargetUpperBoundCoords)
{
    std::vector<int> targetLowerBoundCoords = std::vector<int>(stateDim);
    std::vector<int> targetUpperBoundCoords = std::vector<int>(stateDim);

    for (int dim = 0; dim < stateDim; dim++)
    {
        targetLowerBoundCoords[dim] = pyTargetLowerBoundCoords[dim].cast<int>();
        targetUpperBoundCoords[dim] = pyTargetUpperBoundCoords[dim].cast<int>();
    }
    std::vector<int> targetCells = floodFill(targetLowerBoundCoords, targetUpperBoundCoords);
    memset(table.safeStates, -1, table.stateCount * sizeof(int));

    controller.reserve(table.stateCount);
    memset(controller.data(), -1, table.stateCount * sizeof(int));

    std::queue<int> pendingStates;
    for (int target : targetCells)
    {
        pendingStates.push(target);
        table.safeStates[target] = 0;
    }

    while (!pendingStates.empty())
    {
        int stateIdx = pendingStates.front();
        pendingStates.pop();

        for (int inputIdx = 0; inputIdx < table.inputCount; inputIdx++)
        {
            for (int pred = 0; pred < MAX_PREDECESSORS; pred++)
            {
                int predIdx = table.getRev(stateIdx, inputIdx, pred);
                if (predIdx != -1 && table.safeStates[predIdx] == -1)
                {
                    if ((++table.safeCounts[table.getPosition(predIdx, inputIdx)]) == table.transCounts[table.getPosition(predIdx, inputIdx)])
                    {
                        bool canAdd = true;
                        for (int trans = 0; trans < MAX_TRANSITIONS; trans++)
                        {
                            int transIdx = table.get(predIdx, inputIdx, trans);
                            if (table.safeStates[transIdx] > table.safeStates[stateIdx])
                            {
                                canAdd = false;
                                break;
                            }
                        }
                        if (canAdd)
                        {
                            table.safeStates[predIdx] = table.safeStates[stateIdx] + 1;
                            controller[predIdx] = inputIdx;
                            pendingStates.push(predIdx);
                        }
                    }
                }
            }
        }
    }
}

std::vector<std::vector<int>> Automaton::getXORController(py::tuple pyStartStateCoords,
                                                          py::tuple pyFirstLowerBound,
                                                          py::tuple pyFirstUpperBound,
                                                          py::tuple pySecondLowerBound,
                                                          py::tuple pySecondUpperBound,
                                                          py::tuple pyTargetLowerBound,
                                                          py::tuple pyTargetUpperBound,
                                                          int pathCount)
{

    printf("Get XOR Controller printed\n");
    std::vector<int> firstLowerBound(stateDim), firstUpperBound(stateDim);
    std::vector<int> secondLowerBound(stateDim), secondUpperBound(stateDim);
    std::vector<int> targetLowerBound(stateDim), targetUpperBound(stateDim);

    for (int dim = 0; dim < stateDim; dim++)
    {
        firstLowerBound[dim] = pyFirstLowerBound[dim].cast<int>();
        firstUpperBound[dim] = pyFirstUpperBound[dim].cast<int>();

        secondLowerBound[dim] = pySecondLowerBound[dim].cast<int>();
        secondUpperBound[dim] = pySecondUpperBound[dim].cast<int>();

        targetLowerBound[dim] = pyTargetLowerBound[dim].cast<int>();
        targetUpperBound[dim] = pyTargetUpperBound[dim].cast<int>();
    }

    XORTransitionTable xorTable(this,
                                firstLowerBound, firstUpperBound,
                                secondLowerBound, secondUpperBound,
                                targetLowerBound, targetUpperBound);
    printf("XORTransitionTable created\n");

    

    xorTable.solve();

    std::ofstream logFile("XORController.log");
    logFile << ("______\n");
    for(int i = 0; i < xorTable.controller.size(); i++) {
        logFile << i << "->" << xorTable.safeStates[i] << " " << xorTable.controller[i] << std::endl;
    }
    logFile << ("______\n");
    logFile.close();
    //
    // int startStateIdx = 0;
    // for (int i = 0; i < stateDim; ++i)
    //     startStateIdx += pyStartStateCoords[i].cast<int>() * resolutionStride[i];
    //
    // // Check if start state (concrete start, abstract 0) is safe
    // XORState startXOR = {startStateIdx, 0};
    // int startFlat = xorTable.getFlatIdx(startXOR);
    //
    // if (xorTable.safeStates[startFlat] == -1)
    // {
    //     printf("No path found\n");
    //     return std::vector({std::vector({-1})});
    // }
    //
    std::vector<std::vector<int>> statePaths;
    // for (int _ = 0; _ < pathCount; _++)
    // {
    //     statePaths.push_back(getRandomXORPath(startStateIdx, xorTable));
    // }
    //
    return statePaths;
}

std::vector<int> Automaton::floodFill(const std::vector<int> &lowerBoundCoords, const std::vector<int> &upperBoundCoords)
{
    std::vector<int> curCoords{stateDim};
    for (int i = 0; i < stateDim; ++i)
        curCoords[i] = lowerBoundCoords[i];

    std::vector<int> out;

    while (true)
    {
        int cellIdx = 0;
        for (int i = 0; i < stateDim; ++i)
            cellIdx += curCoords[i] * resolutionStride[i];

        out.push_back(cellIdx);

        int d = 0;
        while (d <= stateDim - 1)
        {
            curCoords[d]++;

            if (curCoords[d] <= upperBoundCoords[d])
                break;

            curCoords[d] = lowerBoundCoords[d];
            d++;
        }

        if (d >= stateDim)
            break;
    }

    return out;
}

inline void Automaton::validateDimension(const py::object &space)
{
    if (space.attr("dimensions").cast<int>() > MAX_DIMENSIONS)
    {
        printf("Error: Space dimensions (%d) > MAX_DIMENSIONS (%d) (at %s)\n",
               space.attr("dimensions").cast<int>(),
               MAX_DIMENSIONS,
               __FILE__);

        exit(EXIT_FAILURE);
    }
}

std::vector<int> Automaton::getRandomPath(int startStateIdx) const
{
    std::vector<int> statesPath;
    statesPath.push_back(startStateIdx);

    int stateIdx = startStateIdx;
    std::random_device rd;
    std::mt19937 gen(rd());

    while (table.safeStates[stateIdx] != 0)
    { // while not a target cell
        int inputIdx = controller[stateIdx];

        // Collect all valid transitions
        std::vector<int> validTransitions;
        for (int trans = 0; trans < MAX_TRANSITIONS; trans++)
        {
            int transIdx = table.get(stateIdx, inputIdx, trans);
            if (transIdx != -1)
            {
                validTransitions.push_back(transIdx);
            }
        }

        // Pick random one
        std::uniform_int_distribution<> dis(0, validTransitions.size() - 1);
        int randomIdx = dis(gen);
        stateIdx = validTransitions[randomIdx];
        statesPath.push_back(stateIdx);
    }

    return statesPath;
}

std::vector<int> Automaton::getRandomXORPath(int startStateIdx, const XORTransitionTable &xorTable) const
{
    std::vector<int> statesPath;
    statesPath.push_back(startStateIdx);

    int cState = startStateIdx;
    int aState = 0;

    std::random_device rd;
    std::mt19937 gen(rd());

    // While not in target abstract state (3)
    while (aState != 3)
    {
        XORState currState = {cState, aState};
        int flatIdx = xorTable.getFlatIdx(currState);

        if (xorTable.safeStates[flatIdx] == -1)
            break;

        int inputIdx = xorTable.controller[flatIdx];
        if (inputIdx == -1)
            break;

        // Get valid transitions from the map
        const auto &transitions = (*xorTable.data)[currState][inputIdx];

        std::vector<XORState> validNext;
        int currentDist = xorTable.safeStates[flatIdx];

        for (const auto &nextState : transitions)
        {
            int nextFlat = xorTable.getFlatIdx(nextState);
            // Move to a state closer to the target
            if (xorTable.safeStates[nextFlat] != -1 && xorTable.safeStates[nextFlat] < currentDist)
            {
                validNext.push_back(nextState);
            }
        }

        if (validNext.empty())
            break;

        std::uniform_int_distribution<> dis(0, validNext.size() - 1);
        XORState next = validNext[dis(gen)];

        cState = next.first;
        aState = next.second;
        statesPath.push_back(cState);
    }

    return statesPath;
}
