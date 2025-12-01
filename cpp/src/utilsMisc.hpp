// Here we put any utility structures that aren't needed by our core functionalites
// the utilities here are used to represent the other specifications as presented in the paper.

#pragma once

#include "automaton.hpp"
#include "utilsHost.hpp"
#include <nvrtc.h>
#include <cuda_runtime.h>
#include <pybind11/embed.h>
#include <cuda.h>
#include <map>
#include <vector>
#include <queue>

struct XORAutomaton
{
    static constexpr int stateCount = 5;
    static constexpr int inputCount = 4;

    int transitionTable[stateCount][inputCount] = {
        {0, 1, 2, 0},
        {1, 1, 4, 3},
        {2, 4, 2, 3},
        {3, 3, 3, 3},
        {4, 4, 4, 4}};
};

using XORState = std::pair<int, int>; // <concrete, abstract>
using Input = int;
#define concrete first
#define abstract second

struct XORTransitionTable
{
    // The augmented automaton after applying the XOR specification
    // We suppose that the security specification is already applied and we deal only with what is left
    Automaton *originalModel;
    XORAutomaton *specification;

    // Instead of the long array, we will leverage the use of STL datastructures to represent the automaton
    std::map<XORState, std::map<Input, std::vector<XORState>>> *data, *revData;

    // store bounds so getPsi can check coordinates
    std::vector<int> firstLowerBound, firstUpperBound;
    std::vector<int> secondLowerBound, secondUpperBound;
    std::vector<int> targetLowerBound, targetUpperBound;

    // Synthesis structures
    std::vector<int> safeStates;
    std::vector<int> controller;
    std::vector<int> transCounts;
    std::vector<int> safeCounts;

    XORTransitionTable(Automaton *model,
                       std::vector<int> firstSpaceLowerBound,
                       std::vector<int> firstSpaceUpperBound,
                       std::vector<int> secondSpaceLowerBound,
                       std::vector<int> secondSpaceUpperBound,
                       std::vector<int> targetSpaceLowerBound,
                       std::vector<int> targetSpaceUpperBound) : 
        originalModel(model),
        firstLowerBound(std::move(firstSpaceLowerBound)),
        firstUpperBound(std::move(firstSpaceUpperBound)),
        secondLowerBound(std::move(secondSpaceLowerBound)),
        secondUpperBound(std::move(secondSpaceUpperBound)),
        targetLowerBound(std::move(targetSpaceLowerBound)),
        targetUpperBound(std::move(targetSpaceUpperBound))
    {
        specification = new XORAutomaton();
        TransitionTableHost *transitionModel = &(model->table);
        // initialize maps
        data = new std::map<XORState, std::map<Input, std::vector<XORState>>>();
        revData = new std::map<XORState, std::map<Input, std::vector<XORState>>>();

        int concreteStates = transitionModel->stateCount;
        int inputs = transitionModel->inputCount;
        int abstractStates = XORAutomaton::stateCount;

        // Resize synthesis vectors
        int totalStates = concreteStates * abstractStates;
        safeStates.resize(totalStates, -1);
        controller.resize(totalStates, -1);
        transCounts.resize(totalStates * inputs, 0);
        safeCounts.resize(totalStates * inputs, 0);

        for (int cState = 0; cState < concreteStates; cState++)
        {
            for (int aState = 0; aState < abstractStates; aState++)
            {
                XORState currState = {cState, aState};

                for (int u = 0; u < inputs; u++)
                {
                    for (int t = 0; t < MAX_TRANSITIONS; t++)
                    {
                        int nextCState = transitionModel->get(cState, u, t);
                        if (nextCState == -1)
                            continue;

                        // The input to the XOR automaton is the label of the next concrete state
                        int label = getPsi(nextCState);
                        int nextAState = specification->transitionTable[aState][label];

                        XORState nextState = {nextCState, nextAState};

                        (*data)[currState][u].push_back(nextState);
                        (*revData)[nextState][u].push_back(currState);

                        // Count transitions
                        transCounts[getFlatTransIdx(currState, u)]++;
                    }
                }
            }
        }
    }

    ~XORTransitionTable()
    {
        delete specification;
        delete data;
        delete revData;
    }

    inline int getPsi(int idx)
    {
        // Unpack linear index `idx` into coordinates using originalModel->resolutionStride.
        // Then check whether coords lie within target (return 3), first (1), second (2), or none (0).
        const auto &stride = originalModel->resolutionStride; // expecting public access
        size_t dim = stride.size();
        std::vector<int> coords(dim);
        int remaining = idx;
        for (size_t i = 0; i < dim; ++i)
        {
            int s = stride[i];
            // avoid division by zero; if stride is zero treat coord as 0
            if (s <= 0)
            {
                coords[i] = 0;
            }
            else
            {
                coords[i] = remaining / s;
                remaining = remaining % s;
            }
        }

        auto within = [&](const std::vector<int> &low, const std::vector<int> &high) -> bool
        {
            if (low.size() != dim || high.size() != dim)
                return false;
            for (size_t d = 0; d < dim; ++d)
            {
                if (coords[d] < low[d] || coords[d] > high[d])
                    return false;
            }
            return true;
        };

        if (within(targetLowerBound, targetUpperBound))
            return 3;
        if (within(firstLowerBound, firstUpperBound))
            return 1;
        if (within(secondLowerBound, secondUpperBound))
            return 2;
        return 0;
    }

    void solve()
    {
        int concreteStates = originalModel->table.stateCount;
        std::queue<XORState> q;

        // 1. Initialize target states (abstract state == 3)
        for (int c = 0; c < concreteStates; ++c)
        {
            XORState s = {c, 3};
            int idx = getFlatIdx(s);
            safeStates[idx] = 0;
            q.push(s);
        }

        // 2. Backward Reachability
        while (!q.empty())
        {
            // printf("%d ", (int)q.size());
            XORState curr = q.front();
            q.pop();

            int currIdx = getFlatIdx(curr);
            int currDist = safeStates[currIdx];

            if (revData->find(curr) == revData->end())
                continue;

            for (auto const &[input, preds] : (*revData)[curr])
            {
                for (const auto &pred : preds)
                {
                    int predIdx = getFlatIdx(pred);

                    // If already safe, skip
                    printf("%d with %d safe state level\n", predIdx, safeStates[predIdx]);
                    if (safeStates[predIdx] != -1)
                        continue;

                    int transIdx = getFlatTransIdx(pred, input);
                    safeCounts[transIdx]++;
                    printf("%d -> %d | %d\n", transIdx, transCounts[transIdx], safeCounts[transIdx]);

                    if (safeCounts[transIdx] == transCounts[transIdx])
                    {
                        // All transitions from (pred, input) lead to safe states
                        safeStates[predIdx] = currDist + 1;
                        controller[predIdx] = input;
                        q.push(pred);
                        printf("Element %d pushed\n", predIdx);
                    }
                }
            }
        }
    }

    inline int getFlatIdx(const XORState &s) const
    {
        return s.first * XORAutomaton::stateCount + s.second;
    }

    inline int getFlatTransIdx(const XORState &s, int input) const
    {
        return getFlatIdx(s) * originalModel->table.inputCount + input;
    }
};
