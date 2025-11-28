# =============================================
#                  Utilities
# =============================================

__includesCodeTemplate="""\
    #include <utilsDevice.hpp>
"""

__definesCodeTemplate="""\
    #define GRAD_DESCENT_ITERS 50
    #define CLAMP(x, a, b) fmaxf(a, fminf(x, b))
"""

__mutexHandlingCodeTemplate="""\
    __device__ __forceinline__
    void acquireLock(int* lock) {{
        while (atomicCAS(lock, 0, 1) != 0);
    }}

    __device__ __forceinline__
    void releaseLock(int* lock) {{
        atomicExch(lock, 0);
    }}
"""



# =============================================
#             Generated Functions
# =============================================

__fAtPointCodeTemplate="""\
    __device__ __forceinline__ 
    void fAtPoint(const float* __restrict__ state, 
                  const float* __restrict__ input, 
                  const float* __restrict__ disturbance,
                  float*       __restrict__ fAtPointOut)
    {{
        {fAtPointCode}
    }}
"""

__stateJacAtPointCodeTemplate="""\
    __device__ __forceinline__ 
    void stateJacAtPoint(const float* __restrict__ states,
                         const float* __restrict__ inputs,
                         const float* __restrict__ disturbances,
                         float*       __restrict__ stateJacAtPointOut)
    {{
        {stateJacAtPointCode}
    }}
"""

__stateJacGradAtPointCodeTemplate="""\
    __device__ __forceinline__ 
    void stateJacGradAtPoint(const float* __restrict__ states,
                             const float* __restrict__ inputs,
                             const float* __restrict__ disturbances,
                             float*       __restrict__ outStates,
                             float*       __restrict__ outInputs,
                             float*       __restrict__ outDisturbances)
    {{
        {stateJacGradAtPointCode}
    }}
"""



# =============================================
#               Helper Functions
# =============================================

__findStateJacExtremumCodeTemplate="""\
    enum Extremum{{
        MAXIMUM,
        MINIMUM
    }};

    __device__ __forceinline__ 
    void findStateJacExtremum(const Extremum            extremum,
                              const float* __restrict__ stateLowerBound,
                              const float* __restrict__ stateUpperBound,
                              const float* __restrict__ inputLowerBound,
                              const float* __restrict__ inputUpperBound,
                              const float* __restrict__ disturbanceLowerBound,
                              const float* __restrict__ disturbanceUpperBound,
                              int                       stateDim,
                              int                       inputDim,
                              int                       disturbDim,
                              float*       __restrict__ out)
    {{
        // Compute adequate timeStep
        float timeStep = 0.f;
        {{
            for (int i = 0; i < stateDim; i++){{
                timeStep += (stateUpperBound[i] - stateLowerBound[i]) * (stateUpperBound[i] - stateLowerBound[i]);
            }}

            for (int i = 0; i < inputDim; i++){{
                timeStep += (inputUpperBound[i] - inputLowerBound[i]) * (inputUpperBound[i] - inputLowerBound[i]);
            }}

            for (int i = 0; i < disturbDim; i++){{
                timeStep += (disturbanceUpperBound[i] - disturbanceLowerBound[i]) * (disturbanceUpperBound[i] - disturbanceLowerBound[i]);
            }}

            timeStep = sqrtf(timeStep)  / 10.f;
        }}


        // Set initial guesses to centers
        float states[MAX_DIMENSIONS*MAX_DIMENSIONS*MAX_DIMENSIONS];
        float inputs[MAX_DIMENSIONS*MAX_DIMENSIONS*MAX_DIMENSIONS];
        float disturbances[MAX_DIMENSIONS*MAX_DIMENSIONS*MAX_DIMENSIONS];

        for (int i = 0; i < stateDim * stateDim; i++){{
            for (int j = 0; j < stateDim; j++){{
                states[i*stateDim + j] = .5f * (stateLowerBound[j] + stateUpperBound[j]);
            }}

            for (int j = 0; j < inputDim; j++){{
                inputs[i*inputDim + j] = .5f * (inputLowerBound[j] + inputUpperBound[j]);
            }}

            for (int j = 0; j < disturbDim; j++){{
                disturbances[i*disturbDim + j] = .5f * (disturbanceLowerBound[i] + disturbanceUpperBound[i]);
            }}
        }}


        float nextStates[MAX_DIMENSIONS*MAX_DIMENSIONS*MAX_DIMENSIONS];
        float nextInputs[MAX_DIMENSIONS*MAX_DIMENSIONS*MAX_DIMENSIONS];
        float nextDisturbances[MAX_DIMENSIONS*MAX_DIMENSIONS*MAX_DIMENSIONS];

        
        for (int iter = 0; iter < GRAD_DESCENT_ITERS; iter++){{
            // Compute state gradients
            stateJacGradAtPoint(states,
                                inputs,
                                disturbances,
                                nextStates,
                                nextInputs,
                                nextDisturbances
            );


            // Update guesses
            for (int i = 0; i < stateDim * stateDim; i++){{
                for (int j = 0; j < stateDim; j++){{
                    states[i*stateDim + j] += ((extremum==MAXIMUM) ? 1.f : -1.f) * nextStates[i*stateDim + j] * timeStep;
                    states[i*stateDim + j] = CLAMP(states[i*stateDim + j], stateLowerBound[j], stateUpperBound[j]);
                }}

                for (int j = 0; j < inputDim; j++){{
                    inputs[i*inputDim + j] += ((extremum==MAXIMUM) ? 1.f : -1.f) * nextInputs[i*inputDim + j] * timeStep;
                    inputs[i*stateDim + j] = CLAMP(inputs[i*stateDim + j], inputLowerBound[j], inputUpperBound[j]);
                }}

                for (int j = 0; j < disturbDim; j++){{
                    disturbances[i*disturbDim + j] += ((extremum==MAXIMUM) ? 1.f : -1.f) * nextDisturbances[i*disturbDim + j] * timeStep;
                    disturbances[i*stateDim + j] = CLAMP(disturbances[i*stateDim + j], disturbanceLowerBound[j], disturbanceUpperBound[j]);
                }}
            }}
        }}


        // Evaluate state Jacobian at final guesses
        stateJacAtPoint(states,
                        inputs,
                        disturbances,
                        out
        );
    }}
"""

__floodFillCodeTemplate="""\
    __device__ __forceinline__
    void floodFill(const int* __restrict__ lowerBoundCoords,
                   const int* __restrict__ upperBoundCoords,
                   const int* __restrict__ resolutionStride,
                   const int               dimensions, 
                   int*       __restrict__ out)
    {{
        int curCoords[MAX_DIMENSIONS];
        for (int i = 0; i < dimensions; ++i) 
            curCoords[i] = lowerBoundCoords[i];

        int curTransition = 0;

        while (true) {{
            int cellIdx = 0;
            for (int i = 0; i < dimensions; ++i) 
                cellIdx += curCoords[i] * resolutionStride[i];
                
            out[curTransition++] = cellIdx;
            
            int d = 0;
            while (d <= dimensions - 1) {{
                curCoords[d]++;

                if (curCoords[d] <= upperBoundCoords[d]) break;

                curCoords[d] = lowerBoundCoords[d];
                d++;
            }}

            if (d >= dimensions) break;
        }}
    }}
"""

__storeOutputCodeTemplate="""\
    __device__ __forceinline__
    void storeOutput(int                   stateIdx,
                     int                   inputIdx,
                     TransitionTableDevice table) 
    {{
        for(int i = 0; i < MAX_TRANSITIONS; i++) {{
            int directOffset = table.getOffset(stateIdx, inputIdx, i);        

            int predecessor = table.dData[directOffset];
            if(predecessor == -1) break;

            int reverseOffset = table.getRevOffset(predecessor, inputIdx);
            int lockOffset = predecessor * table.inputCount + inputIdx;

            acquireLock(&table.dTransitionLocks[lockOffset]);
            while(table.dRevData[reverseOffset] != -1) reverseOffset++;
            table.dRevData[reverseOffset] = stateIdx;
            releaseLock(&table.dTransitionLocks[lockOffset]);
        }}
    }}
"""



# =============================================
#               Full Kernel Code
# =============================================

coopCodeTemplate="""\
    """ + __includesCodeTemplate + """

    """ + __mutexHandlingCodeTemplate + """

    """ + __fAtPointCodeTemplate + """

    """ + __floodFillCodeTemplate + """

    """ + __storeOutputCodeTemplate + """

    extern "C" __global__ 
    void buildAutomatonCoop(const SpaceInfoDevice   stateSpaceInfo,
                            const SpaceInfoDevice   inputSpaceInfo,
                            const SpaceBoundsDevice disturbanceSpaceBounds,
                            TransitionTableDevice   table) 
    {{
        int stateIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (stateIdx >= table.stateCount) return;

        int resolutionStride[MAX_DIMENSIONS];
        resolutionStride[0] = 1;
        for (int i = 1; i < stateSpaceInfo.dimensions; ++i)
            resolutionStride[i] = stateSpaceInfo.dResolutions[i-1] * resolutionStride[i-1];


        float stateCellLowerBound[MAX_DIMENSIONS];
        float stateCellUpperBound[MAX_DIMENSIONS];
        stateSpaceInfo.getCellBounds(stateIdx, stateCellLowerBound, stateCellUpperBound);

        float inputCellCenter[MAX_DIMENSIONS];

        float targetLowerBound[MAX_DIMENSIONS];
        float targetUpperBound[MAX_DIMENSIONS];

        int targetLowerBoundCoords[MAX_DIMENSIONS];
        int targetUpperBoundCoords[MAX_DIMENSIONS];


        for (int inputIdx = 0; inputIdx < inputSpaceInfo.cellCount; ++inputIdx) {{
            inputSpaceInfo.getCellCenter(inputIdx, inputCellCenter);

            fAtPoint(stateCellLowerBound, inputCellCenter, disturbanceSpaceBounds.dLowerBound, targetLowerBound);
            fAtPoint(stateCellUpperBound, inputCellCenter, disturbanceSpaceBounds.dUpperBound, targetUpperBound);

            bool outOfBounds = stateSpaceInfo.getCellCoords(targetLowerBound, targetLowerBoundCoords);
            outOfBounds |= stateSpaceInfo.getCellCoords(targetUpperBound, targetUpperBoundCoords);

            if (!outOfBounds){{
                floodFill(targetLowerBoundCoords, 
                          targetUpperBoundCoords, 
                          resolutionStride,
                          stateSpaceInfo.dimensions, 
                          &table.dData[table.getOffset(stateIdx, inputIdx, 0)]
                );

                storeOutput(stateIdx, inputIdx, table);
            }}

        }}
    }}
"""

nonCoopCodeTemplate="""\
    """ + __includesCodeTemplate + """

    """ + __definesCodeTemplate + """

    """ + __mutexHandlingCodeTemplate + """

    """ + __fAtPointCodeTemplate + """

    """ + __stateJacAtPointCodeTemplate + """

    """ + __stateJacGradAtPointCodeTemplate + """

    """ + __findStateJacExtremumCodeTemplate + """

    """ + __floodFillCodeTemplate + """

    """ + __storeOutputCodeTemplate + """


    extern "C" __global__ 
    void buildAutomatonNonCoop(const SpaceInfoDevice     stateSpaceInfo,
                               const SpaceInfoDevice     inputSpaceInfo,
                               const SpaceBoundsDevice   disturbanceSpaceBounds,
                               const float* __restrict__ maxDisturbJac,
                               TransitionTableDevice     table) 
    {{
        int stateIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (stateIdx >= table.stateCount) return;

        int resolutionStride[MAX_DIMENSIONS];
        resolutionStride[0] = 1;
        for (int i = 1; i < stateSpaceInfo.dimensions; ++i)
            resolutionStride[i] = stateSpaceInfo.dResolutions[i-1] * resolutionStride[i-1];


        float stateCellLowerBound[MAX_DIMENSIONS];
        float stateCellUpperBound[MAX_DIMENSIONS];
        float stateCellCenter[MAX_DIMENSIONS];
        stateSpaceInfo.getCellBounds(stateIdx, stateCellLowerBound, stateCellUpperBound);
        stateSpaceInfo.getCellCenter(stateIdx, stateCellCenter);

        float inputCellLowerBound[MAX_DIMENSIONS];
        float inputCellUpperBound[MAX_DIMENSIONS];
        float inputCellCenter[MAX_DIMENSIONS];

        float targetLowerBound[MAX_DIMENSIONS];
        float targetUpperBound[MAX_DIMENSIONS];

        int targetLowerBoundCoords[MAX_DIMENSIONS];
        int targetUpperBoundCoords[MAX_DIMENSIONS];

        const int stateDim   = stateSpaceInfo.dimensions;
        const int inputDim   = inputSpaceInfo.dimensions;
        const int disturbDim = disturbanceSpaceBounds.dimensions;


        for (int inputIdx = 0; inputIdx < inputSpaceInfo.cellCount; ++inputIdx) {{
            inputSpaceInfo.getCellBounds(inputIdx, inputCellLowerBound, inputCellUpperBound);
            inputSpaceInfo.getCellCenter(inputIdx, inputCellCenter);


            // Compute state Jacobian's absolute upper bound
            float maxStateJac[MAX_DIMENSIONS*MAX_DIMENSIONS];
            {{
                float minStateJac[MAX_DIMENSIONS*MAX_DIMENSIONS];

                findStateJacExtremum(MINIMUM,
                                     stateCellLowerBound,
                                     stateCellUpperBound,
                                     inputCellLowerBound,
                                     inputCellUpperBound,
                                     disturbanceSpaceBounds.dLowerBound,
                                     disturbanceSpaceBounds.dUpperBound,
                                     stateDim,
                                     inputDim,
                                     disturbDim,
                                     minStateJac
                );

                findStateJacExtremum(MAXIMUM,
                                     stateCellLowerBound,
                                     stateCellUpperBound,
                                     inputCellLowerBound,
                                     inputCellUpperBound,
                                     disturbanceSpaceBounds.dLowerBound,
                                     disturbanceSpaceBounds.dUpperBound,
                                     stateDim,
                                     inputDim,
                                     disturbDim,
                                     maxStateJac
                );


                for (int i = 0; i < stateDim * stateDim; i++){{
                    maxStateJac[i] = fmaxf(maxStateJac[i], -minStateJac[i]);
                }}
            }}


            // Compute target space bounds
            fAtPoint(stateCellCenter, inputCellCenter, disturbanceSpaceBounds.dCenter, targetLowerBound);
            fAtPoint(stateCellCenter, inputCellCenter, disturbanceSpaceBounds.dCenter, targetUpperBound);

            for (int i = 0; i < stateDim; i++){{
                for (int j = 0; j < stateDim; j++){{
                    float stateHalfWidth = maxStateJac[i*stateDim + j] * 
                        0.5f * (stateCellUpperBound[j] - stateCellLowerBound[j]);
                    targetLowerBound[i] -= stateHalfWidth;
                    targetUpperBound[i] += stateHalfWidth;
                }}

                for (int j = 0; j < disturbDim; j++){{
                    float disturbHalfWidth = maxDisturbJac[i*disturbDim + j] * 
                        0.5f * (disturbanceSpaceBounds.dUpperBound[j] - disturbanceSpaceBounds.dLowerBound[j]);
                    targetLowerBound[i] -= disturbHalfWidth;
                    targetUpperBound[i] += disturbHalfWidth;
                }}
            }}


            bool outOfBounds = stateSpaceInfo.getCellCoords(targetLowerBound, targetLowerBoundCoords);
            outOfBounds |= stateSpaceInfo.getCellCoords(targetUpperBound, targetUpperBoundCoords);

            if (!outOfBounds){{
                floodFill(targetLowerBoundCoords, 
                          targetUpperBoundCoords, 
                          resolutionStride,
                          stateDim, 
                          &table.dData[table.getOffset(stateIdx, inputIdx, 0)]
                );

                storeOutput(stateIdx, inputIdx, table);
            }}
        }}
    }}
"""
