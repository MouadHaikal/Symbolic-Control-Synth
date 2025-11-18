__includesCodesTemplate="""\
    #include <utilsDevice.hpp>
    __device__ void acquireLock(int* lock) {{
        while (atomicCAS(lock, 0, 1) != 0);
    }}

    __device__ void releaseLock(int* lock) {{
        atomicExch(lock, 0);
    }}
"""

__fAtPointCodeTemplate="""\
    __device__ __forceinline__ 
    void fAtPoint(const float* __restrict__ state, 
                  const float* __restrict__ input, 
                  const float* __restrict__ disturbance,
                  float*       __restrict__ fAtPointOut)
    {{
        {fAtPointCode}
    }}\
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
        for (int i = 0; i < dimensions; ++i) curCoords[i] = lowerBoundCoords[i];

        int curTransition = 0;

        while (true) {{
            int cellIdx = 0;
            for (int i = 0; i < dimensions; ++i) cellIdx += curCoords[i] * resolutionStride[i];
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
                     TransitionTableDevice table) {{
        //printf("storing reverse output");

        // &table.dData[table.getOffset(stateIdx, inputIdx, 0)]

        // we now set the other way around with the reverse array
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

coopCodeTemplate="""\
    """ + __includesCodesTemplate + """

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


        float cellLowerBound[MAX_DIMENSIONS];
        float cellUpperBound[MAX_DIMENSIONS];
        stateSpaceInfo.getCellBounds(stateIdx, cellLowerBound, cellUpperBound);

        float inputCellCenter[MAX_DIMENSIONS];

        float targetLowerBound[MAX_DIMENSIONS];
        float targetUpperBound[MAX_DIMENSIONS];

        int targetLowerBoundCoords[MAX_DIMENSIONS];
        int targetUpperBoundCoords[MAX_DIMENSIONS];

        for (int inputIdx = 0; inputIdx < inputSpaceInfo.cellCount; ++inputIdx) {{
            inputSpaceInfo.getCellCenter(inputIdx, inputCellCenter);

            fAtPoint(cellLowerBound, inputCellCenter, disturbanceSpaceBounds.dLowerBound, targetLowerBound);
            fAtPoint(cellUpperBound, inputCellCenter, disturbanceSpaceBounds.dUpperBound, targetUpperBound);

            stateSpaceInfo.getCellCoords(targetLowerBound, targetLowerBoundCoords);
            stateSpaceInfo.getCellCoords(targetUpperBound, targetUpperBoundCoords);

            floodFill(targetLowerBoundCoords, 
                      targetUpperBoundCoords, 
                      resolutionStride,
                      stateSpaceInfo.dimensions, 
                      &table.dData[table.getOffset(stateIdx, inputIdx, 0)]
            );

            storeOutput(stateIdx, inputIdx, table);
        }}
    }}\
"""


__stateJacAtPointCodeTemplate="""\
    __device__ __forceinline__ 
    void stateJacAtPoint(const float* __restrict__ states,
                         const float* __restrict__ inputs,
                         const float* __restrict__ disturbances,
                         float*       __restrict__ stateJacAtPointOut)
    {{
        {stateJacAtPointCode}
    }}\
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
    }}\
"""

__findStateJacExtremumCodeTemplate="""\
    enum Extremum{{
        MAXIMUM,
        MINIMUM
    }};
    
    __device__ __forceinline__ 
    void findStateJacExtremum(const Extremum      extremum,
                              const float* __restrict__ stateLowerBound,
                              const float* __restrict__ stateUpperBound,
                              const float* __restrict__ inputLowerBound,
                              const float* __restrict__ inputUpperBound,
                              const float* __restrict__ disturbanceLowerBound,
                              const float* __restrict__ disturbanceUpperBound,
                              int                       nx,
                              int                       nu,
                              int                       nw,
                              float*       __restrict__ states,           // x values of current points (len = nx**2 * nx)
                              float*       __restrict__ inputs,           // u values of current points (len = nx**2 * nu)
                              float*       __restrict__ disturbances, 
                              float*       __restrict__ out)              // jacobian max values (len = nx**2)
    {{
        float timeStep = 0.f;

        {{
            for (int i = 0; i < nx; i++){{
                timeStep += (stateUpperBound[i] - stateLowerBound[i]) * (stateUpperBound[i] - stateLowerBound[i]);
            }}

            for (int i = 0; i < nu; i++){{
                timeStep += (inputUpperBound[i] - inputLowerBound[i]) * (inputUpperBound[i] - inputLowerBound[i]);
            }}

            for (int i = 0; i < nw; i++){{
                timeStep += (disturbanceUpperBound[i] - disturbanceLowerBound[i]) * (disturbanceUpperBound[i] - disturbanceLowerBound[i]);
            }}

            timeStep = sqrtf(timeStep)
        }}


        // Filling with initial guesses
        for (int i = 0; i < nx * nx; i++){{
            for (int j = 0; j < nx; j++){{
                states[i*nx + j] = (stateLowerBound[j] + stateUpperBound[j]) / 2.f;
            }}

            for (int j = 0; j < nu; j++){{
                inputs[i*nu + j] = (inputLowerBound[j] + inputUpperBound[j]) / 2.f;
            }}

            for (int j = 0; j < nw; j++){{
                distrubances[i*nw + j] = (disturbanceLowerBound[i] + disturbanceUpperBound[i]) / 2.f;
            }}
        }}

        float nextStates[MAX_DIMENSIONS*MAX_DIMENSIONS*MAX_DIMENSIONS];
        float nextInputs[MAX_DIMENSIONS*MAX_DIMENSIONS*MAX_DIMENSIONS];
        float nextDisturbances[MAX_DIMENSIONS*MAX_DIMENSIONS*MAX_DIMENSIONS];
        
        for (int iter = 0; iter < GRAD_DESCENT_ITERS; iter++){{
            stateJacGradAtPoint(states,
                                inputs,
                                disturbances,
                                nextStates,
                                nextInputs,
                                nextDisturbances
            );

            for (int i = 0; i < nx * nx; i++){{
                for (int j = 0; j < nx; j++){{
                    states[i*nx + j] += ((extremum==MAXIMUM) ? 1.f : -1.f) * nextStates[i*nx + j] * timeStep;
                    states[i*nx + j] = clamp(states[i*nx + j], stateLowerBound[j], stateUpperBound[j]);
                }}

                for (int j = 0; j < nu; j++){{
                    inputs[i*nu + j] += ((extremum==MAXIMUM) ? 1.f : -1.f) * nextInputs[i*nu + j] * timeStep;
                    inputs[i*nx + j] = clamp(inputs[i*nx + j], inputLowerBound[j], inputUpperBound[j]);
                }}

                for (int j = 0; j < nw; j++){{
                    disturbances[i*nw + j] += ((extremum==MAXIMUM) ? 1.f : -1.f) * nextDisturbances[i*nw + j] * timeStep;
                    disturbances[i*nx + j] = clamp(disturbances[i*nx + j], disturbanceLowerBound[j], disturbanceUpperBound[j]);
                }}
            }}
        }}

        stateJacAtPoint(states,
                        inputs,
                        disturbances,
                        out
        )
    }}
"""


nonCoopCodeTemplate="""\
    """ + __includesCodesTemplate + """

    #define GRAD_DESCENT_ITERS 10

    """ + __fAtPointCodeTemplate + """

    """ + __stateJacAtPointCodeTemplate + """

    """ + __stateJacGradAtPointCodeTemplate + """

    """ + __findStateJacExtremumCodeTemplate + """

    """ + __floodFillCodeTemplate + """


    extern "C" __global__ 
    void buildAutomatonNonCoop(const SpaceInfoDevice   stateSpaceInfo,
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

        float inputCellLowerBound[MAX_DIMENSIONS];
        float inputCellUpperBound[MAX_DIMENSIONS];

        float targetLowerBound[MAX_DIMENSIONS];
        float targetUpperBound[MAX_DIMENSIONS];

        int targetLowerBoundCoords[MAX_DIMENSIONS];
        int targetUpperBoundCoords[MAX_DIMENSIONS];

        const int nx = stateSpaceInfo.dimensions;
        const int nu = inputSpaceInfo.dimensions;
        const int nw = disturbanceSpaceBounds.dimensions;

        for (int inputIdx = 0; inputIdx < inputSpaceInfo.cellCount; ++inputIdx) {{
            inputSpaceInfo.getCellBounds(inputIdx, inputCellLowerBound, inputCellUpperBound);

            float maxStateJac[MAX_DIMENSIONS*MAX_DIMENSIONS];

            // Compute max of abs state jac
            {{
                float initialStates[MAX_DIMENSIONS*MAX_DIMENSIONS*MAX_DIMENSIONS];
                float initialInputs[MAX_DIMENSIONS*MAX_DIMENSIONS*MAX_DIMENSIONS];
                float initialDisturbances[MAX_DIMENSIONS*MAX_DIMENSIONS*MAX_DIMENSIONS];

                float minStateJac[MAX_DIMENSIONS*MAX_DIMENSIONS];


                findStateJacExtremum(MINIMUM,
                                     stateCellLowerBound,
                                     stateCellUpperBound,
                                     inputCellLowerBound,
                                     inputCellUpperBound,
                                     disturbanceSpaceBounds.dLowerBound,
                                     disturbanceSpaceBounds.dUpperBound,
                                     nx,
                                     nu,
                                     nw,
                                     initialStates,
                                     initialInputs,
                                     initialDisturbances,
                                     minStateJac
                )


                findStateJacExtremum(MAXIMUM,
                                     stateCellLowerBound,
                                     stateCellUpperBound,
                                     inputCellLowerBound,
                                     inputCellUpperBound,
                                     disturbanceSpaceBounds.dLowerBound,
                                     disturbanceSpaceBounds.dUpperBound,
                                     nx,
                                     nu,
                                     nw,
                                     initialStates,
                                     initialInputs,
                                     initialDisturbances,
                                     maxStateJac
                )

                
                for (int i = 0; i < nx * nx; i++){{
                    maxStateJac[i] = fmax(maxStateJac[i], -minStateJac[i]);
                }}
            }}


        }}
    }}\
"""
