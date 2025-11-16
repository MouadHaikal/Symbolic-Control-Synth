codeTemplateCoop="""\
#include <utilsDevice.hpp>


__device__ __forceinline__ 
void fAtPoint(const float* __restrict__ state, 
              const float* __restrict__ input, 
              const float* __restrict__ disturbance,
              float*       __restrict__ nextState)
{{
    {code}
}}


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


    for (int inputIdx = 0; inputIdx < inputSpaceInfo.cellCount; ++inputIdx) {{
        float cellLowerBound[MAX_DIMENSIONS];
        float cellUpperBound[MAX_DIMENSIONS];
        stateSpaceInfo.getCellBounds(stateIdx, cellLowerBound, cellUpperBound);

        float inputCellCenter[MAX_DIMENSIONS];
        inputSpaceInfo.getCellCenter(inputIdx, inputCellCenter);


        float targetLowerBound[MAX_DIMENSIONS];
        float targetUpperBound[MAX_DIMENSIONS];

        fAtPoint(cellLowerBound, inputCellCenter, disturbanceSpaceBounds.dLowerBound, targetLowerBound);
        fAtPoint(cellUpperBound, inputCellCenter, disturbanceSpaceBounds.dUpperBound, targetUpperBound);

        int targetLowerBoundCoords[MAX_DIMENSIONS];
        int targetUpperBoundCoords[MAX_DIMENSIONS];

        stateSpaceInfo.getCellCoords(targetLowerBound, targetLowerBoundCoords);
        stateSpaceInfo.getCellCoords(targetUpperBound, targetUpperBoundCoords);

        floodFill(targetLowerBoundCoords, 
                  targetUpperBoundCoords, 
                  resolutionStride,
                  stateSpaceInfo.dimensions, 
                  &table.dData[table.getOffset(stateIdx, inputIdx, 0)]
        );
    }}
}}\
"""
