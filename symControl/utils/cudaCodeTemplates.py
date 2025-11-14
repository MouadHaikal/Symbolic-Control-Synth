codeTemplate_f_at_Point="""\

__device__ __forceinline__ 
void fAtPoint(const float* __restrict__ state, 
             const float* __restrict__ input, 
             const float* __restrict__ disturbance,
             float*       __restrict__ nextState)
{{
    {code}
}}


extern "C" __global__ 
void testKernel(const int pointCount, 
                const int stateDim,
                const float * __restrict__ state, 
                const int inputDim,
                const float * __restrict__ input, 
                const int disturbanceDim,
                const float * __restrict__ disturbance, 
                float * __restrict__ nextState) 
{{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gtid >= pointCount) return;

    fAtPoint(&state[gtid * stateDim], &input[gtid * inputDim], &disturbance[gtid * disturbanceDim], &nextState[gtid * stateDim]); 
}}\
"""
