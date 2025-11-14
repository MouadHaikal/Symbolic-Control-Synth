codeTemplate_f_at_Point="""\
#include <math.h>

__device__ __forceinline__ void fAtPoint(const float* __restrict__ state, 
                                         const float* __restrict__ input, 
                                         const float* __restrict__ disturbance,
                                         float*       __restrict__ nextState)
{{
    {code}
}}\
"""
