#pragma once 

#include "utilsDevice.hpp"
#include <nvrtc.h>
#include <cuda_runtime.h>
#include <pybind11/embed.h>
#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <set>

#define BLOCK_SIZE 512

namespace py = pybind11;

struct SpaceBoundsHost{
    int    dimensions;
    float* dLowerBound;
    float* dUpperBound;
    float* dCenter;

    SpaceBoundsHost(py::object space):
        dimensions(space.attr("dimensions").cast<int>())
    {
        float hLowerBound[dimensions];
        float hUpperBound[dimensions];
        float hCenter[dimensions];

        py::tuple pyBounds = space.attr("bounds").cast<py::tuple>();
        for(int i = 0; i < dimensions; i++) {
            py::tuple inner = pyBounds[i].cast<py::tuple>();
            float lower = inner[0].cast<float>();
            float upper = inner[1].cast<float>();
            hLowerBound[i] = lower;
            hUpperBound[i] = upper;
            hCenter[i] = (lower + upper) / 2.0;
        }

        cudaMalloc(&dLowerBound, dimensions * sizeof(float));
        cudaMalloc(&dUpperBound, dimensions * sizeof(float));
        cudaMalloc(&dCenter, dimensions * sizeof(float));

        cudaMemcpy(dLowerBound, hLowerBound, dimensions * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dUpperBound, hUpperBound, dimensions * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dCenter, hCenter, dimensions * sizeof(float), cudaMemcpyHostToDevice);
    }

    ~SpaceBoundsHost() {
        cudaFree(dLowerBound);
        cudaFree(dUpperBound);
        cudaFree(dCenter);
    }
};


struct SpaceInfoHost{
    int    dimensions;
    int    cellCount;
    float* dLowerBound;
    float* dUpperBound;
    float* dCellSize;
    int*   dResolutions;

    SpaceInfoHost(py::object space)
        :dimensions(space.attr("dimensions").cast<int>()), cellCount(space.attr("cellCount").cast<int>())
    {
        float hLowerBound[dimensions];
        float hUpperBound[dimensions];
        int   hResolutions[dimensions];
        float hCellSize[dimensions];
        
        py::tuple pyBounds = space.attr("bounds").cast<py::tuple>();
        for(int i = 0; i < dimensions; i++) {
            py::tuple inner = pyBounds[i].cast<py::tuple>();
            hLowerBound[i] = inner[0].cast<float>();
            hUpperBound[i] = inner[1].cast<float>();
        }
        py::tuple pyResolutions = space.attr("resolutions").cast<py::tuple>();
        for(int i = 0; i < dimensions; i++) {
            hResolutions[i] = pyResolutions[i].cast<int>();
        }
        py::tuple pyCellSize = space.attr("cellSize").cast<py::tuple>();
        for(int i = 0; i < dimensions; i++) {
            hCellSize[i] = pyCellSize[i].cast<float>();
        }
       

        cudaMalloc(&dResolutions, dimensions * sizeof(int));
        cudaMalloc(&dCellSize, dimensions * sizeof(float));
        cudaMalloc(&dLowerBound, dimensions * sizeof(float));
        cudaMalloc(&dUpperBound, dimensions * sizeof(float));

        cudaMemcpy(dResolutions, hResolutions, dimensions * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dCellSize, hCellSize, dimensions * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dLowerBound, hLowerBound, dimensions * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dUpperBound, hUpperBound, dimensions * sizeof(float), cudaMemcpyHostToDevice);
    }

    ~SpaceInfoHost() {
        cudaFree(dResolutions);
        cudaFree(dCellSize);
        cudaFree(dLowerBound);
        cudaFree(dUpperBound);
    }
};


struct TransitionTableHost{
    int*         dData;
    int*         hData;
    int*         dRevData;
    int*         hRevData;
    int*         dTransitionLocks;
    const size_t stateCount;
    const size_t inputCount;

    int*         transCounts;
    int*         safeCounts;
    std::set<int> safeStates; //R


    TransitionTableHost(size_t stateCount, size_t inputCount) 
    : stateCount(stateCount), inputCount(inputCount) 
    {
        size_t size = stateCount * inputCount * MAX_TRANSITIONS * sizeof(int);
        cudaMalloc(&dData, size);
        cudaMemset(dData, EMPTY_CELL, size);
        hData = new int[size/sizeof(int)];

        size_t sizeRev = stateCount * inputCount * MAX_PREDECESSORS * sizeof(int);
        cudaMalloc(&dRevData, sizeRev);
        cudaMemset(dRevData, EMPTY_CELL, sizeRev);
        hRevData = new int[sizeRev/sizeof(int)];

        cudaMalloc(&dTransitionLocks, stateCount * inputCount * sizeof(int));
        cudaMemset(dTransitionLocks, 0, stateCount * inputCount * sizeof(int));

        transCounts = new int[stateCount * inputCount];
        safeCounts = new int[stateCount * inputCount];
        precomputeTransitions();

    }

    ~TransitionTableHost() {
        cudaFree(dData);
        cudaFree(dRevData);
        delete[] hData;
        delete[] hRevData;
        delete[] transCounts;
        delete[] safeCounts;
    }

    void syncDeviceData() {
        cudaMemcpy(hData,
                   dData,
                   stateCount * inputCount * MAX_TRANSITIONS * sizeof(int),
                   cudaMemcpyDeviceToHost
        );

        cudaMemcpy(hRevData,
                   dRevData,
                   stateCount * inputCount * MAX_PREDECESSORS * sizeof(int),
                   cudaMemcpyDeviceToHost
        );
    }

    void precomputeTransitions(){
        for(int stateIdx = 0; stateIdx < stateCount; stateIdx++){
            for(int inputIdx = 0; inputIdx < inputCount; inputIdx++){
                safeCounts[getPosition(stateIdx, inputIdx)] = 0;
                int tot = 0;
                for(int i = 0; i<MAX_TRANSITIONS; i++){
                    if(get(stateIdx,inputIdx,i) != -1) tot++;
                
                }
                transCounts[getPosition(stateIdx, inputIdx)] = tot;
            }
        }
    }

    int getPosition(int stateIdx, int inputIdx) {
        return stateIdx * inputCount + inputIdx;
    }

    int getOffset(int stateIdx, int inputIdx, int transition = 0) const {
        return stateIdx * (inputCount * MAX_TRANSITIONS) + 
        inputIdx * MAX_TRANSITIONS + 
        transition;
    }
    int getRevOffset(int stateIdx, int inputIdx, int predecessor = 0) const {
        return stateIdx * (inputCount * MAX_PREDECESSORS) + 
        inputIdx * MAX_PREDECESSORS + 
        predecessor;
    }

    void set(int stateIdx, int inputIdx, int transition, int val) {
        hData[getOffset(stateIdx, inputIdx, transition)] = val;
    }
    void setRev(int stateIdx, int inputIdx, int predecessor, int val) {
        hRevData[getRevOffset(stateIdx, inputIdx, predecessor)] = val;
    }
    int get(int stateIdx, int inputIdx, int transition) const {
        return hData[getOffset(stateIdx, inputIdx, transition)];
    }
    int getRev(int stateIdx, int inputIdx, int predecessor) const {
        return hRevData[getRevOffset(stateIdx, inputIdx, predecessor)];
    }

    void removeTransitions(int stateIdx, int inputIdx) {
        for(int off = getOffset(stateIdx, inputIdx), _ = 0; _ < MAX_TRANSITIONS; _++, off++) {
            int dstIdx = hData[off];
            if(dstIdx == -1) continue;
            hData[off] = -1;

            for(int revOff = getOffset(stateIdx, inputIdx), _ = 0; _ < MAX_PREDECESSORS; _++, revOff++) {
                if(hRevData[revOff] == stateIdx) {
                    hRevData[revOff] = -1; 
                    break;
                }
            }
        }
    }
};

struct CudaCompilationContext {
    nvrtcProgram program;
    CUdevice cuDevice;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    bool isCooperative;


    CudaCompilationContext(const char* buildAutomatonCode, bool isCooperative): isCooperative(isCooperative) {
        // =============== Compile Program ===============
        nvrtcCreateProgram(&program,
                           buildAutomatonCode,
                           "buildAutomaton.cu",
                           0,
                           NULL,
                           NULL);

        std::vector<std::string> options;
        std::stringstream ss(CUDA_INCLUDE_DIRS);
        std::string path;
        while(ss >> path) options.push_back("--include-path=" + path);
        std::vector<const char*> opt_cstr;
        for(auto& s : options) opt_cstr.push_back(s.c_str());

        nvrtcResult compileResult = nvrtcCompileProgram(program, opt_cstr.size(), opt_cstr.data());


        // =============== Get Compilation Log ===============
        if (compileResult != NVRTC_SUCCESS) {
            size_t logSize;
            nvrtcGetProgramLogSize(program, &logSize);
            if (logSize > 1) {
                char *log = new char[logSize];
                nvrtcGetProgramLog(program, log);
                std::cerr << "Compilation failed:\n" << log << std::endl;
                delete[] log;
            } else {
                std::cerr << "Compilation failed but log is empty." << std::endl;
            }
        } else {
            std::cout << "Compilation succeeded." << std::endl;
        }

        // Get PTX from the program.
        size_t ptxSize;
        nvrtcGetPTXSize(program, &ptxSize);
        char *ptx = new char[ptxSize];
        nvrtcGetPTX(program, ptx);


        // =============== Get Kernel Handle ===============
        cuInit(0);
        cuDeviceGet(&cuDevice, 0);
        cuCtxCreate(&context, NULL, 0, cuDevice);
        cuModuleLoadDataEx(&module, ptx, 0, 0, 0);
        cuModuleGetFunction(&kernel, 
                            module, 
                            (isCooperative ? "buildAutomatonCoop": "buildAutomatonNonCoop")
                            );
    }


    void launchKernel(const py::object& stateSpace,       // DiscreteSpace
                      const py::object& inputSpace,       // DiscreteSpace
                      const py::object& disturbanceSpace, // ContinuousSpace
                      const py::tuple& maxDisturbJac,
                      const TransitionTableHost& table) const
    {
        
        SpaceInfoHost stateSpaceInfoHost{stateSpace};
        SpaceInfoDevice stateSpaceInfoDevice {
            stateSpaceInfoHost.dimensions,
            stateSpaceInfoHost.cellCount,
            stateSpaceInfoHost.dLowerBound,
            stateSpaceInfoHost.dUpperBound,
            stateSpaceInfoHost.dCellSize,
            stateSpaceInfoHost.dResolutions
        };


        SpaceInfoHost inputSpaceInfoHost{inputSpace};
        SpaceInfoDevice inputSpaceInfoDevice {
            inputSpaceInfoHost.dimensions,
            inputSpaceInfoHost.cellCount,
            inputSpaceInfoHost.dLowerBound,
            inputSpaceInfoHost.dUpperBound,
            inputSpaceInfoHost.dCellSize,
            inputSpaceInfoHost.dResolutions
        };


        SpaceBoundsHost disturbanceSpaceBoundsHost{disturbanceSpace};
        SpaceBoundsDevice disturbanceSpaceBoundsDevice {
            disturbanceSpaceBoundsHost.dimensions,
            disturbanceSpaceBoundsHost.dLowerBound,
            disturbanceSpaceBoundsHost.dUpperBound,
            disturbanceSpaceBoundsHost.dCenter
        };


        TransitionTableDevice tableDevice{
            table.dData,
            table.dRevData,
            table.dTransitionLocks,
            table.stateCount,
            table.inputCount
        };


        CUresult resKernelLaunch;
        if (isCooperative) {
            void *args[] =  {
                &stateSpaceInfoDevice,
                &inputSpaceInfoDevice,
                &disturbanceSpaceBoundsDevice,
                &tableDevice
            };

            resKernelLaunch = cuLaunchKernel(kernel,
                                             (int)std::ceil((float)stateSpaceInfoHost.cellCount / BLOCK_SIZE), 1, 1,
                                             BLOCK_SIZE, 1, 1,
                                             0, NULL,
                                             args,
                                             0
            );
        }
        else {
            int stateDim = stateSpaceInfoHost.dimensions;
            int disturbDim = disturbanceSpaceBoundsHost.dimensions;

            float hMaxDisturbJac[stateDim*disturbDim]; 

            for(int i = 0; i < stateDim; i++) {
                py::tuple row = maxDisturbJac[i].cast<py::tuple>();
                for (int j = 0; j < disturbDim; j++) {
                    hMaxDisturbJac[i*disturbDim + j] = row[j].cast<float>();
                }
            }

            float* dMaxDisturbJac;
            cudaMalloc(&dMaxDisturbJac, stateDim * disturbDim * sizeof(float));
            cudaMemcpy(dMaxDisturbJac, hMaxDisturbJac, stateDim * disturbDim * sizeof(float), cudaMemcpyHostToDevice);


            void *args[] =  {
                &stateSpaceInfoDevice,
                &inputSpaceInfoDevice,
                &disturbanceSpaceBoundsDevice,
                &dMaxDisturbJac,
                &tableDevice
            };


            resKernelLaunch = cuLaunchKernel(kernel,
                                             (int)std::ceil((float)stateSpaceInfoHost.cellCount / BLOCK_SIZE), 1, 1,
                                             BLOCK_SIZE, 1, 1,
                                             0, NULL,
                                             args,
                                             0
            );
        }

        if (resKernelLaunch != CUDA_SUCCESS) {
            const char *errorName = nullptr;
            cuGetErrorName(resKernelLaunch, &errorName);
            printf("cuLaunchKernel failed: %s\n", errorName);
        }
        else {
            printf("cuLaunchKernel succeeded.\n");
        }


        cuCtxSynchronize();
    }


    void cleanup() {
        nvrtcDestroyProgram(&program);
        cuModuleUnload(module);
        cuCtxDestroy(context);
    }
};
