#include "automaton.hpp"
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <nvrtc.h>
#include <cstddef>
#include <iostream>

#define BLOCK_SIZE 512


// py::scoped_interpreter guard{};
// py::object DiscreteSpace = py::module_::import("symControl.space.discreteSpace").attr("DiscreteSpace");
// py::object ContinuousSpace = py::module_::import("symControl.space.discreteSpace").attr("ContinuousSpace"); 



Automaton::Automaton(py::object stateSpace,       // DiscreteSpace
                     py::object inputSpace,       // DiscreteSpace
                     py::object disturbanceSpace, // ContinuousSpace
                     bool isCooperative,
                     py::tuple maxDisturbJac,
                     const char* buildAutomatonCode)
    : table(stateSpace.attr("cellCount").cast<size_t>(), inputSpace.attr("cellCount").cast<size_t>())
{ 
    // py::scoped_interpreter guard{};
    if (stateSpace.attr("dimensions").cast<int>() > MAX_DIMENSIONS) {
        printf("Error: State space dimensions (%d) > MAX_DIMENSIONS (%d) (at %s)\n", 
               stateSpace.attr("dimensions").cast<int>(), 
               MAX_DIMENSIONS,
               __FILE__
        );

        exit(EXIT_FAILURE);
    }


    // =============== Compile Program ===============
    nvrtcProgram prog;
    nvrtcCreateProgram(&prog,
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

    nvrtcResult compileResult = nvrtcCompileProgram(prog, opt_cstr.size(), opt_cstr.data());



    // =============== Get Compilation Log ===============
    if (compileResult != NVRTC_SUCCESS) {
        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        if (logSize > 1) {
            char *log = new char[logSize];
            nvrtcGetProgramLog(prog, log);
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
    nvrtcGetPTXSize(prog, &ptxSize);
    char *ptx = new char[ptxSize];
    nvrtcGetPTX(prog, ptx);


    // =============== Get Kernel Handle ===============
    CUdevice cuDevice;
    CUcontext context;
    CUmodule module;
    CUfunction buildAutomaton;

    cuInit(0);
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&context, NULL, 0, cuDevice);
    cuModuleLoadDataEx(&module, ptx, 0, 0, 0);
    cuModuleGetFunction(&buildAutomaton, 
                        module, 
                        (isCooperative ? "buildAutomatonCoop": "buildAutomatonNonCoop")
    );




    // =============== Call Kernel ===============
    // stateSpaceInfo creation
    int stateDimensions = stateSpace.attr("dimensions").cast<int>();
    float stateLowerBound[stateDimensions];
    float stateUpperBound[stateDimensions];
    int stateResolutions[stateDimensions];
    float stateCellSize[stateDimensions];
    int stateCellCount = stateSpace.attr("cellCount").cast<int>();
    {
        py::tuple pyStateBounds = stateSpace.attr("bounds").cast<py::tuple>();
        for(int i = 0; i < stateDimensions; i++) {
            py::tuple inner = pyStateBounds[i].cast<py::tuple>();
            stateLowerBound[i] = inner[0].cast<float>();
            stateUpperBound[i] = inner[1].cast<float>();
        }
        py::tuple pyStateResolutions = stateSpace.attr("resolutions").cast<py::tuple>();
        for(int i = 0; i < stateDimensions; i++) {
            stateResolutions[i] = pyStateResolutions[i].cast<int>();
        }
        py::tuple pyStateCellSize = stateSpace.attr("cellSize").cast<py::tuple>();
        for(int i = 0; i < stateDimensions; i++) {
            stateCellSize[i] = pyStateCellSize[i].cast<float>();
        }
    }
    SpaceInfoHost stateSpaceInfoHost {
        stateDimensions,
        stateLowerBound,
        stateUpperBound,
        stateResolutions,
        stateCellSize,
        stateCellCount
    };
    SpaceInfoDevice stateSpaceInfoDevice {
        stateSpaceInfoHost.dimensions,
        stateSpaceInfoHost.dLowerBound,
        stateSpaceInfoHost.dUpperBound,
        stateSpaceInfoHost.dCellSize,
        stateSpaceInfoHost.dResolutions,
        stateSpaceInfoHost.cellCount
    };


    // inputSpaceInfo creation
    int inputDimensions = inputSpace.attr("dimensions").cast<int>();
    float inputLowerBound[inputDimensions];
    float inputUpperBound[inputDimensions];
    int inputResolutions[inputDimensions];
    float inputCellSize[inputDimensions];
    int inputCellCount = inputSpace.attr("cellCount").cast<int>();
    {
        py::tuple pyInputBounds = inputSpace.attr("bounds").cast<py::tuple>();
        for(int i = 0; i < inputDimensions; i++) {
            py::tuple inner = pyInputBounds[i].cast<py::tuple>();
            inputLowerBound[i] = inner[0].cast<float>();
            inputUpperBound[i] = inner[1].cast<float>();
        }
        py::tuple pyInputResolutions = inputSpace.attr("resolutions").cast<py::tuple>();
        for(int i = 0; i < inputDimensions; i++) {
            inputResolutions[i] = pyInputResolutions[i].cast<int>();
        }
        py::tuple pyInputCellSize = inputSpace.attr("cellSize").cast<py::tuple>();
        for(int i = 0; i < inputDimensions; i++) {
            inputCellSize[i] = pyInputCellSize[i].cast<float>();
        }
    }
    SpaceInfoHost inputSpaceInfoHost {
        inputDimensions,
        inputLowerBound,
        inputUpperBound,
        inputResolutions,
        inputCellSize,
        inputCellCount
    };
    SpaceInfoDevice inputSpaceInfoDevice {
        inputSpaceInfoHost.dimensions,
        inputSpaceInfoHost.dLowerBound,
        inputSpaceInfoHost.dUpperBound,
        inputSpaceInfoHost.dCellSize,
        inputSpaceInfoHost.dResolutions,
        inputSpaceInfoHost.cellCount
    };

    // disturbanceSpaceBound creation
    int disturbanceDimensions = disturbanceSpace.attr("dimensions").cast<int>();
    float disturbanceLowerBound[disturbanceDimensions];
    float disturbanceUpperBound[disturbanceDimensions];
    float disturbanceCenter[disturbanceDimensions];
    {
        py::tuple pyDisturbanceBounds = disturbanceSpace.attr("bounds").cast<py::tuple>();
        for(int i = 0; i < disturbanceDimensions; i++) {
            py::tuple inner = pyDisturbanceBounds[i].cast<py::tuple>();
            float lower = inner[0].cast<float>();
            float upper = inner[1].cast<float>();
            disturbanceLowerBound[i] = lower;
            disturbanceUpperBound[i] = upper;
            disturbanceCenter[i] = (lower + upper) / 2.0;
        }
    }
    SpaceBoundsHost disturbanceSpaceBoundsHost {
        disturbanceDimensions,
        disturbanceLowerBound,
        disturbanceUpperBound,
        disturbanceCenter
    };
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


    if (isCooperative) {
        void *args[] =  {
            &stateSpaceInfoDevice,
            &inputSpaceInfoDevice,
            &disturbanceSpaceBoundsDevice,
            &tableDevice
        };

        cuLaunchKernel(buildAutomaton,
                       (int)std::ceil((float)stateCellCount / BLOCK_SIZE), 1, 1,
                       BLOCK_SIZE, 1, 1,
                       0, NULL,
                       args,
                       0
        );
    }
    else {
        float hMaxDisturbJac[stateDimensions*disturbanceDimensions]; 

        for(int i = 0; i < stateDimensions; i++) {
            py::tuple row = maxDisturbJac[i].cast<py::tuple>();
            for (int j = 0; j < disturbanceDimensions; j++) {
                hMaxDisturbJac[i*disturbanceDimensions + j] = row[j].cast<float>();
            }
        }

        float* dMaxDisturbJac;
        cudaMalloc(&dMaxDisturbJac, stateDimensions * disturbanceDimensions * sizeof(float));
        cudaMemcpy(dMaxDisturbJac, hMaxDisturbJac, stateDimensions * disturbanceDimensions * sizeof(float), cudaMemcpyHostToDevice);


        void *args[] =  {
            &stateSpaceInfoDevice,
            &inputSpaceInfoDevice,
            &disturbanceSpaceBoundsDevice,
            &dMaxDisturbJac,
            &tableDevice
        };


        CUresult res = cuLaunchKernel(buildAutomaton,
                       (int)std::ceil((float)stateCellCount / BLOCK_SIZE), 1, 1,
                       BLOCK_SIZE, 1, 1,
                       0, NULL,
                       args,
                       0
        );

        if (res != CUDA_SUCCESS) {
            const char *errorName = nullptr;
            cuGetErrorName(res, &errorName);
            printf("cuLaunchKernel failed: %s\n", errorName);
        }
        else {
            printf("cuLaunchKernel succeeded.\n");
        }
    }


    cuCtxSynchronize();


    // =============== Cleanup ===============
    nvrtcDestroyProgram(&prog);
    cuModuleUnload(module);
    cuCtxDestroy(context);









    // =============== Testing ===============
    // printf("=== Combined Direct + Reverse Table ===\n");
    //
    // int stateCount = 10;
    // int inputCount = inputSpaceInfoHost.cellCount;
    //
    // // Host buffers
    // int hDir[stateCount * inputCount * MAX_TRANSITIONS];
    // int hRev[stateCount * inputCount * MAX_PREDECESSORS];
    //
    // // Copy forward transitions
    // cudaMemcpy(
    //     hDir,
    //     table.dData,
    //     stateCount * inputCount * MAX_TRANSITIONS * sizeof(int),
    //     cudaMemcpyDeviceToHost
    // );
    //
    // // Copy reverse transitions
    // cudaMemcpy(
    //     hRev,
    //     table.dRevData,
    //     stateCount * inputCount * MAX_PREDECESSORS * sizeof(int),
    //     cudaMemcpyDeviceToHost
    // );
    //
    // for (int s = 0; s < stateCount; s++) {
    //     for (int u = 0; u < inputCount; u++) {
    //
    //         printf("\n==============================\n");
    //         printf("     State %d | Input %d\n", s, u);
    //         printf("==============================\n");
    //
    //         int baseDir = (s * inputCount + u) * MAX_TRANSITIONS;
    //         int baseRev = (s * inputCount + u) * MAX_PREDECESSORS;
    //
    //         // ---- DIRECT ----
    //         printf("Direct : ");
    //         bool dirEmpty = true;
    //         for (int t = 0; t < MAX_TRANSITIONS; t++) {
    //             int v = hDir[baseDir + t];
    //             if (v != -1) dirEmpty = false;
    //         }
    //         if (dirEmpty) {
    //             printf("[ none ]\n");
    //         } else {
    //             for (int t = 0; t < MAX_TRANSITIONS; t++) {
    //                 int v = hDir[baseDir + t];
    //                 if (v != -1) printf("%d ", v);
    //             }
    //             printf("\n");
    //         }
    //
    //         // ---- REVERSE ----
    //         printf("Reverse: ");
    //         bool revEmpty = true;
    //         for (int p = 0; p < MAX_PREDECESSORS; p++) {
    //             int v = hRev[baseRev + p];
    //             if (v != -1) revEmpty = false;
    //         }
    //         if (revEmpty) {
    //             printf("[ none ]\n");
    //         } else {
    //             for (int p = 0; p < MAX_PREDECESSORS; p++) {
    //                 int v = hRev[baseRev + p];
    //                 if (v != -1) printf("%d ", v);
    //             }
    //             printf("\n");
    //         }
    //
    //         printf("-----------------------------------\n");
    //     }
    // }
    //
    // printf("\n=== End Combined Table ===\n");
}

Automaton::~Automaton() {

}
