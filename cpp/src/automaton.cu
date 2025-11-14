#include <random>
#include "automaton.cuh"
#include <nvrtc.h>
#include <iostream>


// py::scoped_interpreter guard{};
py::object DiscreteSpace = py::module_::import("symControl.space.discreteSpace").attr("DiscreteSpace");
py::object ContinuousSpace = py::module_::import("symControl.space.discreteSpace").attr("ContinuousSpace"); 



Automaton::Automaton(py::object stateSpace,       // DiscreteSpace
                     py::object controlSpace,     // DiscreteSpace
                     py::object disturbanceSpace, // ContinuousSpace
                     const char* fAtPointCode)
{

    // std::cout << stateSpace.attr("dimensions").cast<int>() << std::endl;
    nvrtcProgram prog;
    nvrtcCreateProgram(&prog,
                       fAtPointCode,
                       "fAtPoint.cu",
                       0,
                       NULL,
                       NULL);

    nvrtcCompileProgram(prog, 0, nullptr);

    // Obtain compilation log from the program.
    size_t logSize;
    nvrtcGetProgramLogSize(prog, &logSize);
    char *log = new char[logSize];
    nvrtcGetProgramLog(prog, log);
    printf("%s\n", log);
    // Obtain PTX from the program.
    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char *ptx = new char[ptxSize];
    nvrtcGetPTX(prog, ptx);
    // end of log

    cuInit(0);
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&context, NULL, 0, cuDevice);
    cuModuleLoadDataEx(&module, ptx, 0, 0, 0);
    cuModuleGetFunction(&testKernel, module, "testKernel");

    int stateDim = stateSpace.attr("dimensions").cast<int>();
    int controlDim = controlSpace.attr("dimensions").cast<int>();
    int disturbanceDim = disturbanceSpace.attr("dimensions").cast<int>();






    int inputCount = 100;
    std::random_device rd;
    std::mt19937 gen(rd());

    float stateBounds[stateDim][2];
    float controlBounds[controlDim][2];
    float disturbanceBounds[disturbanceDim][2];

    py::tuple pyStateBounds = stateSpace.attr("bounds").cast<py::tuple>();
    for(int i = 0; i < stateDim; i++) {
        py::tuple inner = pyStateBounds[i].cast<py::tuple>();
        stateBounds[i][0] = inner[0].cast<float>();
        stateBounds[i][1] = inner[1].cast<float>();
    }

    py::tuple pyControlBounds = controlSpace.attr("bounds").cast<py::tuple>();
    for(int i = 0; i < controlDim; i++) {
        py::tuple inner = pyControlBounds[i].cast<py::tuple>();
        controlBounds[i][0] = inner[0].cast<float>();
        controlBounds[i][1] = inner[1].cast<float>();
    }

    py::tuple pyDisturbanceBounds = disturbanceSpace.attr("bounds").cast<py::tuple>();
    for(int i = 0; i < disturbanceDim; i++) {
        py::tuple inner = pyDisturbanceBounds[i].cast<py::tuple>();
        disturbanceBounds[i][0] = inner[0].cast<float>();
        disturbanceBounds[i][1] = inner[1].cast<float>();
    }

    float states[stateDim][inputCount];
    float controls[controlDim][inputCount];
    float disturbances[disturbanceDim][inputCount];
    float nextStates[stateDim][inputCount];

    for (int d = 0; d < stateDim; d++) {
        std::uniform_real_distribution<float> dist(stateBounds[d][0], 
                                                   stateBounds[d][1]);
        for (int i = 0; i < inputCount; i++) {
            states[d][i] = dist(gen);
        }
    }

    for (int d = 0; d < controlDim; d++) {
        std::uniform_real_distribution<float> dist(controlBounds[d][0],
                                                   controlBounds[d][1]);
        for (int i = 0; i < inputCount; i++) {
            controls[d][i] = dist(gen);
        }
    }

    for (int d = 0; d < disturbanceDim; d++) {
        std::uniform_real_distribution<float> dist(disturbanceBounds[d][0],
                                                   disturbanceBounds[d][1]);
        for (int i = 0; i < inputCount; i++) {
            disturbances[d][i] = dist(gen);
        }
    }

    CUdeviceptr dStates, dControls, dDisturbances, dNextStates;
    cuMemAlloc(&dStates, stateDim * inputCount * sizeof(float));
    cuMemAlloc(&dControls, controlDim * inputCount * sizeof(float));
    cuMemAlloc(&dDisturbances, disturbanceDim * inputCount * sizeof(float));
    cuMemAlloc(&dNextStates, stateDim * inputCount * sizeof(float));
    
    cuMemcpyHtoD(dStates, states, sizeof(states));
    cuMemcpyHtoD(dControls, controls, sizeof(controls));
    cuMemcpyHtoD(dDisturbances, disturbances, sizeof(disturbances));

    void *args[] = {
        &inputCount,
        &stateDim,
        &dStates,
        &controlDim,
        &dControls,
        &disturbanceDim,
        &dDisturbances,
        &dNextStates
    };


    int blockSize = 32;
    cuLaunchKernel(testKernel,
                   (int)std::ceil((float)inputCount / blockSize), 1, 1,
                   blockSize, 1, 1,
                   0, NULL,
                   args,
                   0);

    cuCtxSynchronize();
    cuMemcpyDtoH(nextStates, dNextStates, sizeof(states));

    std::cout << nextStates[0][0] << std::endl;

    nvrtcDestroyProgram(&prog);

    cuMemFree(dStates);
    cuMemFree(dControls);
    cuMemFree(dDisturbances);
    cuMemFree(dNextStates);

    cuModuleUnload(module);
    cuCtxDestroy(context);
}

Automaton::~Automaton() {

}

