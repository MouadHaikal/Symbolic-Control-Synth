cmake_minimum_required(VERSION 3.15...4.0)
project(bindings LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(pybind11)

file(GLOB SRC_FILES src/*.cpp src/*.cu)

pybind11_add_module(bindings ${SRC_FILES})


set_target_properties(bindings PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/symControl
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/symControl
)


find_package(CUDAToolkit REQUIRED)


target_link_libraries(bindings PRIVATE 
    CUDA::cudart
    cuda
    CUDA::nvrtc
)

set(NVRTC_INCLUDE_DIRS
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_SOURCE_DIR}/cpp/src
)
string(JOIN " " CUDA_INCLUDE_DIRS_STR ${NVRTC_INCLUDE_DIRS})

target_compile_definitions(bindings PRIVATE
    CUDA_INCLUDE_DIRS="${CUDA_INCLUDE_DIRS_STR}"
)
