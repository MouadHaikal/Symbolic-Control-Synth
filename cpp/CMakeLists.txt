cmake_minimum_required(VERSION 3.15...4.0)
project(bindings LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


add_subdirectory(pybind11)

# set(BUILD_TESTS OFF)
# set(BUILD_BENCHMARKS OFF)
# set(BUILD_EXAMPLES OFF)
# add_subdirectory(cuCollections)
#
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
    # cuco
)
