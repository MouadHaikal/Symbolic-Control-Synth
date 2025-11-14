cmake_minimum_required(VERSION 3.15...4.0)
project(bindings LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(pybind11)

file(GLOB SRC_FILES src/*.cpp)
pybind11_add_module(bindings ${SRC_FILES})

set_target_properties(bindings PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/symControl
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/symControl
)
