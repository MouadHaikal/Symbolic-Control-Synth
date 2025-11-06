cmake_minimum_required(VERSION 3.15...4.0)
project(bindings LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(pybind11)

file(GLOB BINDINGS_SRC CONFIGURE_DEPENDS src/*.cpp)
file(GLOB BINDINGS_HEADERS CONFIGURE_DEPENDS src/*.hpp)

add_library(libindings STATIC ${BINDINGS_SRC})
target_include_directories(libindings PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(libindings PROPERTIES POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(ExternalModules main.cpp)
target_include_directories(ExternalModules 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Set output directory to symControl package
set_target_properties(ExternalModules PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/symControl
)
target_link_libraries(ExternalModules PRIVATE libindings)
