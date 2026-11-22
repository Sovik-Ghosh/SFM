cmake_minimum_required(VERSION 3.24)

project(SFM
  VERSION 1.0.2
  DESCRIPTION "Incremental Structure-from-Motion, C++/CUDA runtime"
  LANGUAGES CXX CUDA)

option(SFM_WERROR "Treat warnings in project code as errors" OFF)
option(SFM_BUILD_TESTS "Build unit tests" ON)
option(SFM_BUILD_VIEWER "Build apps/sfm_viewer (needs GLFW 3 and OpenGL dev packages)" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(CompilerWarnings)
include(ThirdParty)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

if(SFM_BUILD_TESTS)
  enable_testing()
endif()

add_subdirectory(inference)
