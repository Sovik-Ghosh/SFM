cmake_minimum_required(VERSION 3.24)

project(SFM
  VERSION 1.4.0
  DESCRIPTION "Incremental Structure-from-Motion, C++/CUDA runtime"
  LANGUAGES CXX)

# OFF: CPU-only build, no CUDA toolkit needed (the CUDA libraries, apps and tests are left out;
# every stage still has its CPU method).
option(SFM_WITH_CUDA "Build the CUDA stages (needs the CUDA toolkit)" ON)
option(SFM_WERROR "Treat warnings in project code as errors" OFF)
option(SFM_BUILD_TESTS "Build unit tests" ON)
option(SFM_BUILD_VIEWER "Build apps/sfm_viewer (needs GLFW 3 and OpenGL dev packages)" ON)
option(SFM_BUILD_PYTHON "Build the Python module epipole (pybind11; pyproject.toml turns it on)" OFF)
if(SFM_BUILD_PYTHON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # the static libraries go into a shared module
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(SFM_WITH_CUDA)
  # Default: Turing to Blackwell consumer GPUs (RTX 20xx 75, 30xx 86, 40xx 89, 50xx 120), plus PTX
  # of each so newer GPUs JIT-compile. sm_72 is the minimum (u8 tensor-core matching). A GPU left
  # out of the list makes the run fall back to the CPU (logged as run.backend_fallback).
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;86;89;120")
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Installed programs look for bundled libraries in <prefix>/lib, then the system paths (not the
# build machine's CUDA directory), so an install tree or release archive can be moved.
if(NOT DEFINED CMAKE_INSTALL_RPATH)
  set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(CompilerWarnings)
include(ThirdParty)

if(SFM_WITH_CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()
find_package(Threads REQUIRED)

if(SFM_BUILD_TESTS)
  enable_testing()
endif()

add_subdirectory(inference)
if(SFM_BUILD_PYTHON)
  add_subdirectory(python)
endif()
