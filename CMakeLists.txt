cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(TIOGA_UTILS CXX)

option(ENABLE_AMR_WIND "Enable interface to AMR-wind" ON)
option(ENABLE_NALU_WIND "Enable interface to nalu-wind" OFF)
option(TU_ENABLE_CUDA "Enable CUDA builds" OFF)

set(TU_CUDA_SM "70" CACHE STRING "CUDA arch option")

if (TU_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "9.0")
    message(FATAL_ERROR "Your nvcc version is ${CMAKE_CUDA_COMPILER_VERSION}. "
      "Please use CUDA toolkit version 9.0 or newer")
  endif()
  set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-gencode=arch=compute_${TU_CUDA_SM},code=sm_${TU_CUDA_SM}")
endif()

if (ENABLE_AMR_WIND)
  add_subdirectory(ext)
endif()

find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED)

set(CMAKE_PREFIX_PATH ${Trilinos_DIR} ${CMAKE_PREFIX_PATH})
find_package(Trilinos REQUIRED)

set(CMAKE_PREFIX_PATH ${TIOGA_DIR} ${CMAKE_PREFIX_PATH})
find_package(TIOGA REQUIRED)

set(CMAKE_PREFIX_PATH ${YAML_DIR} ${CMAKE_PREFIX_PATH})
find_package(YAML-CPP REQUIRED)

if (ENABLE_NALU_WIND)
  find_library(NALU_LIBRARIES NAMES nalu libnalu.a PATHS ${NALU_DIR}/lib ${NALU_DIR}/build)
  find_path(NALU_INCLUDES Realm.h PATHS ${NALU_DIR}/include)
endif()

add_library(tioga_stk OBJECT)
add_executable(stk2tioga)

if (ENABLE_AMR_WIND)
  add_library(tioga_amr OBJECT)
  add_executable(exatioga)
endif()

add_subdirectory(src)
