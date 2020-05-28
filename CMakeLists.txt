cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(TIOGA_UTILS CXX)

option(ENABLE_AMR_WIND "Enable interface to AMR-wind" ON)
option(ENABLE_NALU_WIND "Enable interface to nalu-wind" OFF)

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

add_subdirectory(src)
