
cmake_minimum_required(VERSION 2.8.11)

project(TIOGA_UTILS CXX Fortran)

find_package(MPI REQUIRED)

set(CMAKE_PREFIX_PATH ${Trilinos_DIR} ${CMAKE_PREFIX_PATH})
set(CMAKE_PREFIX_PATH ${Trilinos_PATH} ${CMAKE_PREFIX_PATH})
find_package(Trilinos REQUIRED)

find_library(YAML_LIBRARIES NAMES yaml-cpp libyaml-cpp.a PATHS ${YAML_DIR}/lib)
find_path(YAML_INCLUDES yaml.h PATHS ${YAML_DIR}/include/yaml-cpp)

set(CMAKE_PREFIX_PATH ${TIOGA_DIR} ${CMAKE_PREFIX_PATH})
find_package(TIOGA REQUIRED)

find_library(NALU_LIBRARIES NAMES nalu libnalu.a PATHS ${NALU_DIR}/lib ${NALU_DIR}/build)
find_path(NALU_INCLUDES Realm.h PATHS ${NALU_DIR}/include)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(SYSTEM ${Trilinos_INCLUDE_DIRS})
include_directories(SYSTEM ${Trilinos_TPL_INCLUDE_DIRS})
include_directories(SYSTEM ${YAML_INCLUDES}/..)
include_directories(${TIOGA_INCLUDE_DIRS})
include_directories(${NALU_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR})

set(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Trilinos_CXX_COMPILER_FLAGS}")

# Fix flags for Intel C++ compiler
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -restrict ")
endif()

add_executable(stk2tioga
  TiogaBlock.cpp
  TiogaSTKIface.cpp

  MotionBase.cpp
  MeshRotation.cpp
  MeshMotion.cpp
  Timer.cpp

  stk2tioga.cpp)
target_link_libraries(stk2tioga
  ${NALU_LIBRARIES}
  ${Trilinos_LIBRARIES}
  ${YAML_LIBRARIES}
  ${TIOGA_LIBRARIES})
