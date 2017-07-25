#
#	This is the top level CMakeLists.txt file
#	It should be located within master/nalu
#	But not in the BUILD folder itself
#
###########################################################################################################

cmake_minimum_required (VERSION 2.8.11)

###########################################################################################################
#
#  THESE PATHS MAY NEED TO BE CHANGED DEPENDING ON WHERE THEY ARE STORED!!!
#

SET(CMAKE_FIND_LIBRARY_PREFIXES lib)
SET(CMAKE_FIND_LIBRARY_SUFFIXES a)

MESSAGE("   Trilinos_DIR = ${Trilinos_DIR}")
SET(CMAKE_PREFIX_PATH ${Trilinos_DIR} ${CMAKE_PREFIX_PATH})
SET(CMAKE_PREFIX_PATH ${Trilinos_PATH} ${CMAKE_PREFIX_PATH})
find_package(Trilinos REQUIRED)

# This needs to be cleaned up
find_library(YAML_LIBRARY NAMES libyaml-cpp.a PATHS ${YAML_DIR}/lib )
find_path(YAML_INCLUDES yaml.h PATHS ${YAML_DIR}/include/yaml-cpp )
include_directories(${YAML_INCLUDES}/..)

# find_library(TIOGA_LIBS NAMES libtioga.a PATHS ${TIOGA_DIR}/build)
# find_path(TIOGA_INCLUDES tioga.h PATHS ${TIOGA_DIR}/src)
# include_directories(${TIOGA_DIR}/src)

set(CMAKE_PREFIX_PATH ${TIOGA_DIR} ${CMAKE_PREFIX_PATH})
find_package(TIOGA REQUIRED)
include_directories(${TIOGA_INCLUDE_DIRS})

find_library(NALU_LIBS NAMES libnalu.a PATHS ${NALU_DIR}/build)
find_path(NALU_INCLUDES Realm.h PATHS ${NALU_DIR}/include)
include_directories(${NALU_DIR}/include)


include_directories(${Trilinos_INCLUDE_DIRS})
include_directories(${Trilinos_TPL_INCLUDE_DIRS})

#######################################     TRILINOS	 ##################################################

MESSAGE("\nFound Trilinos!  Here are the details: ")
# Uncomment lines to view more details about Trilinos
MESSAGE("   Trilinos_DIR = ${Trilinos_DIR}")
MESSAGE("   Trilinos_VERSION = ${Trilinos_VERSION}")
MESSAGE("   Trilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
#MESSAGE("   Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
#MESSAGE("   Trilinos_BIN_DIRS = ${Trilinos_BIN_DIRS}")
#MESSAGE("   Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
#MESSAGE("   Trilinos_LIBRARY_DIRS = ${Trilinos_LIBRARY_DIRS}")
#MESSAGE("   Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
#MESSAGE("   Trilinos_TPL_INCLUDE_DIRS = ${Trilinos_TPL_INCLUDE_DIRS}")
#MESSAGE("   Trilinos_TPL_LIBRARIES = ${Trilinos_TPL_LIBRARIES}")
#MESSAGE("   Trilinos_TPL_LIBRARY_DIRS = ${Trilinos_TPL_LIBRARY_DIRS}")
#MESSAGE("   Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
MESSAGE("   Trilinos_CXX_COMPILER_FLAGS = ${Trilinos_CXX_COMPILER_FLAGS}")
MESSAGE("End of Trilinos details\n")

MESSAGE("Setting and checking of compilers:")
SET(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER} )
SET(CMAKE_C_COMPILER ${Trilinos_C_COMPILER} )
SET(CMAKE_Fortran_COMPILER ${Trilinos_Fortran_COMPILER} )

# Build Nalu as shared libraries if Trilinos was compiled that way

IF(Trilinos_BUILD_SHARED_LIBS)
  SET(Nalu_LIBRARY_TYPE SHARED)
  SET(BUILD_SHARED_LIBS ON)
  MESSAGE("-- Building Nalu with shared libraries")
ELSE(Trilinos_BUILD_SHARED_LIBS)
  SET(Nalu_LIBRARY_TYPE STATIC)
ENDIF(Trilinos_BUILD_SHARED_LIBS)

#######################################     NALU     ##########################

PROJECT(Nalu)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Trilinos_CXX_COMPILER_FLAGS}")
SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${Trilinos_Fortran_COMPILER_FLAGS}")
MESSAGE("-- CMAKE_CXX_FLAGS     = ${CMAKE_CXX_FLAGS}")
MESSAGE("-- CMAKE_Fortran_FLAGS = ${CMAKE_Fortran_FLAGS}")

add_executable(stk2tioga
  TiogaBlock.cpp
  TiogaSTKIface.cpp
  stk2tioga.cpp)
target_link_libraries(stk2tioga ${NALU_LIBS} ${Trilinos_LIBRARIES} ${YAML_LIBRARY} ${TIOGA_LIBRARIES})
