set(AMREX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/amrex")
include(${CMAKE_SOURCE_DIR}/cmake/set_amrex_options.cmake)
list(APPEND CMAKE_MODULE_PATH "${AMREX_PATH}/Tools/CMake")
add_subdirectory(${AMREX_PATH})

