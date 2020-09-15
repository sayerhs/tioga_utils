target_compile_features(tioga_stk PRIVATE cxx_std_11)
set_target_properties(tioga_stk PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(tioga_stk
  SYSTEM PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${Trilinos_INCLUDE_DIRS}
  ${Trilinos_TPL_INCLUDE_DIRS}
  ${TIOGA_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
  )
target_link_libraries(tioga_stk PUBLIC
  $<$<BOOL:${MPI_CXX_FOUND}>:MPI::MPI_CXX>
  ${Trilinos_LIBRARIES}
  ${TIOGA_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  )

if (ENABLE_NALU_WIND)
  target_compile_definitions(tioga_stk PUBLIC HAS_NALU_WIND)
  target_include_directories(tioga_stk
    SYSTEM PUBLIC
    ${NALU_INCLUDES})
  target_link_libraries(${NALU_LIBRARIES})
endif()

target_sources(tioga_stk PRIVATE
  TiogaRef.cpp
  TiogaBlock.cpp
  TiogaSTKIface.cpp
  StkIface.cpp

  MotionBase.cpp
  MeshRotation.cpp
  MeshMotion.cpp
  Timer.cpp
  )

target_sources(stk2tioga PRIVATE stk2tioga.cpp)
target_link_libraries(stk2tioga PUBLIC tioga_stk)

if (TU_ENABLE_CUDA)
  target_link_libraries(tioga_stk PUBLIC
    ${CUDA_cusparse_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES})
  separate_arguments(Trilinos_CXX_COMPILER_FLAGS)
  target_compile_options(tioga_stk PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${Trilinos_CXX_COMPILER_FLAGS}>)
  target_compile_options(tioga_stk PUBLIC $<$<COMPILE_LANGUAGE:CXX>:--expt-relaxed-constexpr>)
endif()

if (ENABLE_AMR_WIND)
  target_include_directories(tioga_amr
    SYSTEM PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Trilinos_INCLUDE_DIRS}
    ${Trilinos_TPL_INCLUDE_DIRS}
    ${TIOGA_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
    )
  add_subdirectory(amr)

  if (TU_ENABLE_CUDA)
    separate_arguments(Trilinos_CXX_COMPILER_FLAGS)
    target_compile_options(tioga_amr PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${Trilinos_CXX_COMPILER_FLAGS}>)
    target_compile_options(tioga_amr PUBLIC $<$<COMPILE_LANGUAGE:CXX>:--expt-relaxed-constexpr>)
  endif()

  target_sources(exatioga PRIVATE
    ExaTioga.cpp
    exatioga_main.cpp)
  target_link_libraries(exatioga PUBLIC tioga_stk tioga_amr)
endif()
