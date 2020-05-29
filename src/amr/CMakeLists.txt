target_sources(tioga_amr PRIVATE
  amrex_yaml.cpp

  FieldRepo.cpp
  StructMesh.cpp
  TiogaAMRIface.cpp

  CartBoxRefinement.cpp
  )

target_include_directories(tioga_amr PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tioga_amr PUBLIC amrex)
