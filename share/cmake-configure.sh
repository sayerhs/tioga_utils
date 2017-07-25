#!/bin/bash

# Example configuration script for TIOGA Utilities

# Set the Spack compiler flavor
COMPILER=gcc

# Point to appropriate spack installed libraries or local installations
trilinos_install_dir=$(spack location -i nalu-trilinos %${COMPILER})
yaml_install_dir=$(spack location -i yaml-cpp %${COMPILER})
tioga_install_dir=${HOME}/code/tioga/install/
nalu_install_dir=$(spack location -i nalu %${COMPILER})

EXTRA_ARGS="$@"

cmake \
    -DTrilinos_DIR:PATH=$trilinos_install_dir \
    -DYAML_DIR:PATH=$yaml_install_dir \
    -DCMAKE_BUILD_TYPE=Release \
    $EXTRA_ARGS \
    ../src
