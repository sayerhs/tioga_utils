# ExaWind TIOGA utilities

This repository contains utilities to demonstrate overset connectivity within
the [ExaWind](https://github.com/exawind) project using
[TIOGA](https://github.com/jsitaraman/tioga) between unstructured meshes in
Exodus-II format using [STK](https://github.com/trilinos/Trilinos) and
block-structured meshes using [AMReX](https://github.com/AMReX-Codes/amrex)

The code generates two executables:

- `stk2tioga` -- Standalone utility to test overset connectivity between
  unstructured meshes that are all in one Exodus file.
  
- `exatioga` -- Utility to test overset connectivity between multiple
  unstructured meshes (provided in Exodus-II format) with a single AMR
  background mesh with multiple levels of refinement.

## Building from source

The following dependencies must be available on your system:

- [TIOGA](https://github.com/jsitaraman/tioga) -- For use with AMR meshes, you
  should checkout the `exawind` branch of TIOGA.
- [Trilinos](https://github.com/trilinos/Trilinos)
- [AMReX](https://github.com/AMReX-Codes/amrex)

```console
# Clone the git repository
git clone --recurse-submodules -b exawind git@github.com:sayerhs/tioga_utils.git

# In an already cloned repository
git fetch origin
git checkout --track origin/exawind
git submodule update --init --recursive
```

We recommend using
[exawind-builder](https://exawind-builder.readthedocs.io/en/latest/) to
configure and build the dependencies as well as `tioga-utils`.
