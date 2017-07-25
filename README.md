
# TIOGA + Trilinos STK Utilities

Contains test code to demonstrate [TIOGA](https://github.com/jsitaraman/tioga/)
to [STK](https://github.com/trilinos/Trilinos) integration.

## Building from source

The following dependencies must be available on your system:

- [Trilinos](https://github.com/trilinos/Trilinos)
- [TIOGA](https://github.com/jsitaraman/tioga)
- [Nalu](https://github.com/NaluCFD/Nalu)


See [`share/cmake_configure.sh`](https://github.com/sayerhs/tioga_utils/blob/master/share/cmake-configure.sh) for an example CMake configuration script. 

## Usage

```
mpiexec -np <NPROCS> stk2tioga <input_file>
```
