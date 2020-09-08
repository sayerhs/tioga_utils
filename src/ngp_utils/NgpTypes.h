#ifndef NGPTYPES_H
#define NGPTYPES_H

#include "stk_mesh/base/NgpMesh.hpp"

namespace tioga_nalu {
namespace ngp {

#ifdef KOKKOS_ENALE_CUDA
using MemSpace = Kokkos::CudaSpace;
using UVMSpace = Kokkos::CudaUVMSpace;
#elif defined(KOKKOS_ENABLE_OPENMP)
using MemSpace = Kokkos::OpenMP;
using UVMSpace = Kokkos::OpenMP;
#else
using MemSpace = Kokkos::HostSpace;
using MemSpace = Kokkos::HostSpace;
#endif

using HostSpace = Kokkos::DefaultHostExecutionSpace;
using DeviceSpace = Kokkos::DefaultExecutionSpace;
using MemLayout = Kokkos::LayoutRight;

template <typename T, typename Layout = MemLayout, typename Space = MemSpace>
using NgpArray = Kokkos::View<T, Layout, Space>;

template <typename T, typename Layout = MemLayout, typename Space = MemSpace>
struct NgpDualArray
{
    using ArrayType = NgpArray<T, Layout, Space>;
    using HostArrayType = typename ArrayType::HostMirror;

    ArrayType d_view;
    HostArrayType h_view;

    NgpDualArray() : d_view(), h_view() {}

    NgpDualArray(const std::string& label, const size_t len)
        : d_view(label, len),
          h_view(Kokkos::create_mirror_view(d_view))
    {}

    void init(const std::string& label, const size_t len)
    {
        d_view = ArrayType(label, len);
        h_view = Kokkos::create_mirror_view(d_view);
    }

    void sync_to_device()
    {
        Kokkos::deep_copy(d_view, h_view);
    }

    void sync_to_host()
    {
        Kokkos::deep_copy(h_view, d_view);
    }
};

template <typename Mesh=stk::mesh::NgpMesh>
struct NGPMeshTraits
{
    using TeamPolicy = Kokkos::TeamPolicy<typename Mesh::MeshExecSpace,
                                          stk::mesh::ScheduleType>;
    using ShmemType = typename Mesh::MeshExecSpace::scratch_memory_space;
    using MeshIndex = typename Mesh::MeshIndex;
};

}
}

#endif /* NGPTYPES_H */
