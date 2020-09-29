#ifndef NGPAMRTYPES_H
#define NGPAMRTYPES_H

#include "AMReX_Gpu.H"

namespace tioga_amr {

template <typename T>
struct NgpAmrDualArray
{
    using ArrayType = amrex::Gpu::DeviceVector<T>;
    using HostArrayType = amrex::Vector<T>;

    ArrayType d_view;
    HostArrayType h_view;

    NgpAmrDualArray() : d_view(), h_view() {}

    NgpAmrDualArray(const size_t len) : d_view(len), h_view(len) {}

    void resize(const size_t len)
    {
        d_view.resize(len);
        h_view.resize(len);
    }

    size_t size() const { return d_view.size(); }

    void sync_to_device()
    {
        amrex::Gpu::copy(
            amrex::Gpu::hostToDevice, h_view.begin(), h_view.end(),
            d_view.begin());
    }

    void sync_to_host()
    {
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost, d_view.begin(), d_view.end(),
            h_view.begin());
    }
};

}

#endif /* NGPAMRTYPES_H */
