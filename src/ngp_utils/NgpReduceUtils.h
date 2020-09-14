#ifndef NGPREDUCEUTILS_H
#define NGPREDUCEUTILS_H

#include "ngp_utils/NgpTypes.h"

namespace tioga_nalu {
namespace ngp {

/** A custom Kokkos reduction operator for array types
 *
 *  Useful when you want to accumulate multiple quantities, e.g., computing an
 *  area-weighted average.
 */
template <typename ScalarType, int N>
struct NgpReduceArray
{
    ScalarType array_[N];

    KOKKOS_INLINE_FUNCTION
    NgpReduceArray() {}

    KOKKOS_INLINE_FUNCTION
    NgpReduceArray(ScalarType val)
    {
        for (int i = 0; i < N; ++i) array_[i] = val;
    }

    KOKKOS_INLINE_FUNCTION
    NgpReduceArray(const NgpReduceArray& rhs)
    {
        for (int i = 0; i < N; ++i) array_[i] = rhs.array_[i];
    }

    // See discussion in https://github.com/trilinos/Trilinos/issues/6125 for
    // details on the overloads.

    KOKKOS_INLINE_FUNCTION
    NgpReduceArray& operator=(const NgpReduceArray& rhs)
    {
        for (int i = 0; i < N; ++i) array_[i] = rhs.array_[i];
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    NgpReduceArray& operator=(const volatile NgpReduceArray& rhs)
    {
        for (int i = 0; i < N; ++i) array_[i] = rhs.array_[i];
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    volatile NgpReduceArray& operator=(const NgpReduceArray& rhs) volatile
    {
        for (int i = 0; i < N; ++i) array_[i] = rhs.array_[i];
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    volatile NgpReduceArray&
    operator=(const volatile NgpReduceArray& rhs) volatile
    {
        for (int i = 0; i < N; ++i) array_[i] = rhs.array_[i];
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    void operator+=(const NgpReduceArray& rhs)
    {
        for (int i = 0; i < N; ++i) array_[i] += rhs.array_[i];
    }

    KOKKOS_INLINE_FUNCTION
    void operator+=(const volatile NgpReduceArray& rhs) volatile
    {
        for (int i = 0; i < N; ++i) array_[i] += rhs.array_[i];
    }

    KOKKOS_INLINE_FUNCTION
    void operator*=(const NgpReduceArray& rhs)
    {
        for (int i = 0; i < N; ++i) array_[i] *= rhs.array_[i];
    }

    KOKKOS_INLINE_FUNCTION
    void operator*=(const volatile NgpReduceArray& rhs) volatile
    {
        for (int i = 0; i < N; ++i) array_[i] *= rhs.array_[i];
    }
};

using ArrayInt2 = NgpReduceArray<int, 2>;
using ArrayInt3 = NgpReduceArray<int, 3>;

using ArrayDbl2 = NgpReduceArray<double, 2>;
using ArrayDbl3 = NgpReduceArray<double, 3>;
} // namespace ngp
} // namespace tioga_nalu

namespace Kokkos {

template<>
struct reduction_identity<tioga_nalu::ngp::ArrayInt2>
{
    KOKKOS_FORCEINLINE_FUNCTION
    static tioga_nalu::ngp::ArrayInt2 sum()
    { return tioga_nalu::ngp::ArrayInt2(0); }

    KOKKOS_FORCEINLINE_FUNCTION
    static tioga_nalu::ngp::ArrayInt2 prod()
    { return tioga_nalu::ngp::ArrayInt2(1); }
};

template<>
struct reduction_identity<tioga_nalu::ngp::ArrayInt3>
{
    KOKKOS_FORCEINLINE_FUNCTION
    static tioga_nalu::ngp::ArrayInt3 sum()
    { return tioga_nalu::ngp::ArrayInt3(0); }

    KOKKOS_FORCEINLINE_FUNCTION
    static tioga_nalu::ngp::ArrayInt3 prod()
    { return tioga_nalu::ngp::ArrayInt3(1); }
};

template<>
struct reduction_identity<tioga_nalu::ngp::ArrayDbl2>
{
    KOKKOS_FORCEINLINE_FUNCTION
    static tioga_nalu::ngp::ArrayDbl2 sum()
    { return tioga_nalu::ngp::ArrayDbl2(0.0); }

    KOKKOS_FORCEINLINE_FUNCTION
    static tioga_nalu::ngp::ArrayDbl2 prod()
    { return tioga_nalu::ngp::ArrayDbl2(1.0); }
};

template<>
struct reduction_identity<tioga_nalu::ngp::ArrayDbl3>
{
    KOKKOS_FORCEINLINE_FUNCTION
    static tioga_nalu::ngp::ArrayDbl3 sum()
    { return tioga_nalu::ngp::ArrayDbl3(0.0); }

    KOKKOS_FORCEINLINE_FUNCTION
    static tioga_nalu::ngp::ArrayDbl3 prod()
    { return tioga_nalu::ngp::ArrayDbl3(1.0); }
};

}

#endif /* NGPREDUCEUTILS_H */
