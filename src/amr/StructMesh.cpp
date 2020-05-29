#include "StructMesh.h"

namespace tioga_amr {

StructMesh::StructMesh()
    : m_repo(*this)
{}

void StructMesh::initialize_mesh(const amrex::Real current_time)
{
    InitFromScratch(current_time);

    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "AMReX grid summary: " << std::endl
                       << "  Problem domain: " << Geom(0).ProbDomain() << std::endl;
        printGridSummary(amrex::OutStream(), 0, finestLevel());
    }
}

void StructMesh::MakeNewLevelFromScratch(
    int lev,
    amrex::Real ,
    const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm)
{
    SetBoxArray(lev, ba);
    SetDistributionMap(lev, dm);
}

void StructMesh::MakeNewLevelFromCoarse(
    int lev,
    amrex::Real ,
    const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm)
{
    SetBoxArray(lev, ba);
    SetDistributionMap(lev, dm);
}

void StructMesh::RemakeLevel(
    int lev,
    amrex::Real ,
    const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm)
{
    SetBoxArray(lev, ba);
    SetDistributionMap(lev, dm);
}

void StructMesh::ClearLevel(int lev)
{}

void StructMesh::ErrorEst(
    int /* lev */, amrex::TagBoxArray& /* tags */, amrex::Real /* time */, int /* ngrow */)
{
    amrex::Abort("Not implemented");
}

}
