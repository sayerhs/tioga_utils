#include "StructMesh.h"
#include "CartBoxRefinement.h"
#include "amrex_yaml.h"

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
    amrex::Real time,
    const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm)
{
    SetBoxArray(lev, ba);
    SetDistributionMap(lev, dm);

    m_repo.make_new_level_from_scratch(lev, time, ba, dm);
}

void StructMesh::MakeNewLevelFromCoarse(
    int lev,
    amrex::Real time,
    const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm)
{
    SetBoxArray(lev, ba);
    SetDistributionMap(lev, dm);

    m_repo.make_new_level_from_coarse(lev, time, ba, dm);
}

void StructMesh::RemakeLevel(
    int lev,
    amrex::Real time,
    const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm)
{
    SetBoxArray(lev, ba);
    SetDistributionMap(lev, dm);

    m_repo.remake_level(lev, time, ba, dm);
}

void StructMesh::ClearLevel(int lev)
{
    m_repo.clear_level(lev);
}

void StructMesh::ErrorEst(
    int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow)
{
    for (auto& rc: m_refine_vec)
        (*rc)(lev, tags, time, ngrow);
}

void StructMesh::load(const YAML::Node& node)
{
    if (!node["refinement"]) return;

    const auto& rnode = node["refinement"];
    AMREX_ALWAYS_ASSERT(rnode.IsSequence());
    for (int i=0; i < rnode.size(); ++i) {
        const auto& ref = rnode[i];
        if (ref["type"]) {
            // Only static refinement supported now
            const auto& ref_type = ref["type"].as<std::string>();
            AMREX_ALWAYS_ASSERT(ref_type == "static");
        }
        std::unique_ptr<CartBoxRefinement> obj(new CartBoxRefinement);
        obj->initialize(*this, ref);
        m_refine_vec.push_back(std::move(obj));
    }
}

}
