#include "CartBoxRefinement.h"
#include "amrex_yaml.h"

namespace tioga_amr {
namespace {

amrex::BoxArray realbox_to_boxarray(
    const amrex::Vector<amrex::RealBox>& rbx, const amrex::Geometry& geom)
{
    amrex::BoxList bx_list;
    const auto* problo = geom.ProbLo();
    const auto* probhi = geom.ProbHi();
    const auto* dx = geom.CellSize();

    for (const auto& rb : rbx) {
        amrex::IntVect lo, hi;

        for (int i = 0; i < AMREX_SPACEDIM; ++i) {
            amrex::Real bbox_min = amrex::max(rb.lo()[i], problo[i]);
            amrex::Real bbox_max = amrex::min(rb.hi()[i], probhi[i]);
            amrex::Real rlo =
                amrex::Math::floor((bbox_min - problo[i]) / dx[i]);
            amrex::Real rhi = amrex::Math::ceil((bbox_max - problo[i]) / dx[i]);
            lo[i] = static_cast<int>(rlo);
            hi[i] = static_cast<int>(rhi);
        }
        bx_list.push_back({lo, hi});
    }

    return amrex::BoxArray(std::move(bx_list));
}
} // namespace

void CartBoxRefinement::initialize(
    const amrex::AmrCore& mesh, const YAML::Node& node)
{
    const auto& geom = mesh.Geom();
    const int max_lev = geom.size();

    const int nlev_in = node["num_levels"].as<int>();

    // Issue a warning if the max levels in the input file is less than what's
    // requested in the refinement file.
    if (max_lev < nlev_in)
        amrex::Print() << "WARNING: AmrMesh::finestLevel() is less than the "
                          "requested levels in static refinement file"
                       << std::endl;

    // Set the number of levels to the minimum of what is in the input file and
    // the simulation
    m_nlevels = amrex::min(max_lev, nlev_in);

    if (m_nlevels < 1) return;

    for (int lev = 0; lev < m_nlevels; ++lev) {
        const std::string key = "level_" + std::to_string(lev);
        const auto& info = node[key];

        if (!info.IsSequence()) {
            throw std::runtime_error(
                "Error parsing refinement criteria for level: " +
                std::to_string(lev));
        }

        amrex::Vector<amrex::RealBox> rbx_list;
        for (int ib = 0; ib < info.size(); ++ib) {
            const auto& binfo = info[ib];
            const auto lovec = binfo["lo"].as<amrex::Vector<amrex::Real>>();
            const auto hivec = binfo["hi"].as<amrex::Vector<amrex::Real>>();
            AMREX_ALWAYS_ASSERT(hivec.size() == 3);
            AMREX_ALWAYS_ASSERT(hivec.size() == 3);
            rbx_list.emplace_back(lovec[0], lovec[1], lovec[2], hivec[0], hivec[1], hivec[2]);
        }

        auto ba = realbox_to_boxarray(rbx_list, geom[lev]);
        m_real_boxes.push_back(std::move(rbx_list));
        m_boxarrays.push_back(std::move(ba));
    }
}

void CartBoxRefinement::operator()(
    int level, amrex::TagBoxArray& tags, amrex::Real, int)
{
    if (level < m_nlevels) tags.setVal(m_boxarrays[level], amrex::TagBox::SET);
}

} // namespace tioga_amr
