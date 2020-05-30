#include "TiogaAMRIface.h"
#include "Timer.h"
#include "amrex_yaml.h"

#include "AMReX_ParmParse.H"

#include "tioga.h"

namespace tioga_amr {

TiogaAMRIface::TiogaAMRIface()
{}

TiogaAMRIface::~TiogaAMRIface() = default;

void TiogaAMRIface::load(const YAML::Node& node)
{
    auto tmon = tioga_nalu::get_timer("TiogaAMRIface::load");
    populate_parameters(node, "amr");
    populate_parameters(node, "geometry");

    if (node["field"]) {
        const auto& fnode = node["field"];
        get_optional(fnode, "num_ghost", m_num_ghost);
        get_optional(fnode, "num_cell_vars", m_ncell_vars);
        get_optional(fnode, "num_node_vars", m_nnode_vars);
    }

    m_mesh.reset(new StructMesh());
    m_mesh->load(node);
}

void TiogaAMRIface::initialize()
{
    auto tmon = tioga_nalu::get_timer("TiogaAMRIface::initialize");
    amrex::Print() << "Initializing AMReX mesh" << std::endl;
    m_mesh->initialize_mesh();

    auto& repo = m_mesh->repo();
    repo.declare_int_field("iblank_cell", 1, m_num_ghost);
    repo.declare_int_field("iblank", 1, m_num_ghost, FieldLoc::NODE);

    if (m_ncell_vars > 0) {
        m_qcell = &repo.declare_field("qcell", m_ncell_vars, m_num_ghost);
    }
    if (m_nnode_vars > 0) {
        m_qnode = &repo.declare_field("qnode", m_nnode_vars, m_num_ghost);
    }
}

void TiogaAMRIface::register_mesh(TIOGA::tioga& tg)
{
    auto tmon = tioga_nalu::get_timer("TiogaAMRIface::register_mesh");
    auto& mesh = *m_mesh;
    const int nlevels = mesh.finestLevel() + 1;
    const int iproc = amrex::ParallelDescriptor::MyProc();
    const int nproc = amrex::ParallelDescriptor::NProcs();
    const auto* problo = mesh.Geom(0).ProbLo();

    int ngrids_global = 0;
    int ngrids_local = 0;
    for (int lev=0; lev < nlevels; ++lev) {
        ngrids_global += mesh.boxArray(lev).size();

        const auto& dmap = mesh.DistributionMap(lev);
        for (long d=0; d < dmap.size(); ++d) {
            if (dmap[d] == iproc) ++ngrids_local;
        }
    }

    std::vector<int> gint_data(ints_per_grid * ngrids_global);
    std::vector<double> greal_data(reals_per_grid * ngrids_global);
    std::vector<int> lgrid_id(nproc, 0);
    std::vector<std::vector<int>> gid_map(nlevels);

    int igp = 0; // Global index of the grid
    int iix = 0; // Index into the integer array
    int irx = 0; // Index into the real array

    for (int lev=0; lev < nlevels; ++lev) {
        const auto& ba = mesh.boxArray(lev);
        const auto& dm = mesh.DistributionMap(lev);
        const amrex::Real* dx = mesh.Geom(lev).CellSize();

        for (long d = 0; d < dm.size(); ++d) {
            gint_data[iix++] = igp;             // Global index of this patch
            gint_data[iix++] = lev;             // Level of this patch
            gint_data[iix++] = dm[d];           // MPI rank of this patch
            gint_data[iix++] = lgrid_id[dm[d]]; // Local ID for this patch

            const auto& bx = ba[d];
            const int* lo = bx.loVect();
            const int* hi = bx.hiVect();

            for (int i = 0; i < AMREX_SPACEDIM; ++i) {
                gint_data[iix + i] = lo[i];
                gint_data[iix + AMREX_SPACEDIM + i] = hi[i];

                greal_data[irx + i] = problo[i] + lo[i] * dx[i];
                greal_data[irx + AMREX_SPACEDIM + i] = dx[i];
            }
            iix += 2 * AMREX_SPACEDIM;
            irx += 2 * AMREX_SPACEDIM;

            if (iproc == dm[d]) {
                gid_map[lev].push_back(igp);
            }

            // Increment global ID counter
            ++igp;
            // Increment local index
            ++lgrid_id[dm[d]];
        }
    }

    tg.register_amr_global_data(
        m_num_ghost, gint_data.data(), greal_data.data(), ngrids_global);
    tg.set_amr_patch_count(ngrids_local);

    // Register local patches
    int ilp = 0;
    auto& ibcell = mesh.repo().get_int_field("iblank_cell");
    for (int lev=0; lev < nlevels; ++lev) {
        auto& idmap = gid_map[lev];
        auto& ibfab = ibcell(lev);

        // Reset iblanks to 1 before registering with TIOGA
        ibfab.setVal(1);
        int ii = 0;
        for (amrex::MFIter mfi(ibfab); mfi.isValid(); ++mfi) {
            auto& ib = ibfab[mfi];
            tg.register_amr_local_data(ilp++, idmap[ii++], ib.dataPtr());
        }
    }
}

} // namespace tioga_amr
