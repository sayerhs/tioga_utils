#include "TiogaAMRIface.h"
#include "Timer.h"
#include "amrex_yaml.h"

#include "AMReX_ParmParse.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_PlotFileUtil.H"

#include "TiogaMeshInfo.h"
#include "tioga.h"

namespace tioga_amr {
namespace {

template <typename T1, typename T2>
void amr_to_tioga(T1& lhs, T2& rhs)
{
    lhs.sz = rhs.size();
    lhs.hptr = rhs.h_view.data();
    lhs.dptr = rhs.d_view.data();
}

void output_grid_summary(
    int nglobal,
    int nlocal,
    std::vector<int>& int_data,
    std::vector<double>& real_data)
{
    amrex::Print() << "AMR grid summary: "
                   << "Num. grids = " << nglobal << std::endl;
    for (int i = 0; i < nglobal; ++i) {
        const int ii = i * 10;
        const int ir = i * 6;

        amrex::Print() << "AMR grid: " << i << std::endl;
        for (int j = 0; j < 4; ++j)
            amrex::Print() << std::setw(4) << int_data[ii + j] << " ";
        amrex::Print() << std::endl;
        for (int j = 4; j < 10; ++j)
            amrex::Print() << std::setw(4) << int_data[ii + j] << " ";
        amrex::Print() << std::endl;

        for (int j = 0; j < AMREX_SPACEDIM; ++j)
            amrex::Print() << std::setw(12) << real_data[ir + j] << " ";
        amrex::Print() << std::endl;
        for (int j = 0; j < AMREX_SPACEDIM; ++j)
            amrex::Print() << std::setw(12)
                           << real_data[ir + AMREX_SPACEDIM + j] << " ";
        amrex::Print() << std::endl;
    }
}
} // namespace

NgpAMRInfo::NgpAMRInfo(const int nglobal, const int nlocal)
    : level(nglobal)
    , mpi_rank(nglobal)
    , local_id(nglobal)
    , ilow(nglobal)
    , ihigh(nglobal)
    , dims(nglobal)
    , xlo(AMREX_SPACEDIM * nglobal)
    , dx(AMREX_SPACEDIM * nglobal)
    , global_idmap(nlocal)
    , iblank_node(nlocal)
    , iblank_cell(nlocal)
    , qcell(nlocal)
    , qnode(nlocal)
{}


TiogaAMRIface::TiogaAMRIface()
    : m_info(new TIOGA::AMRMeshInfo)
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
        repo.declare_field("qcell_ref", m_ncell_vars, m_num_ghost);
        amrex::Print() << "Number of cell variables: " << m_ncell_vars << std::endl;
    }
    if (m_nnode_vars > 0) {
        m_qnode = &repo.declare_field("qnode", m_nnode_vars, m_num_ghost, FieldLoc::NODE);
        repo.declare_field("qnode_ref", m_nnode_vars, m_num_ghost, FieldLoc::NODE);
        amrex::Print() << "Number of nodal variables: " << m_nnode_vars << std::endl;
    }
}

void TiogaAMRIface::register_mesh(TIOGA::tioga& tg, const bool verbose)
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

    m_info->ngrids_global = ngrids_global;
    m_info->ngrids_local = ngrids_local;
    m_amr_data.reset(new NgpAMRInfo(ngrids_global, ngrids_local));
    std::vector<int> lgrid_id(nproc, 0);

    int igp = 0; // Global index of the grid
    int ilp = 0; // Counter for local patches
    int iix = 0; // Index into the integer array

    for (int lev = 0; lev < nlevels; ++lev) {
        const auto& ba = mesh.boxArray(lev);
        const auto& dm = mesh.DistributionMap(lev);
        const amrex::Real* dx = mesh.Geom(lev).CellSize();

        auto& ad = *m_amr_data;
        for (long d=0; d < dm.size(); ++d) {
            ad.level.h_view[iix] = lev;           // AMR Level of this patch
            ad.mpi_rank.h_view[iix] = dm[d];      // MPI rank of this patch
            ad.local_id.h_view[iix] = lgrid_id[dm[d]]; // Local ID for this patch

            const auto& bx = ba[d];
            const int* lo = bx.loVect();
            const int* hi = bx.hiVect();

            const int ioff = AMREX_SPACEDIM * iix;
            for (int i=0; i < AMREX_SPACEDIM; ++i) {
                ad.ilow.h_view[ioff + i] = lo[i];
                ad.ihigh.h_view[ioff + i] =  hi[i];
                ad.dims.h_view[ioff + i] = (hi[i] - lo[i]) + 1;

                ad.xlo.h_view[ioff + i] = problo[i] + lo[i] * dx[i];
                ad.dx.h_view[ioff + i] = dx[i];
            }

            if (iproc == dm[d]) {
                ad.global_idmap.h_view[ilp++] = igp;
            }

            // Increment array counter
            ++iix;
            // Increment global ID counter
            ++igp;
            // Increment local index
            ++lgrid_id[dm[d]];
        }
    }
    m_amr_data->level.sync_to_device();
    m_amr_data->mpi_rank.sync_to_device();
    m_amr_data->local_id.sync_to_device();
    m_amr_data->ilow.sync_to_device();
    m_amr_data->ihigh.sync_to_device();
    m_amr_data->dims.sync_to_device();
    m_amr_data->xlo.sync_to_device();
    m_amr_data->dx.sync_to_device();
    m_amr_data->global_idmap.sync_to_device();

    // Reset local patch counter
    ilp = 0;
    auto& ibcell = mesh.repo().get_int_field("iblank_cell");
    auto& ibnode = mesh.repo().get_int_field("iblank");
    for (int lev=0; lev < nlevels; ++lev) {
        auto& ad = *m_amr_data;
        auto& ibfab = ibcell(lev);
        auto& ibnodefab = ibnode(lev);

        // Reset iblanks to 1 before registering with TIOGA
        ibfab.setVal(1);
        ibnodefab.setVal(1);
        for (amrex::MFIter mfi(ibfab); mfi.isValid(); ++mfi) {
            auto& ib = ibfab[mfi];
            auto& ibn = ibnodefab[mfi];
            ad.iblank_cell.h_view[ilp] = ib.dataPtr();
            ad.iblank_node.h_view[ilp] = ibn.dataPtr();
            ad.iblank_cell.d_view[ilp] = ib.dataPtr();
            ad.iblank_node.d_view[ilp] = ibn.dataPtr();

            if (m_ncell_vars > 0) {
                ad.qcell.h_view[ilp] = (*m_qcell)(lev)[mfi].dataPtr();
                ad.qcell.d_view[ilp] = (*m_qcell)(lev)[mfi].dataPtr();
            } else {
                ad.qcell.h_view[ilp] = nullptr;
                ad.qcell.d_view[ilp] = nullptr;
            }
            if (m_nnode_vars > 0) {
                ad.qnode.h_view[ilp] = (*m_qnode)(lev)[mfi].dataPtr();
                ad.qnode.d_view[ilp] = (*m_qnode)(lev)[mfi].dataPtr();
            } else {
                ad.qnode.h_view[ilp] = nullptr;
                ad.qnode.d_view[ilp] = nullptr;
            }

            ++ilp;
        }
    }

    amr_to_tioga_info();
    tg.register_amr_grid(m_info.get());
}

void TiogaAMRIface::amr_to_tioga_info()
{
    auto& ad = *m_amr_data;
    auto& mi = *m_info;

    mi.num_ghost = m_num_ghost;
    mi.nvar_cell = m_ncell_vars;
    mi.nvar_node = m_nnode_vars;

    amr_to_tioga(mi.level, ad.level);
    amr_to_tioga(mi.mpi_rank, ad.mpi_rank);
    amr_to_tioga(mi.local_id, ad.local_id);
    amr_to_tioga(mi.ilow, ad.ilow);
    amr_to_tioga(mi.ihigh, ad.ihigh);
    amr_to_tioga(mi.dims, ad.dims);
    amr_to_tioga(mi.xlo, ad.xlo);
    amr_to_tioga(mi.dx, ad.dx);
    amr_to_tioga(mi.global_idmap, ad.global_idmap);
    amr_to_tioga(mi.iblank_node, ad.iblank_node);
    amr_to_tioga(mi.iblank_cell, ad.iblank_cell);
    amr_to_tioga(mi.qcell, ad.qcell);
    amr_to_tioga(mi.qnode, ad.qnode);
}

void TiogaAMRIface::register_solution(TIOGA::tioga& tg)
{
    if (num_total_vars() < 1) return;

    init_var(*m_qcell, m_ncell_vars, 0.5);
    init_var(*m_qnode, m_nnode_vars, 0.0);
    auto tmon = tioga_nalu::get_timer("TiogaAMRIface::register_solution");

    tg.register_amr_solution();

#if 0
    const int nlevels = m_mesh->repo().num_active_levels();
    int ipatch_cell = 0;
    int ipatch_node = 0;

    for (int lev=0; lev < nlevels; ++lev) {
        if (m_ncell_vars > 0) {
            auto& qref = m_mesh->repo().get_field("qcell_ref");
            auto& qfab = (*m_qcell)(lev);
            for (amrex::MFIter mfi(qfab); mfi.isValid(); ++mfi) {
                auto& qarr = qfab[mfi];
                tg.register_amr_solution(ipatch_cell++, qarr.dataPtr(), m_ncell_vars, 0);
            }
            auto& qref_fab = qref(lev);
            amrex::MultiFab::Copy(qref_fab, qfab, 0, 0,
                                  qref.num_comp(), qref.num_grow());
        }
        if (m_nnode_vars > 0) {
            auto& qfab = (*m_qnode)(lev);
            for (amrex::MFIter mfi(qfab); mfi.isValid(); ++mfi) {
                auto& qarr = qfab[mfi];
                tg.register_amr_solution(ipatch_node++, qarr.dataPtr(), 0, m_nnode_vars);
            }
            auto& qref = m_mesh->repo().get_field("qnode_ref");
            auto& qref_fab = qref(lev);
            amrex::MultiFab::Copy(qref_fab, qfab, 0, 0,
                                  qref.num_comp(), qref.num_grow());
        }
    }
#endif
}

void TiogaAMRIface::update_solution()
{
    if (num_total_vars() < 1) return;
    auto tmon = tioga_nalu::get_timer("TiogaAMRIface::update_solution");
    const int nlevels = m_mesh->repo().num_active_levels();

    amrex::Real rnorm = 0.0;
    int counter = 0;
    for (int lev=0; lev < nlevels; ++lev) {
        if (m_ncell_vars > 0) {
            const int ncomp = m_ncell_vars;
            auto& qref = m_mesh->repo().get_field("qcell_ref");
            auto& qfab = (*m_qcell)(lev);
            auto& qref_fab = qref(lev);
            for (amrex::MFIter mfi(qfab); mfi.isValid(); ++mfi) {
                auto bx = mfi.tilebox();
                counter += bx.numPts() * ncomp;
                const auto qarr = qfab.array(mfi);
                const auto qref_arr = qref_fab.array(mfi);

                amrex::LoopOnCpu(bx, [=, &rnorm](int i, int j, int k) {
                    for (int n = 0; n < ncomp; ++n) {
                        amrex::Real diff =
                            qarr(i, j, k, n) - qref_arr(i, j, k, n);
                        rnorm += diff * diff;
                    }
                });
            }
        }
        if (m_nnode_vars > 0) {
            const int ncomp = m_nnode_vars;
            auto& qref = m_mesh->repo().get_field("qnode_ref");
            auto& qfab = (*m_qnode)(lev);
            auto& qref_fab = qref(lev);
            for (amrex::MFIter mfi(qfab); mfi.isValid(); ++mfi) {
                auto bx = mfi.tilebox();
                counter += bx.numPts() * ncomp;
                const auto qarr = qfab.array(mfi);
                const auto qref_arr = qref_fab.array(mfi);

                amrex::LoopOnCpu(bx, [=, &rnorm](int i, int j, int k) {
                    for (int n = 0; n < ncomp; ++n) {
                        amrex::Real diff =
                            qarr(i, j, k, n) - qref_arr(i, j, k, n);
                        rnorm += diff * diff;
                    }
                });
            }
        }
    }

    rnorm /= static_cast<amrex::Real>(counter);
    rnorm = std::sqrt(rnorm);
    amrex::ParallelDescriptor::ReduceRealMax(
        rnorm, amrex::ParallelDescriptor::IOProcessorNumber());
    amrex::Print() << "TIOGA interpolation error (max L2 norm) for AMR mesh: "
                   << rnorm << std::endl;
}

void TiogaAMRIface::write_outputs(const int time_index, const double time)
{
    auto tmon = tioga_nalu::get_timer("TiogaAMRIface::write_outputs");

    // Total variables = cell + node + iblank_cell + iblank_node
    const int num_out_vars = num_total_vars() + 2;
    auto& repo = m_mesh->repo();
    auto& qout = m_mesh->repo().declare_field("qout", num_out_vars, 0);
    auto& ibcell = repo.get_int_field("iblank_cell");
    auto& ibnode = repo.get_int_field("iblank");
    amrex::Vector<std::string> vnames{"iblank_cell", "iblank"};
    for (int n=0; n < m_ncell_vars; ++n) {
        vnames.push_back("qcell" + std::to_string(n));
    }
    for (int n=0; n < m_nnode_vars; ++n) {
        vnames.push_back("qnode" + std::to_string(n));
    }

    const int nlevels = m_mesh->finestLevel() + 1;
    for (int lev=0; lev < nlevels; ++lev) {
        auto& qfab = qout(lev);
        {
            auto& ibc = ibcell(lev);
            amrex::MultiFab::Copy(qfab, amrex::ToMultiFab(ibc), 0, 0, 1, 0);
        }
        {
            auto& ibn = ibnode(lev);
            amrex::MultiFab::Copy(qfab, amrex::ToMultiFab(ibn), 0, 1, 1, 0);
        }

        int icomp = 2;
        if (m_ncell_vars > 0) {
            auto& qvars = repo.get_field("qcell")(lev);
            amrex::MultiFab::Copy(qfab, qvars, 0, icomp, m_ncell_vars, 0);
            icomp += m_ncell_vars;
        }

        if (m_nnode_vars > 0) {
            auto& qvars = repo.get_field("qnode")(lev);
            amrex::average_node_to_cellcenter(qfab, icomp, qvars, 0, m_nnode_vars, 0);
        }
    }

    amrex::Vector<int> istep(m_mesh->finestLevel() + 1, time_index);
    const std::string& plt_filename = amrex::Concatenate("plt", time_index);
    amrex::Print() << "Writing plot file: " << plt_filename << std::endl;
    amrex::WriteMultiLevelPlotfile(
        plt_filename, nlevels, qout.vec_const_ptrs(), vnames,
        m_mesh->Geom(), time, istep, m_mesh->refRatio());
}

void TiogaAMRIface::init_var(Field& qcell, const int nvars, const amrex::Real offset)
{
    if (nvars < 1) return;
    auto tmon = tioga_nalu::get_timer("TiogaAMRIface::init_var");

    auto& repo = m_mesh->repo();
    const int nlevels = repo.num_active_levels();

    for (int lev = 0; lev < nlevels; ++lev) {
        const auto& geom = m_mesh->Geom(lev);
        const auto* problo = geom.ProbLo();
        const auto* dx = geom.CellSize();
        auto& qfab = qcell(lev);

        for (amrex::MFIter mfi(qfab); mfi.isValid(); ++mfi) {
            const auto bx = mfi.growntilebox(m_num_ghost);
            const auto qarr = qfab.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const amrex::Real x = problo[0] + (i + offset) * dx[0];
                const amrex::Real y = problo[1] + (j + offset) * dx[1];
                const amrex::Real z = problo[2] + (k + offset) * dx[2];

                for (int n = 0; n < nvars; ++n) {
                    const amrex::Real xfac = 1.0 * ((n + 1) << n);
                    const amrex::Real yfac = 1.0 * ((n + 2) << n);
                    const amrex::Real zfac = 1.0 * ((n + 3) << n);
                    qarr(i, j, k, n) = xfac * x + yfac * y +  zfac * z;
                }
            });
        }
    }
}

} // namespace tioga_amr
