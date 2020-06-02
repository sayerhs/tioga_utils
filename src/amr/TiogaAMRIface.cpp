#include "TiogaAMRIface.h"
#include "Timer.h"
#include "amrex_yaml.h"

#include "AMReX_ParmParse.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_PlotFileUtil.H"

#include "tioga.h"

namespace tioga_amr {
namespace {

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
        repo.declare_field("qcell_ref", m_ncell_vars, m_num_ghost);
        amrex::Print() << "Number of cell variables: " << m_ncell_vars << std::endl;
    }
    if (m_nnode_vars > 0) {
        m_qnode = &repo.declare_field("qnode", m_nnode_vars, m_num_ghost);
        repo.declare_field("qnode_ref", m_nnode_vars, m_num_ghost);
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

    if (verbose) {
        output_grid_summary(ngrids_global, ngrids_local, gint_data, greal_data);
    }
}

void TiogaAMRIface::register_solution(TIOGA::tioga& tg)
{
    if (num_total_vars() < 1) return;

    init_var(*m_qcell, m_ncell_vars, 0.5);
    init_var(*m_qnode, m_nnode_vars, 0.0);
    auto tmon = tioga_nalu::get_timer("TiogaAMRIface::register_solution");
    const int nlevels = m_mesh->repo().num_active_levels();
    int ilp = 0;

    for (int lev=0; lev < nlevels; ++lev) {
        if (m_ncell_vars > 0) {
            auto& qref = m_mesh->repo().get_field("qcell_ref");
            const bool isnodal = false;
            auto& qfab = (*m_qcell)(lev);
            for (amrex::MFIter mfi(qfab); mfi.isValid(); ++mfi) {
                auto& qarr = qfab[mfi];
                tg.register_amr_solution(ilp++, qarr.dataPtr(), isnodal);
            }
            auto& qref_fab = qref(lev);
            amrex::MultiFab::Copy(qref_fab, qfab, 0, 0,
                                  qref.num_comp(), qref.num_grow());
        }
        if (m_nnode_vars > 0) {
            const bool isnodal = true;
            auto& qfab = (*m_qnode)(lev);
            for (amrex::MFIter mfi(qfab); mfi.isValid(); ++mfi) {
                auto& qarr = qfab[mfi];
                tg.register_amr_solution(ilp++, qarr.dataPtr(), isnodal);
            }
            auto& qref = m_mesh->repo().get_field("qnode_ref");
            auto& qref_fab = qref(lev);
            amrex::MultiFab::Copy(qref_fab, qfab, 0, 0,
                                  qref.num_comp(), qref.num_grow());
        }
    }
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

                amrex::ParallelFor(bx, [&](int i, int j, int k) {
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

                amrex::ParallelFor(bx, [&](int i, int j, int k) {
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
    const int num_out_vars = num_total_vars() + 1;
    auto& repo = m_mesh->repo();
    auto& qout = m_mesh->repo().declare_field("qout", num_out_vars, 0);
    auto& ibcell = repo.get_int_field("iblank_cell");
    amrex::Vector<std::string> vnames{"iblank_cell"};
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

        int icomp = 1;
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

            amrex::ParallelFor(bx, [&](int i, int j, int k) noexcept {
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
