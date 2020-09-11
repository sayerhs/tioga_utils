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
        get_optional(fnode, "stk_sol", m_stk_sol);
        get_optional(fnode, "amr_sol", m_amr_sol);
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

    m_ints.resize(ints_per_grid * ngrids_global);
    m_reals.resize(reals_per_grid * ngrids_global);
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
            m_ints[iix++] = igp;             // Global index of this patch
            m_ints[iix++] = lev;             // Level of this patch
            m_ints[iix++] = dm[d];           // MPI rank of this patch
            m_ints[iix++] = lgrid_id[dm[d]]; // Local ID for this patch

            const auto& bx = ba[d];
            const int* lo = bx.loVect();
            const int* hi = bx.hiVect();

            for (int i = 0; i < AMREX_SPACEDIM; ++i) {
                m_ints[iix + i] = lo[i];
                m_ints[iix + AMREX_SPACEDIM + i] = hi[i];

                m_reals[irx + i] = problo[i] + lo[i] * dx[i];
                m_reals[irx + AMREX_SPACEDIM + i] = dx[i];
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
        m_num_ghost, m_ints.data(), m_reals.data(), ngrids_global);
    tg.set_amr_patch_count(ngrids_local);

    // Register local patches
    int ilp = 0;
    auto& ibcell = mesh.repo().get_int_field("iblank_cell");
    auto& ibnode = mesh.repo().get_int_field("iblank");
    for (int lev=0; lev < nlevels; ++lev) {
        auto& idmap = gid_map[lev];
        auto& ibfab = ibcell(lev);
        auto& ibnodefab = ibnode(lev);

        // Reset iblanks to 1 before registering with TIOGA
        ibfab.setVal(1);
        ibnodefab.setVal(1);
        int ii = 0;
        for (amrex::MFIter mfi(ibfab); mfi.isValid(); ++mfi) {
            auto& ib = ibfab[mfi];
            auto& ibn = ibnodefab[mfi];
            tg.register_amr_local_data(
                ilp++, idmap[ii++], ib.dataPtr(), ibn.dataPtr());
        }
    }

    if (verbose) {
        output_grid_summary(ngrids_global, ngrids_local, m_ints, m_reals);
    }
}

void TiogaAMRIface::register_solution(TIOGA::tioga& tg)
{
    if (num_total_vars() < 1) return;
    auto tmon = tioga_nalu::get_timer("TiogaAMRIface::register_solution");

    init_var(*m_qcell, m_ncell_vars, 0.5);
    init_var(*m_qnode, m_nnode_vars, 0.0);
    const int nlevels = m_mesh->repo().num_active_levels();
    int ipatch_cell = 0;
    int ipatch_node = 0;

    for (int lev=0; lev < nlevels; ++lev) {
        if (m_ncell_vars > 0) {
            auto& qfab = (*m_qcell)(lev);
            for (amrex::MFIter mfi(qfab); mfi.isValid(); ++mfi) {
                auto& qarr = qfab[mfi];
                tg.register_amr_solution(ipatch_cell++, qarr.dataPtr(), m_ncell_vars, 0);
            }
        }
        if (m_nnode_vars > 0) {
            auto& qfab = (*m_qnode)(lev);
            for (amrex::MFIter mfi(qfab); mfi.isValid(); ++mfi) {
                auto& qarr = qfab[mfi];
                tg.register_amr_solution(ipatch_node++, qarr.dataPtr(), 0, m_nnode_vars);
            }
        }
    }
}

void TiogaAMRIface::update_solution()
{
  update_solution(true);
  update_solution(false);
}

void TiogaAMRIface::update_solution(const bool isField)
{
    if (num_total_vars() < 1) return;
    auto tmon = tioga_nalu::get_timer("TiogaAMRIface::update_solution");
    const int nlevels = m_mesh->repo().num_active_levels();
    int sol = isField ? m_amr_sol : m_stk_sol;

    amrex::Real rnorm = 0.0;
    int counter = 0;
    for (int lev=0; lev < nlevels; ++lev) {
        const auto& geom = m_mesh->Geom(lev);
        const auto* problo = geom.ProbLo();
        const auto* dx = geom.CellSize();

        if (m_ncell_vars > 0) {
            const int ncomp = m_ncell_vars;
            auto& qref = m_mesh->repo().get_field("qcell_ref");
            auto& qref_fab = qref(lev);
            auto& qfab = (*m_qcell)(lev);
            auto& ibcell = m_mesh->repo().get_int_field("iblank_cell");
            auto& ibcell_fab = ibcell(lev);
            for (amrex::MFIter mfi(qfab); mfi.isValid(); ++mfi) {
                auto bx = mfi.tilebox();
                const auto qarr = qfab.array(mfi);
                const auto qref_arr = qref_fab.array(mfi);
                const auto ibcell_arr = ibcell_fab.array(mfi);

                amrex::ParallelFor(bx, [&](int i, int j, int k) {
                    if ((isField && (ibcell_arr(i,j,k,0) == 1)) ||
                        (!isField && (ibcell_arr(i,j,k,0) == -1))) {
                        counter += ncomp;

                        const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
                        const amrex::Real y = problo[1] + (j + 0.5) * dx[1];
                        const amrex::Real z = problo[2] + (k + 0.5) * dx[2];

                        for (int n = 0; n < ncomp; ++n) {
                            double qref = get_sol(x, y, z, n, sol);
                            amrex::Real diff = qarr(i, j, k, n) - qref;
                            rnorm += diff * diff;
                        }
                    }
                });
            }
        }
        if (m_nnode_vars > 0) {
            const int ncomp = m_nnode_vars;
            auto& qref = m_mesh->repo().get_field("qnode_ref");
            auto& qref_fab = qref(lev);
            auto& qfab = (*m_qnode)(lev);
            auto& ibnode = m_mesh->repo().get_int_field("iblank");
            auto& ibnode_fab = ibnode(lev);
            for (amrex::MFIter mfi(qfab); mfi.isValid(); ++mfi) {
                auto bx = mfi.tilebox();
                const auto qarr = qfab.array(mfi);
                const auto qref_arr = qref_fab.array(mfi);
                const auto ibnode_arr = ibnode_fab.array(mfi);

                amrex::ParallelFor(bx, [&](int i, int j, int k) {
                    if ((isField && (ibnode_arr(i,j,k,0) == 1)) ||
                        (!isField && (ibnode_arr(i,j,k,0) == -1))) {
                        counter += ncomp;

                        const amrex::Real x = problo[0] + i * dx[0];
                        const amrex::Real y = problo[1] + j * dx[1];
                        const amrex::Real z = problo[2] + k * dx[2];

                        for (int n = 0; n < ncomp; ++n) {
                            double qref = get_sol(x, y, z, n, sol);
                            amrex::Real diff = qarr(i, j, k, n) - qref;
                            rnorm += diff * diff;
                        }
                    }
                });
            }
        }
    }

    if(counter > 0) {
        rnorm /= static_cast<amrex::Real>(counter);
        rnorm = std::sqrt(rnorm);
    }
    amrex::ParallelDescriptor::ReduceRealMax(
        rnorm, amrex::ParallelDescriptor::IOProcessorNumber());
    if (isField) {
        amrex::Print() << "TIOGA interpolation error (max L2 norm) for AMR mesh on field points: "
                       << rnorm << std::endl;
    }
    else {
      amrex::Print() << "TIOGA interpolation error (max L2 norm) for AMR mesh on fringe points: "
                     << rnorm << std::endl;
    }
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

            amrex::ParallelFor(bx, [&](int i, int j, int k) noexcept {
                const amrex::Real x = problo[0] + (i + offset) * dx[0];
                const amrex::Real y = problo[1] + (j + offset) * dx[1];
                const amrex::Real z = problo[2] + (k + offset) * dx[2];

                for (int n = 0; n < nvars; ++n) {
                    qarr(i, j, k, n) = get_sol(x, y, z, n, m_amr_sol);
                }
            });
        }
    }
}

double TiogaAMRIface::get_sol(const double x, const double y, const double z,
    const int n, const int sol)
{
    double val = 0.0;

    switch (sol) {
        case 0: { // constant
            val = 1.0 * ((n + 1) << n);
            break;
        }
        case 1: { // linear
            double xfac = 1.0 * ((n + 1) << n);
            double yfac = 1.0 * ((n + 2) << n);
            double zfac = 1.0 * ((n + 3) << n);
            val = xfac * x + yfac * y +  zfac * z;
            break;
        }
        default :
            amrex::Print() << "Invalid solution request for AMR mesh: " << sol << std::endl;
    }

    return val;
}

} // namespace tioga_amr
