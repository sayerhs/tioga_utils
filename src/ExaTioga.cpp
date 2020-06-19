#include <iostream>

#include "ExaTioga.h"
#include "TiogaRef.h"
#include "amrex_yaml.h"
#include "Timer.h"

#include "stk_util/environment/WallTime.hpp"
#include "stk_util/environment/perf_util.hpp"

#include "tioga.h"

namespace tioga_amr {

namespace {
void print_memory_diag(const stk::ParallelMachine& comm)
{
    const double factor = 1024.0;
    size_t curr_max, curr_min, curr_avg;
    stk::get_current_memory_usage_across_processors(
        comm, curr_max, curr_min, curr_avg);

    const int iproc = stk::parallel_machine_rank(comm);
    if (iproc == 0)
        std::cout << "Memory usage (KB): Avg. = " << (1.0 * curr_avg) / factor
                  << "; Min. = " << (1.0 * curr_min) / factor
                  << "; Max. = " << (1.0 * curr_max) / factor << std::endl;
}

} // namespace

ExaTioga::ExaTioga(stk::ParallelMachine& comm)
    : m_comm(comm)
    , m_stk(comm)
    , m_amr()
    , m_tioga(tioga_nalu::TiogaRef::self().get())
{
    const int iproc = stk::parallel_machine_rank(comm);
    const int nproc = stk::parallel_machine_size(comm);
    m_tioga.setCommunicator(comm, iproc, nproc);
}

void ExaTioga::init_amr(const YAML::Node& node)
{
    m_amr.load(node["amr_wind"]);
    m_amr.initialize();
}

void ExaTioga::init_stk(const YAML::Node& node)
{
    m_stk.num_cell_vars() = m_amr.num_cell_vars();
    m_stk.num_node_vars() = m_amr.num_node_vars();
    m_stk.load_and_initialize_all(node["nalu_wind"]);
}

void ExaTioga::execute(const YAML::Node& doc)
{
    perform_connectivity();
    exchange_solution();
    print_memory_diag(m_comm);

    bool do_write = true;
    get_optional(doc, "write_outputs", do_write);
    if (do_write) {
        m_amr.write_outputs(0, 0.0);
        m_stk.write_outputs(doc["nalu_wind"], 0.0);
    }
}

void ExaTioga::perform_connectivity()
{
    amrex::Print() << "Register STK mesh" << std::endl;
    m_stk.register_mesh();
    amrex::Print() << "Register AMR mesh" << std::endl;
    m_amr.register_mesh(m_tioga);
    stk::parallel_machine_barrier(m_comm);

    {
        auto tmon = tioga_nalu::get_timer("tioga::profile");
        amrex::Print() << "TIOGA profile" << std::endl;
        m_tioga.profile();
        stk::parallel_machine_barrier(m_comm);
    }
    {
        auto tmon = tioga_nalu::get_timer("tioga::performConnectivity");
        amrex::Print() << "TIOGA unstructured connectivity" << std::endl;
        m_tioga.performConnectivity();
        stk::parallel_machine_barrier(m_comm);
    }
    {
        auto tmon = tioga_nalu::get_timer("tioga::performConnectivityAMR");
        amrex::Print() << "TIOGA AMR connectivity" << std::endl;
        m_tioga.performConnectivityAMR();
        stk::parallel_machine_barrier(m_comm);
    }

    m_stk.post_connectivity_work();
    amrex::Print() << "Domain connectivity completed successfully" << std::endl;
}

void ExaTioga::exchange_solution()
{
    if (m_amr.num_total_vars() < 1) {
        amrex::Print() << "No solution variables available. Skipping solution update"
                       << std::endl;
    }
    amrex::Print() << "Registering solution variables" << std::endl;
    m_stk.register_solution();
    m_amr.register_solution(m_tioga);

    {
        auto tmon = tioga_nalu::get_timer("tioga::dataUpdate_AMR");
        m_tioga.dataUpdate_AMR();
        stk::parallel_machine_barrier(m_comm);
    }

    m_amr.update_solution();
    m_stk.update_solution();
}

}
