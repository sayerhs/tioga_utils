#include "ExaTioga.h"
#include "TiogaRef.h"
#include "amrex_yaml.h"
#include "Timer.h"

#include "tioga.h"

namespace tioga_amr {

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
    m_stk.load_and_initialize_all(node["nalu_wind"]);
}

void ExaTioga::execute()
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
    amrex::Print() << "Domain connectivity completed successfully" << std::endl;
}

}
