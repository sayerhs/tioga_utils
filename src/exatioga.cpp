#include "ExaTioga.h"
#include "TiogaRef.h"
#include "amrex_yaml.h"

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
}

}
