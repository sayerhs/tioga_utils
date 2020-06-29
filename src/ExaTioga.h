#ifndef EXATIOGA_H
#define EXATIOGA_H

#include "TiogaAMRIface.h"
#include "StkIface.h"

namespace YAML {
class Node;
}

namespace TIOGA {
class tioga;
}

namespace tioga_amr {

class ExaTioga
{
public:
    ExaTioga(stk::ParallelMachine& comm);

    void init_amr(const YAML::Node&);

    void init_stk(const YAML::Node&);

    void execute(const YAML::Node&);

    void perform_connectivity();

    void exchange_solution();

    void check_solution_norm();

private:
    void run_timesteps(const bool do_write, size_t ofileID);

    stk::ParallelMachine m_comm;

    tioga_nalu::StkIface m_stk;
    TiogaAMRIface m_amr;

    TIOGA::tioga& m_tioga;
};

}

#endif /* EXATIOGA_H */
