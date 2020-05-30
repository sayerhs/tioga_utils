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

    void execute();

    void perform_connectivity();

private:
    stk::ParallelMachine m_comm;

    tioga_nalu::StkIface m_stk;
    TiogaAMRIface m_amr;

    TIOGA::tioga& m_tioga;
};

}

#endif /* EXATIOGA_H */
