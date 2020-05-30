#ifndef STKIFACE_H
#define STKIFACE_H

#include <memory>

#include "TiogaSTKIface.h"
#include "MeshMotion.h"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_io/StkMeshIoBroker.hpp"

namespace YAML {
class Node;
}

namespace tioga_nalu {

class StkIface
{
public:
    StkIface(stk::ParallelMachine& comm);

    void load(const YAML::Node&);

    void setup();

    void populate_bulk_data();

    void initialize();

    void load_and_initialize_all(const YAML::Node&);

    void write_outputs(const YAML::Node&, const double time = 0.0);

    void register_mesh()
    {
        tg_->register_mesh();
    }

    void post_connectivity_work()
    {
        tg_->post_connectivity_work();
    }

    std::string coordinates_name() const
    {
        return (has_motion_? "current_coordinates" : "coordinates");
    }

private:
    stk::ParallelMachine comm_;
    stk::mesh::MetaData meta_;
    stk::mesh::BulkData bulk_;
    stk::io::StkMeshIoBroker stkio_;

    std::unique_ptr<TiogaSTKIface> tg_;
    std::unique_ptr<MeshMotion> motion_;

    bool has_motion_{false};
};

}


#endif /* STKIFACE_H */
