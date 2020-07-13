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

    size_t write_outputs(const YAML::Node&, const double time = 0.0);

    void write_outputs(size_t fh, const double time);

    void register_mesh()
    {
        tg_->register_mesh();
    }

    void post_connectivity_work()
    {
        tg_->post_connectivity_work();
    }

    void register_solution();

    void update_solution()
    {
        tg_->update_solution(num_vars());
    }

    void move_mesh(int nt)
    {
        motion_->execute(nt);
    }

    std::string coordinates_name() const
    {
        return (has_motion_? "current_coordinates" : "coordinates");
    }

    bool has_motion() const { return has_motion_; }

    int num_timesteps() const
    {
        return (has_motion() ? motion_->num_steps() : 0);
    }

    double current_time() const
    {
        return (has_motion() ? motion_->current_time() : 0.0);
    }

    int& num_cell_vars() { return ncell_vars_; }
    const int& num_cell_vars() const { return ncell_vars_; }

    int& num_node_vars() { return nnode_vars_; }
    const int& num_node_vars() const { return nnode_vars_; }

    const int num_vars() const { return ncell_vars_ + nnode_vars_; }

private:
    void init_vars();

    stk::ParallelMachine comm_;
    stk::mesh::MetaData meta_;
    stk::mesh::BulkData bulk_;
    stk::io::StkMeshIoBroker stkio_;

    std::unique_ptr<TiogaSTKIface> tg_;
    std::unique_ptr<MeshMotion> motion_;

    int ncell_vars_{0};
    int nnode_vars_{0};

    bool has_motion_{false};
};

}


#endif /* STKIFACE_H */
