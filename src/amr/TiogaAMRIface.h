#ifndef TIOGAAMRIFACE_H
#define TIOGAAMRIFACE_H

#include <memory>

#include "StructMesh.h"
#include "NgpAMRTypes.h"

namespace YAML {
class Node;
}

namespace TIOGA {
class tioga;
struct AMRMeshInfo;
}

namespace tioga_amr {

struct NgpAMRInfo
{
    template <typename T> using AType = NgpAmrDualArray<T>;

    NgpAMRInfo(const int nglobal, const int nlocal);

    // Arrays of size ngrids_global
    AType<int> level;
    AType<int> mpi_rank;
    AType<int> local_id;
    AType<int> ilow;
    AType<int> ihigh;
    AType<int> dims;
    AType<amrex::Real> xlo;
    AType<amrex::Real> dx;

    // Arrays of size ngrids_local
    AType<int> global_idmap;
    AType<int*> iblank_node;
    AType<int*> iblank_cell;
    AType<amrex::Real*> qcell;
    AType<amrex::Real*> qnode;
};

class TiogaAMRIface
{
public:
    TiogaAMRIface();

    ~TiogaAMRIface();

    void load(const YAML::Node&);

    void initialize();

    void register_mesh(TIOGA::tioga&, const bool verbose=false);

    void register_solution(TIOGA::tioga&);

    void update_solution();

    void write_outputs(const int time_index=0, const double time=0.0);

    int num_total_vars() const { return m_ncell_vars + m_nnode_vars; }

    int num_cell_vars() const { return m_ncell_vars; }

    int num_node_vars() const { return m_nnode_vars; }

    // Public for CUDA

    void init_var(Field&, const int nvars, const amrex::Real offset);

private:
    void amr_to_tioga_info();

    std::unique_ptr<StructMesh> m_mesh;

    std::unique_ptr<TIOGA::AMRMeshInfo> m_info;
    std::unique_ptr<NgpAMRInfo> m_amr_data;

    //! Reference to cell variable field
    Field* m_qcell{nullptr};

    //! Reference to node variable field
    Field* m_qnode{nullptr};

    std::vector<int> m_ints;

    std::vector<double> m_reals;

    //! Number of ghost cells
    int m_num_ghost{3};

    //! Number of components in the cell field
    int m_ncell_vars{0};

    //! Number of components for the node field
    int m_nnode_vars{0};

    //! Integers per grid for TIOGA call
    static constexpr int ints_per_grid{10};

    //! Reals per grid for TIOGA call
    static constexpr int reals_per_grid{6};
};

}

#endif /* TIOGAAMRIFACE_H */
