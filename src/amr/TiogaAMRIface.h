#ifndef TIOGAAMRIFACE_H
#define TIOGAAMRIFACE_H

#include <memory>

#include "StructMesh.h"

namespace YAML {
class Node;
}

namespace TIOGA {
class tioga;
}

namespace tioga_amr {

class TiogaAMRIface
{
public:
    TiogaAMRIface();

    ~TiogaAMRIface();

    void load(const YAML::Node&);

    void initialize();

    void register_mesh(TIOGA::tioga&);

private:
    std::unique_ptr<StructMesh> m_mesh;

    //! Integers per grid for TIOGA call
    static constexpr int ints_per_grid{10};

    //! Reals per grid for TIOGA call
    static constexpr int reals_per_grid{6};

    //! Number of ghost points
    static constexpr int num_ghost{3};
};

}

#endif /* TIOGAAMRIFACE_H */
