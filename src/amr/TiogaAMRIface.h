#ifndef TIOGAAMRIFACE_H
#define TIOGAAMRIFACE_H

#include <memory>

#include "StructMesh.h"

namespace YAML {
class Node;
}

namespace tioga_amr {

class TiogaAMRIface
{
public:
    TiogaAMRIface();

    ~TiogaAMRIface();

    void load(const YAML::Node&);

    void initialize();

private:
    std::unique_ptr<StructMesh> m_mesh;
};

}

#endif /* TIOGAAMRIFACE_H */
