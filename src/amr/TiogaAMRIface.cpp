#include "TiogaAMRIface.h"
#include "amrex_yaml.h"

#include "AMReX_ParmParse.H"

namespace tioga_amr {

TiogaAMRIface::TiogaAMRIface(const YAML::Node& node)
{
    load(node);

    m_mesh.reset(new StructMesh());
}

TiogaAMRIface::~TiogaAMRIface() = default;

void TiogaAMRIface::load(const YAML::Node& node)
{
    populate_parameters(node, "amr");
    populate_parameters(node, "geometry");
}

void TiogaAMRIface::initialize()
{
    m_mesh->initialize_mesh();
}

} // namespace tioga_amr
