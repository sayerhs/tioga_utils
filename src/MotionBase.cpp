
#include "MotionBase.h"

#include "stk_mesh/base/Part.hpp"

namespace tioga_nalu {

void MotionBase::setup()
{
    for (auto pName: partNames_) {
        stk::mesh::Part* part = meta_.get_part(pName);
        if (nullptr == part)
            throw std::runtime_error(
                "MeshMotion: Invalid part name encountered: " + pName);
        else
            partVec_.push_back(part);
    }

    VectorFieldType& coordinates = meta_.declare_field<VectorFieldType>(
        stk::topology::NODE_RANK, "coordinates");
    VectorFieldType& current_coordinates = meta_.declare_field<VectorFieldType>(
        stk::topology::NODE_RANK, "current_coordinates");
    VectorFieldType& mesh_displacement = meta_.declare_field<VectorFieldType>(
        stk::topology::NODE_RANK, "mesh_displacement");

    for (auto* p: partVec_) {
        stk::mesh::put_field(coordinates, *p);
        stk::mesh::put_field(current_coordinates, *p);
        stk::mesh::put_field(mesh_displacement, *p);
    }
}

} // tioga_nalu
