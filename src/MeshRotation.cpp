
#include "MeshRotation.h"

#include <cassert>
#include <cmath>

namespace tioga_nalu {

MeshRotation::MeshRotation(
    stk::mesh::MetaData& meta,
    stk::mesh::BulkData& bulk,
    const YAML::Node& node
) : MotionBase(meta, bulk)
{
    load(node);
}

void MeshRotation::load(const YAML::Node& node)
{
    const auto& fparts = node["mesh_parts"];
    if (fparts.Type() == YAML::NodeType::Scalar) {
        partNames_.push_back(fparts.as<std::string>());
    } else {
        partNames_ = fparts.as<std::vector<std::string>>();
    }

    omega_ = node["omega"].as<double>();
    axis_ = node["axis"].as<std::vector<double>>();
    origin_ = node["origin"].as<std::vector<double>>();

    assert(axis_.size() == 3);
    assert(origin_.size() == 3);
}

void MeshRotation::initialize(double initial_time)
{
    rotate_mesh(initial_time);
}

void MeshRotation::execute(double current_time)
{
    rotate_mesh(current_time);
}

void MeshRotation::rotate_mesh(double current_time)
{
    const int ndim = meta_.spatial_dimension();
    VectorFieldType* modelCoords = meta_.get_field<VectorFieldType>(
        stk::topology::NODE_RANK, "coordinates");
    VectorFieldType* currCoords = meta_.get_field<VectorFieldType>(
        stk::topology::NODE_RANK, "current_coordinates");
    VectorFieldType* displacement = meta_.get_field<VectorFieldType>(
        stk::topology::NODE_RANK, "mesh_displacement");

    stk::mesh::Selector sel = stk::mesh::selectUnion(partVec_);
    const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);

    double mag = 0.0;
    for (int d=0; d < ndim; d++)
        mag += axis_[d] * axis_[d];
    mag = std::sqrt(mag);
    const double angle = omega_ * current_time;
    const double cosang = std::cos(0.5*angle);
    const double sinang = std::sin(0.5*angle);
    const double q0 = cosang;
    const double q1 = sinang * axis_[0] / mag;
    const double q2 = sinang * axis_[1] / mag;
    const double q3 = sinang * axis_[2] / mag;

    for (auto b: bkts) {
        for (size_t in=0; in < b->size(); in++) {
            auto node = (*b)[in];
            double* oldxyz = stk::mesh::field_data(*modelCoords, node);
            double* xyz = stk::mesh::field_data(*currCoords, node);
            double* dx = stk::mesh::field_data(*displacement, node);

            const double cx = oldxyz[0] - origin_[0];
            const double cy = oldxyz[1] - origin_[1];
            const double cz = oldxyz[2] - origin_[2];

            xyz[0] = (q0*q0 + q1*q1 - q2*q2 - q3*q3) * cx +
                2.0 * (q1*q2 - q0*q3) * cy +
                2.0 * (q0*q2 + q1*q3) * cz + origin_[0];

            xyz[1] = 2.0 * (q1*q2 + q0*q3) * cx +
                (q0*q0 - q1*q1 + q2*q2 - q3*q3) * cy +
                2.0 * (q2*q3 - q0*q1) * cz + origin_[1];

            xyz[2] = 2.0 * (q1*q3 - q0*q2) * cx +
                2.0 * (q0*q1 + q2*q3) * cy +
                (q0*q0 - q1*q1 - q2*q2 + q3*q3) * cz + origin_[2];

            for (int d=0; d < ndim; d++)
                dx[d] = xyz[d] - oldxyz[d];
        }
    }
}

} // tioga_nalu
