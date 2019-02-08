
#include "MeshMotion.h"
#include "MeshRotation.h"

#include "stk_mesh/base/Field.hpp"

namespace tioga_nalu {

MeshMotion::MeshMotion(
    stk::mesh::MetaData& meta,
    stk::mesh::BulkData& bulk,
    const YAML::Node& node
) : meta_(meta),
    bulk_(bulk)
{
    load(node);
}

void MeshMotion::load(const YAML::Node& node)
{
    const auto& minfo = node["motion_group"];

    const int num_groups = minfo.size();
    meshMotionVec_.resize(num_groups);

    for (int i=0; i < num_groups; i++) {
        const auto& motion_def = minfo[i];
        std::string type = "rotation";
        if (motion_def["type"]) type = motion_def["type"].as<std::string>();

        if (type == "rotation") {
            meshMotionVec_[i].reset(new MeshRotation(meta_, bulk_, motion_def));
        } else {
            throw std::runtime_error("MeshMotion: Invalid mesh motion type: " + type);
        }
    }

    if (node["start_time"])
        startTime_ = node["start_time"].as<double>();
    if (node["num_time_steps"]) {
        numSteps_ = node["num_time_steps"].as<int>();
        deltaT_ = node["delta_t"].as<double>();
    }
}

void MeshMotion::setup()
{
    VectorFieldType& coordinates = meta_.declare_field<VectorFieldType>(
        stk::topology::NODE_RANK, "coordinates");
    VectorFieldType& current_coordinates = meta_.declare_field<VectorFieldType>(
        stk::topology::NODE_RANK, "current_coordinates");
    VectorFieldType& mesh_displacement = meta_.declare_field<VectorFieldType>(
        stk::topology::NODE_RANK, "mesh_displacement");

    stk::mesh::put_field_on_mesh(coordinates, meta_.universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(current_coordinates, meta_.universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(mesh_displacement, meta_.universal_part(), nullptr);

    for (auto& mm: meshMotionVec_)
        mm->setup();
}

void MeshMotion::initialize()
{
    init_coordinates();
    if (startTime_ > 0.0)
        for (auto& mm: meshMotionVec_)
            mm->initialize(startTime_);
}

void MeshMotion::execute(const int istep)
{
    const double curr_time = startTime_ + (istep + 1) * deltaT_;
    currentTime_ = curr_time;

    for (auto& mm: meshMotionVec_)
        mm->execute(curr_time);
}

void MeshMotion::init_coordinates()
{
    const int ndim = meta_.spatial_dimension();
    VectorFieldType* modelCoords = meta_.get_field<VectorFieldType>(
        stk::topology::NODE_RANK, "coordinates");
    VectorFieldType* currCoords = meta_.get_field<VectorFieldType>(
        stk::topology::NODE_RANK, "current_coordinates");

    stk::mesh::Selector sel = meta_.universal_part();
    const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);

    for (auto b: bkts) {
        for (size_t in=0; in < b->size(); in++) {
            auto node = (*b)[in];
            double* oldxyz = stk::mesh::field_data(*modelCoords, node);
            double* xyz = stk::mesh::field_data(*currCoords, node);

            for (int d=0; d < ndim; d++)
                xyz[d] = oldxyz[d];
        }
    }
}

} // tioga_nalu
