#ifndef MESHMOTION_H
#define MESHMOTION_H

#include "MotionBase.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/BulkData.hpp"

#include "yaml-cpp/yaml.h"

#include <memory>

namespace tioga_nalu {

class MeshMotion
{
public:
    MeshMotion(
        stk::mesh::MetaData&,
        stk::mesh::BulkData&,
        const YAML::Node&);

    ~MeshMotion() {}

    virtual void setup();

    virtual void initialize();

    virtual void execute(const int);

    int num_steps() { return numSteps_; }

private:
    MeshMotion() = delete;
    MeshMotion(const MeshMotion&) = delete;

    void load(const YAML::Node&);

    void init_coordinates();

    stk::mesh::MetaData& meta_;

    stk::mesh::BulkData& bulk_;

    std::vector<std::unique_ptr<MotionBase>> meshMotionVec_;

    double startTime_{0.0};
    double deltaT_{0.0};

    int numSteps_{0};
};


} // tioga_nalu

#endif /* MESHMOTION_H */
