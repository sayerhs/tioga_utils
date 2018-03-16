#ifndef MESHROTATION_H
#define MESHROTATION_H

#include "MotionBase.h"

#include "yaml-cpp/yaml.h"

namespace tioga_nalu {

class MeshRotation : public MotionBase
{
public:
    MeshRotation(
        stk::mesh::MetaData&,
        stk::mesh::BulkData&,
        const YAML::Node&);

    virtual ~MeshRotation() {}

    virtual void initialize(double);

    virtual void execute(double);

private:
    MeshRotation() = delete;
    MeshRotation(const MeshRotation&) = delete;

    void load(const YAML::Node&);

    void rotate_mesh(double);

    std::vector<double> origin_{0.0, 0.0, 0.0};

    std::vector<double> axis_;

    double omega_{0.0};
};


} // tioga_nalu

#endif /* MESHROTATION_H */
