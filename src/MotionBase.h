#ifndef MOTIONBASE_H
#define MOTIONBASE_H

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/CoordinateSystems.hpp"
#include "stk_mesh/base/Field.hpp"

#include <vector>
#include <string>

namespace tioga_nalu {

typedef stk::mesh::Field<double, stk::mesh::Cartesian> VectorFieldType;
typedef stk::mesh::Field<double> ScalarFieldType;

class MotionBase
{
public:
  MotionBase(stk::mesh::MetaData &meta, stk::mesh::BulkData &bulk)
      : meta_(meta), bulk_(bulk) {}

  virtual ~MotionBase() {}

  virtual void setup();

  virtual void initialize(double) = 0;

  virtual void execute(double) = 0;

private:
    MotionBase() = delete;
    MotionBase(const MotionBase&) = delete;

protected:
    stk::mesh::MetaData& meta_;

    stk::mesh::BulkData& bulk_;

    std::vector<std::string> partNames_;

    stk::mesh::PartVector partVec_;
};


} // tioga_nalu

#endif /* MOTIONBASE_H */
