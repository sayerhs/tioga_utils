

#include <Shards_BasicTopologies.hpp>

#include <stk_util/parallel/Parallel.hpp>
#include <stk_util/parallel/BroadcastArg.hpp>

#include <stk_mesh/base/FindRestriction.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/Comm.hpp>
#include <stk_mesh/base/Stencils.hpp>
#include <stk_mesh/base/TopologyDimensions.hpp>
#include <stk_mesh/base/FEMHelpers.hpp>

#include <stk_io/IossBridge.hpp>
#include <stk_io/StkMeshIoBroker.hpp>
#include <Ionit_Initializer.h>

#include <cmath>

#include "TiogaSTKIface.h"

typedef stk::mesh::Field<double, stk::mesh::Cartesian> VectorFieldType;
typedef stk::mesh::Field<double> ScalarFieldType;

const double pi = std::acos(-1.0);

int main(int argc, char** argv)
{
  stk::ParallelMachine comm = stk::parallel_machine_init(&argc, &argv);

  int iproc = stk::parallel_machine_rank(comm);
  int nproc = stk::parallel_machine_size(comm);
  stk::mesh::MetaData meta;
  stk::mesh::BulkData bulk(meta, comm);

  std::string yaml_filename;
  if (argc == 2) {
    yaml_filename = argv[1];
  } else {
    throw std::runtime_error("Need input file");
  }
  YAML::Node inpfile = YAML::LoadFile(yaml_filename);

  stk::io::StkMeshIoBroker stkio(comm);

  if (nproc > 1)
    stkio.property_add(Ioss::Property("DECOMPOSITION_METHOD", "rcb"));

  std::string inp_mesh = inpfile["input_mesh"].as<std::string>();
  stkio.add_mesh_database(inp_mesh, stk::io::READ_MESH);
  stkio.set_bulk_data(bulk);
  stkio.create_input_mesh();
  stkio.add_all_mesh_fields_as_input_fields();

  const YAML::Node& oset_info = inpfile["overset_info"];
  tioga_nalu::TiogaSTKIface tg(meta, bulk, oset_info);
  tg.setup();

  stkio.populate_bulk_data();
  tg.initialize();

  tg.execute();

  tg.check_soln_norm();

  bool do_write = true;
  if (inpfile["write_outputs"])
    do_write = inpfile["write_outputs"].as<bool>();

  if (do_write) {
    ScalarFieldType* ibf = meta.get_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "iblank");
    ScalarFieldType* ibcell = meta.get_field<ScalarFieldType>(
      stk::topology::ELEM_RANK, "iblank_cell");

    std::string out_mesh = inpfile["output_mesh"].as<std::string>();
    if (iproc == 0)
      std::cout << "Writing output file: " << out_mesh << std::endl;
    size_t fh = stkio.create_output_mesh(out_mesh, stk::io::WRITE_RESTART);
    stkio.add_field(fh, *ibf);
    stkio.add_field(fh, *ibcell);
    //stkio.add_field(fh, displ);

    stkio.begin_output_step(fh, 0.0);
    stkio.write_defined_output_fields(fh);
    stkio.end_output_step(fh);

    stk::mesh::Entity node = bulk.get_entity(
      stk::topology::NODE_RANK, 4332);
    if (bulk.is_valid(node)) {
      double* ibval = stk::mesh::field_data(*ibf, node);
      std::cout << "IBLANK: " << iproc << "\t" << *ibval << "\t"
                << bulk.bucket(node).owned() << std::endl;
    } else {
      std::cout << "IBLANK: " << iproc << "Doesnt exist" << std::endl;
    }
  }

  // Bouncing cylinder moving mesh test
  // VectorFieldType& displ = meta.declare_field<VectorFieldType>(
  //   stk::topology::NODE_RANK, "mesh_displacement");
  // stk::mesh::Part* cyl_block = meta.get_part("cylinder");
  // stk::mesh::put_field(displ, *cyl_block);
  // VectorFieldType* coords = meta.get_field<VectorFieldType>(
  //   stk::topology::NODE_RANK, "coordinates");

  // for (int i=0; i<41; i++) {
  //   //std::cout << bulk.parallel_rank() << "\t" << i << std::endl;
  //   stk::mesh::Selector mselect(*cyl_block);
  //   stk::mesh::BucketVector bkts = bulk.get_buckets(
  //     stk::topology::NODE_RANK, mselect);
  //   for (auto b: bkts) {
  //     for (size_t in=0; in < b->size(); in++) {
  //       double* pts = stk::mesh::field_data(*coords, (*b)[in]);
  //       double* dx = stk::mesh::field_data(displ, (*b)[in]);
  //       pts[0] += 0.1;
  //       pts[2] += std::sin(2.0*pi/40.0*i) - dx[2];
  //       dx[0] = i*0.1;
  //       dx[2] = std::sin(2.0*pi/40.0*i);
  //     }
  //   }

  //   tg.execute();
  //   stkio.begin_output_step(fh, 1.0*(i+1));
  //   stkio.write_defined_output_fields(fh);
  //   stkio.end_output_step(fh);
  // }

  stk::parallel_machine_finalize();
  return 0;
}
