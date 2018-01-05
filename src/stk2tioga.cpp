

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
#include "tioga.h"

typedef stk::mesh::Field<double, stk::mesh::Cartesian> VectorFieldType;
typedef stk::mesh::Field<double> ScalarFieldType;

const double pi = std::acos(-1.0);

void tag_procs(stk::mesh::MetaData& meta, stk::mesh::BulkData& bulk)
{
  int iproc = bulk.parallel_rank();
  ScalarFieldType *ipnode = meta.get_field<ScalarFieldType>
    (stk::topology::NODE_RANK, "pid_node");
  ScalarFieldType *ipelem = meta.get_field<ScalarFieldType>
    (stk::topology::ELEM_RANK, "pid_elem");

  stk::mesh::Selector msel = meta.locally_owned_part();
  const stk::mesh::BucketVector& nbkts = bulk.get_buckets(
    stk::topology::NODE_RANK, msel);

  for (auto b: nbkts) {
    double* ip = stk::mesh::field_data(*ipnode, *b);
    for(size_t in=0; in < b->size(); in++) {
      ip[in] = iproc;
    }
  }

  const stk::mesh::BucketVector& ebkts = bulk.get_buckets(
    stk::topology::ELEM_RANK, msel);

  for (auto b: ebkts) {
    double* ip = stk::mesh::field_data(*ipelem, *b);
    for(size_t in=0; in < b->size(); in++) {
      ip[in] = iproc;
    }
  }
}

void move_mesh(stk::mesh::MetaData& meta, stk::mesh::BulkData& bulk)
{
  double omega = 4.0 * 6.81041368647038;
  VectorFieldType* coords = meta.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "coordinates");

  stk::mesh::PartVector pvec(6);
  for (int i=0; i < 6; i++) {
    std::string pname = "Unspecified-" + std::to_string(i+2) + "-HEX";
    pvec[i] = meta.get_part(pname);
  }

  stk::mesh::Selector mselect = stk::mesh::selectUnion(pvec);
  stk::mesh::BucketVector bkts = bulk.get_buckets(
    stk::topology::NODE_RANK, mselect);
  for (auto b: bkts) {
    for (size_t in=0; in < b->size(); in++) {
      double *pts = stk::mesh::field_data(*coords, (*b)[in]);
      double xold = pts[0];
      double zold = pts[2];
      pts[0] = xold * std::cos(omega) + zold * std::sin(omega);
      pts[2] = - xold * std::sin(omega) + zold * std::cos(omega);
    }
  }
}

int main(int argc, char** argv)
{
  stk::ParallelMachine comm = stk::parallel_machine_init(&argc, &argv);

  int iproc = stk::parallel_machine_rank(comm);
  int nproc = stk::parallel_machine_size(comm);
  stk::mesh::MetaData meta;
  stk::mesh::BulkData bulk(meta, comm, stk::mesh::BulkData::NO_AUTO_AURA);

  std::string yaml_filename;
  if (argc == 2) {
    yaml_filename = argv[1];
  } else {
    throw std::runtime_error("Need input file");
  }
  YAML::Node inpfile = YAML::LoadFile(yaml_filename);

  stk::io::StkMeshIoBroker stkio(comm);

  if ((nproc > 1) && inpfile["decomposition_method"]) {
    auto decomp_method = inpfile["decomposition_method"].as<std::string>();
    stkio.property_add(Ioss::Property("DECOMPOSITION_METHOD", decomp_method));
  }

  if (iproc == 0)
      std::cout << "Preparing mesh meta data... " << std::endl;
  std::string inp_mesh = inpfile["input_mesh"].as<std::string>();
  stkio.add_mesh_database(inp_mesh, stk::io::READ_MESH);
  stkio.set_bulk_data(bulk);
  stkio.create_input_mesh();
  stkio.add_all_mesh_fields_as_input_fields();

  const YAML::Node& oset_info = inpfile["overset_info"];
  tioga_nalu::TiogaSTKIface tg(meta, bulk, oset_info);
  if (iproc == 0)
      std::cout << "Calling TIOGA setup... " << std::endl;
  tg.setup();

  ScalarFieldType& ipnode = meta.declare_field<ScalarFieldType>
    (stk::topology::NODE_RANK, "pid_node");
  ScalarFieldType& ipelem = meta.declare_field<ScalarFieldType>
    (stk::topology::ELEM_RANK, "pid_elem");
  stk::mesh::put_field(ipnode, meta.universal_part());
  stk::mesh::put_field(ipelem, meta.universal_part());

  if (iproc == 0)
      std::cout << "Loading mesh... " << std::endl;
  stkio.populate_bulk_data();
  if (iproc == 0)
      std::cout << "Initializing TIOGA... " << std::endl;
  tg.initialize();

  if (iproc == 0)
      std::cout << "Performing overset connectivity... " << std::endl;
  tg.execute();

  if (iproc == 0)
      std::cout << "Checking interpolation norms... " << std::endl;

  stk::parallel_machine_barrier(bulk.parallel());
  tg.check_soln_norm();
  bool do_write = true;
  if (inpfile["write_outputs"])
    do_write = inpfile["write_outputs"].as<bool>();

  if (do_write) {
    tag_procs(meta, bulk);
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
    stkio.add_field(fh, ipnode);
    stkio.add_field(fh, ipelem);

    stkio.begin_output_step(fh, 0.0);
    stkio.write_defined_output_fields(fh);
    stkio.end_output_step(fh);
  }

  bool dump_partitions = false;
  if (inpfile["dump_tioga_partitions"])
      dump_partitions = inpfile["dump_tioga_partitions"].as<bool>();
  if (dump_partitions) {
      if (iproc == 0)
          std::cout << "Dumping tioga partitions... " << std::endl;
      tg.tioga_iface().writeData(0, 0);
  }

  stk::parallel_machine_barrier(bulk.parallel());

  // stk::parallel_machine_finalize();
  return 0;
}
