
#include "TiogaSTKIface.h"

#include "master_element/MasterElement.h"
#include "master_element/Hex8CVFEM.h"
#include "stk_util/parallel/ParallelReduce.hpp"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <string>

#include "tioga.h"

namespace tioga_nalu {

TiogaSTKIface::TiogaSTKIface(
  stk::mesh::MetaData& meta,
  stk::mesh::BulkData& bulk,
  const YAML::Node& node
) : meta_(meta),
    bulk_(bulk),
    tg_(new tioga()),
    ovsetGhosting_(nullptr),
    inactivePartName_("nalu_overset_hole_elements")
{
  load(node);
}

TiogaSTKIface::~TiogaSTKIface()
{}

void
TiogaSTKIface::load(const YAML::Node& node)
{
  const YAML::Node& oset_groups = node["mesh_group"];

  int num_meshes = oset_groups.size();
  blocks_.resize(num_meshes);

  for (int i = 0; i < num_meshes; i++) {
    blocks_[i].reset(new TiogaBlock(meta_, bulk_, oset_groups[i], i + 1));
  }
}

void TiogaSTKIface::setup()
{
  for (auto& tb: blocks_) {
    tb->setup();
  }

  // Initialize the inactive part
  inactivePart_ = &meta_.declare_part(inactivePartName_, stk::topology::ELEM_RANK);
}

void TiogaSTKIface::initialize()
{
  tg_->setCommunicator(bulk_.parallel(),
                       bulk_.parallel_rank(),
                       bulk_.parallel_size());

  for (auto& tb: blocks_) {
    tb->initialize();
  }
}

void TiogaSTKIface::initialize_ghosting()
{
  // TODO: Update ghosting modification to use optimized version in
  // Non-conformal case.
  bulk_.modification_begin();
  if (ovsetGhosting_ == nullptr) {
    const std::string ghostName = "nalu_overset_ghosting";
    ovsetGhosting_ = &(bulk_.create_ghosting(ghostName));
  } else {
    bulk_.destroy_ghosting(*ovsetGhosting_);
  }
  bulk_.modification_end();
}

void TiogaSTKIface::execute()
{
  reset_data_structures();

  initialize_ghosting();

  // Update the coordinates for TIOGA and register updates to the TIOGA mesh block.
  for (auto& tb: blocks_) {
    tb->update_coords();
    tb->register_block(*tg_);
  }

  // Determine overset connectivity
  tg_->profile();
  tg_->performConnectivity();

  for (auto& tb: blocks_) {
    // Update IBLANK information at nodes and elements
    tb->update_iblanks();
    tb->update_iblank_cell();

    // For each block determine donor elements that needs to be ghosted to other
    // MPI ranks
    tb->get_donor_info(*tg_, elemsToGhost_);
  }

  // TODO: Combine bulk modification for ghosting and inactive part population

  // Collect all elements to be ghosted and update ghosting so that the elements
  // are available when generating {fringeNode, donorElement} pairs in the next
  // step.
  update_ghosting();

  populate_inactive_part();

  // Update overset fringe connectivity information for Constraint based algorithm
  update_fringe_info();
}

void TiogaSTKIface::reset_data_structures()
{
  elemsToGhost_.clear();
  ovsetInfo_.clear();

  // Reset inactivePart_
  bulk_.modification_begin();
  {
    stk::mesh::PartVector add_parts;
    stk::mesh::PartVector remove_parts;
    remove_parts.push_back(inactivePart_);
    for (auto elem : holeElems_) {
      bulk_.change_entity_parts(elem, add_parts, remove_parts);
    }
  }
  bulk_.modification_end();

  holeElems_.clear();
}

void TiogaSTKIface::update_ghosting()
{
  uint64_t g_ghostCount = 0;
  uint64_t nGhostLocal = elemsToGhost_.size();
  stk::all_reduce_sum(bulk_.parallel(), &nGhostLocal, &g_ghostCount, 1);

  if (g_ghostCount > 0) {
    bulk_.modification_begin();
    bulk_.change_ghosting(*ovsetGhosting_, elemsToGhost_);
    bulk_.modification_end();
  }

  // std::cout << bulk_.parallel_rank() << "\t" << g_ghostCount << "\t" << nGhostLocal << std::endl;
  // stk::mesh::Part& cpart = bulk_.ghosting_part(*ovsetGhosting_);
  // stk::mesh::Selector sel = cpart;
  // const stk::mesh::BucketVector& cgbkts = bulk_.get_buckets(stk::topology::ELEM_RANK, sel);
  // size_t num_elems = 0;
  // for (size_t i=0; i<cgbkts.size(); i++) {
  //   num_elems += cgbkts[i]->size();
  // }
  // std::cout << bulk_.parallel_rank() << "\t" << num_elems << std::endl;

}

void TiogaSTKIface::populate_inactive_part()
{
  stk::mesh::PartVector toParts;
  toParts.push_back(inactivePart_);

  bulk_.modification_begin();
  {
    for (auto& tb: blocks_) {
      auto iblank_cell = tb->iblank_cell();
      auto elem_gid = tb->elem_id_map();

      for (auto ib: iblank_cell) {
        if (ib != 0) continue;

        stk::mesh::Entity elem = bulk_.get_entity(
          stk::topology::ELEM_RANK, elem_gid[ib]);
        bulk_.change_entity_parts(elem, toParts);
        holeElems_.push_back(elem);
      }
    }
  }
  bulk_.modification_end();
  // std::cout << bulk_.parallel_rank() << "\t" << holeElems_.size() << "\t"
  //           << oldSize << std::endl;
}

void TiogaSTKIface::update_fringe_info()
{
  double maxError = -1.0e16;
  int iproc = bulk_.parallel_rank();
  int nproc = bulk_.parallel_size();
  // std::string fname = "tioga_fringe." + std::to_string(nproc) + "." + std::to_string(iproc);
  // std::ofstream fout(fname, std::ios::out);
  std::unique_ptr<sierra::nalu::MasterElement> meSCS(new sierra::nalu::HexSCS());
  std::vector<double> elemxyz(24);
  std::vector<double> intxyz(3);
  std::vector<int> receptors;
  tg_->getReceptorInfo(receptors);
  ovsetInfo_.resize(receptors.size()/3);
  //std::cout << bulk_.parallel_rank() << "\t" << receptors.size();

  VectorFieldType *coords = meta_.get_field<VectorFieldType>
    (stk::topology::NODE_RANK, "coordinates");
  size_t ncount = receptors.size();
  for (size_t i=0, ip=0; i<ncount; i+=3, ip++) {
    int nid = receptors[i];                          // TiogaBlock node index
    int mtag = receptors[i+1] - 1;                   // Block index
    int donorID = receptors[i+2];                    // STK Global ID of the donor element
    int nodeID = blocks_[mtag]->node_id_map()[nid];  // STK Global ID of the fringe node
    stk::mesh::Entity node = bulk_.get_entity(stk::topology::NODE_RANK, nodeID);
    stk::mesh::Entity elem = bulk_.get_entity(stk::topology::ELEM_RANK, donorID);

    // int nodeRank = bulk_.parallel_owner_rank(node);
    // if (nodeRank != myRank) {
    //   std::cout << myRank << "\t" << nodeID << "\t" << donorID << std::endl;
    //   continue;
    // }

#ifndef NDEBUG
    if (!bulk_.is_valid(elem))
      throw std::runtime_error(
        "Invalid element encountered in overset mesh connectivity");
#endif

    ovsetInfo_[ip].reset(new OversetInfo);
    auto& info = ovsetInfo_[ip];
    info->node_ = node;
    info->donorElem_ = elem;

    const double* nxyz = stk::mesh::field_data(*coords, node);
    for (int i=0; i<3; i++) {
      info->nodalCoords_[i] = nxyz[i];
    }

    const stk::mesh::Entity* elem_nodes = bulk_.begin_nodes(elem);
    for (unsigned int in=0; in<bulk_.num_nodes(elem); in++) {
      stk::mesh::Entity enode = elem_nodes[in];
      const double* exyz = stk::mesh::field_data(*coords, enode);
      for (int j=0; j < 3; j++) {
        const int offset = j * 8 + in;
        elemxyz[offset] = exyz[j];
      }
    }
    meSCS->isInElement(
      elemxyz.data(), info->nodalCoords_.data(), info->isoCoords_.data());
    meSCS->interpolatePoint(3, info->isoCoords_.data(), elemxyz.data(), intxyz.data());

    double error = 0.0;
    for (int i=0; i<3; i++) {
      error += info->nodalCoords_[i] - intxyz[i];
    }
    if (std::fabs(error) > maxError) maxError = error;
  }

  if (bulk_.parallel_rank() == 0)
    std::cout << "\nNalu CVFEM interpolation results: " << std::endl;

  stk::parallel_machine_barrier(bulk_.parallel());
  if (ncount > 0) {
    std::cout << "    Proc: " << iproc
              << "; Max error = " << maxError << std::endl;
  }
}

void TiogaSTKIface::check_soln_norm()
{
  stk::parallel_machine_barrier(bulk_.parallel());
  if (bulk_.parallel_rank() == 0) {
    std::cout << "\n\n-- Interpolation error statistics --\n"
              << "Proc ID.    BodyTag    Error(L2 norm)" << std::endl;
  }
  for (auto& tb: blocks_) {
    tb->register_solution(*tg_);
  }

  tg_->dataUpdate(1, 0);

  int nblocks = blocks_.size();
  for (int i=0; i<nblocks; i++) {
    auto& tb = blocks_[i];
    double rnorm = tb->calculate_residuals();
    std::cout << bulk_.parallel_rank() << "\t" << i << "\t" << rnorm << std::endl;
  }
}

}  // tioga
