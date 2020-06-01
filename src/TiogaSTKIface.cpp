
#include "TiogaSTKIface.h"
#include "TiogaRef.h"
#include "Timer.h"

#ifdef HAS_NALU_WIND
#include "NaluEnv.h"

#include "master_element/MasterElementFactory.h"
#include "master_element/MasterElement.h"
#include "master_element/Hex8CVFEM.h"
#include "utils/StkHelpers.h"
#endif

#include "stk_util/parallel/ParallelReduce.hpp"
#include "stk_mesh/base/FieldParallel.hpp"

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
  const YAML::Node& node,
  const std::string& coordsName
) : meta_(meta),
    bulk_(bulk),
    tg_(TiogaRef::self().get()),
    ovsetGhosting_(nullptr),
    coordsName_(coordsName)
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
    blocks_[i].reset(
        new TiogaBlock(meta_, bulk_, oset_groups[i], coordsName_, i + 1));
  }
}

void TiogaSTKIface::setup()
{
  auto timeMon = get_timer("TiogaSTKIface::setup");

  for (auto& tb: blocks_) {
    tb->setup();
  }
}

void TiogaSTKIface::initialize()
{
  // tg_.setCommunicator(bulk_.parallel(),
  //                      bulk_.parallel_rank(),
  //                      bulk_.parallel_size());

  auto timeMon = get_timer("TiogaSTKIface::initialize");
  for (auto& tb: blocks_) {
    tb->initialize();
  }
}

void TiogaSTKIface::execute()
{
  reset_data_structures();

  // Update the coordinates for TIOGA and register updates to the TIOGA mesh block.
  for (auto& tb: blocks_) {
    tb->update_coords();
    tb->register_block(tg_);
  }

  // Determine overset connectivity
  {
      auto timeMon = get_timer("TIOGA::profile");
      tg_.profile();
  }
  {
      auto timeMon = get_timer("TIOGA::performConnectivity");
      tg_.performConnectivity();
  }

  for (auto& tb: blocks_) {
    // Update IBLANK information at nodes and elements
    tb->update_iblanks();
    tb->update_iblank_cell();

    // For each block determine donor elements that needs to be ghosted to other
    // MPI ranks
    tb->get_donor_info(tg_, elemsToGhost_);
  }

  // Synchronize IBLANK data for shared nodes
  ScalarFieldType* ibf = meta_.get_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "iblank");
  std::vector<const stk::mesh::FieldBase*> pvec{ibf};
  stk::mesh::copy_owned_to_shared(bulk_, pvec);

#ifdef HAS_NALU_WIND
  get_receptor_info();

  // Collect all elements to be ghosted and update ghosting so that the elements
  // are available when generating {fringeNode, donorElement} pairs in the next
  // step.
  update_ghosting();

  // Update overset fringe connectivity information for Constraint based algorithm
  // update_fringe_info();
  populate_overset_info();
#endif
}

void TiogaSTKIface::reset_data_structures()
{
  auto timeMon = get_timer("TiogaSTKIface::reset_data_structures");
  elemsToGhost_.clear();
  ovsetInfo_.clear();

  receptorIDs_.clear();
  donorIDs_.clear();
}

void TiogaSTKIface::update_ghosting()
{
#ifdef HAS_NALU_WIND
  std::vector<stk::mesh::EntityKey> recvGhostsToRemove;

  if (ovsetGhosting_ != nullptr) {
    stk::mesh::EntityProcVec currentSendGhosts;
    ovsetGhosting_->send_list(currentSendGhosts);

    sierra::nalu::compute_precise_ghosting_lists(
      bulk_, elemsToGhost_, currentSendGhosts, recvGhostsToRemove);
  }

  size_t local[2] = {elemsToGhost_.size(), recvGhostsToRemove.size()};
  size_t global[2] = {0, 0};
  stk::all_reduce_sum(bulk_.parallel(), local, global, 2);

  if ((global[0] > 0) || (global[1] > 0)) {
    bulk_.modification_begin();
    if (ovsetGhosting_ == nullptr) {
      const std::string ghostName = "nalu_overset_ghosting";
      ovsetGhosting_ = &(bulk_.create_ghosting(ghostName));
    }
    bulk_.change_ghosting(*ovsetGhosting_, elemsToGhost_, recvGhostsToRemove);
    bulk_.modification_end();

    sierra::nalu::populate_ghost_comm_procs(bulk_, *ovsetGhosting_, ghostCommProcs_);
  }
#endif
}

void TiogaSTKIface::check_soln_norm()
{
  auto timeMon = get_timer("TiogaSTKIface::check_soln_norm");
  stk::parallel_machine_barrier(bulk_.parallel());
  // if (bulk_.parallel_rank() == 0) {
  //   std::cout << "\n\n-- Interpolation error statistics --\n"
  //             << "Proc ID.    BodyTag    Error(L2 norm)" << std::endl;
  // }
  for (auto& tb: blocks_) {
    tb->register_solution_old(tg_);
  }

  {
      auto timeMon = get_timer("TIOGA::dataUpdate");
      tg_.dataUpdate(1, 0);
  }

  int nblocks = blocks_.size();
  double maxNorm = -1.0e16;
  double g_maxNorm = -1.0e16;
  for (int i=0; i<nblocks; i++) {
    auto& tb = blocks_[i];
    double rnorm = tb->calculate_residuals_old();
    maxNorm = std::max(rnorm, maxNorm);
  }

  stk::all_reduce_max(bulk_.parallel(), &maxNorm, &g_maxNorm, 1);
  if (bulk_.parallel_rank() == 0)
      std::cout << "TIOGA Interpolation error: max L2 error: "
                << g_maxNorm << std::endl;
}

void
TiogaSTKIface::get_receptor_info()
{
  ScalarFieldType* ibf = meta_.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "iblank");
  auto timeMon = get_timer("TiogaSTKIface::get_receptor_info");

  std::vector<unsigned long> nodesToReset;

  // Ask TIOGA for the fringe points and their corresponding donor element
  // information
  std::vector<int> receptors;
  {
      auto timeMon1 = get_timer("TIOGA::getReceptorInfo");
      tg_.getReceptorInfo(receptors);
  }

  // Process TIOGA receptors array and fill in the oversetInfoVec used for
  // subsequent Nalu computations.
  //
  // TIOGA returns a integer array that contains 3 entries per receptor node:
  //   - the local node index within the tioga mesh data array
  //   - the local mesh tag (block index) for that mesh during registration
  //   - the STK global ID for the donor element
  //
  size_t ncount = receptors.size();
  stk::mesh::EntityId donorID = std::numeric_limits<stk::mesh::EntityId>::max();
#ifdef TIOGA_HAS_UINT64T
  int rec_offset = 4;
#else
  int rec_offset = 3;
#endif
  for (size_t i=0; i<ncount; i+=rec_offset) {
    int nid = receptors[i];                          // TiogaBlock node index
    int mtag = receptors[i+1] - 1;                   // Block index
#ifdef TIOGA_HAS_UINT64T
    std::memcpy(&donorID, &receptors[i+2], sizeof(uint64_t));
#else
    donorID = receptors[i+2];
#endif
    int donorID = receptors[i+2];                    // STK Global ID of the donor element
    int nodeID = blocks_[mtag]->node_id_map()[nid];  // STK Global ID of the fringe node
    stk::mesh::Entity node = bulk_.get_entity(stk::topology::NODE_RANK, nodeID);

    if (!bulk_.bucket(node).owned()) {
      // We have a shared node that is marked as fringe. Ensure that the owning
      // proc also has this marked as fringe.
      double ibval = *stk::mesh::field_data(*ibf, node);

      if (ibval > -1.0) {
        // Disagreement between owner and shared status of iblank. Communicate
        // to owner that it must be a fringe.
        nodesToReset.push_back(bulk_.parallel_owner_rank(node));
        nodesToReset.push_back(nodeID);
        nodesToReset.push_back(donorID);
      }
    }

    // Stash the IDs for populating OversetInfo
    donorIDs_.push_back(donorID);
    receptorIDs_.push_back(nodeID);
  }

  int numLocal = nodesToReset.size();
  int iproc = bulk_.parallel_rank();
  int nproc = bulk_.parallel_size();
  std::vector<int> nbPerProc(nproc);
  MPI_Allgather(&numLocal, 1, MPI_INT, nbPerProc.data(), 1, MPI_INT, bulk_.parallel());

  // Total number of entities across all procs
  int nTotalEntities = 0;
  for (auto ielm: nbPerProc)
      nTotalEntities += ielm;

  // If no disagreements were detected then we are done here
  if (nTotalEntities < 1) return;

#if 1
  if (iproc == 0)
      std::cout << "TIOGA: Detected fringe/field mismatch on " << (nTotalEntities/3)
                << " entities" << std::endl;
#endif

  // Prepare data structures for reconciliation
  std::vector<int> offsets(nproc+1);
  std::vector<unsigned long> allEntities(nTotalEntities);

  offsets[0] = 0;
  for (int i=1; i <= nproc; ++i) {
    offsets[i] = offsets[i-1] + nbPerProc[i-1];
  }

  MPI_Allgatherv(nodesToReset.data(), numLocal, MPI_UNSIGNED_LONG, allEntities.data(),
                 nbPerProc.data(), offsets.data(), MPI_UNSIGNED_LONG, bulk_.parallel());

  for (int i=0; i < nTotalEntities; i+=3) {
    int nodeProc = allEntities[i];
    stk::mesh::EntityId nodeID = allEntities[i+1];
    stk::mesh::EntityId donorID = allEntities[i+2];

    // Add the receptor donor pair to populate OversetInfo
    if (iproc == nodeProc) {
      receptorIDs_.push_back(nodeID);
      donorIDs_.push_back(donorID);
    }

    // Setup for ghosting
    stk::mesh::Entity elem = bulk_.get_entity(stk::topology::ELEM_RANK, donorID);
    if (bulk_.is_valid(elem) &&
        (bulk_.parallel_owner_rank(elem) == iproc) &&
        (nodeProc != iproc)) {
      // Found the owning proc for this donor element. Request ghosting
      stk::mesh::EntityProc elem_proc(elem, nodeProc);
      elemsToGhost_.push_back(elem_proc);
    }
  }
}

void
TiogaSTKIface::populate_overset_info()
{
#ifdef HAS_NALU_WIND
  int nDim = meta_.spatial_dimension();
  int iproc = bulk_.parallel_rank();
  int nproc = bulk_.parallel_size();
  double maxError = -1.0e16;
  std::ofstream outfile;
  auto timeMon = get_timer("TiogaSTKIface::populate_overset_info");

  std::vector<double> elemCoords;

  VectorFieldType *coords = meta_.get_field<VectorFieldType>
    (stk::topology::NODE_RANK, coordsName_);

  size_t numReceptors = receptorIDs_.size();
  for (size_t i=0; i < numReceptors; i++) {
    stk::mesh::EntityId nodeID = receptorIDs_[i];
    stk::mesh::EntityId donorID = donorIDs_[i];
    stk::mesh::Entity node = bulk_.get_entity(stk::topology::NODE_RANK, nodeID);
    stk::mesh::Entity elem = bulk_.get_entity(stk::topology::ELEM_RANK, donorID);

#if 1
    // The donor element must have already been ghosted to the required MPI
    // rank, so validity check should always succeed.
    if (!bulk_.is_valid(elem))
      throw std::runtime_error(
        "Invalid element encountered in overset mesh connectivity");
#endif

    const stk::topology elemTopo = bulk_.bucket(elem).topology();
    sierra::nalu::MasterElement* meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(elemTopo);
    const int nodesPerElem = bulk_.num_nodes(elem);
    std::vector<double> elemxyz(3*nodesPerElem, 0.0);
    std::vector<double> intxyz(3);
    OversetInfo info;
    info.node_ = node;
    info.donorElem_ = elem;

    const double* nxyz = stk::mesh::field_data(*coords, node);
    for (int j=0; j<3; j++) {
      info.nodalCoords_[j] = nxyz[j];
    }

    const stk::mesh::Entity* elem_nodes = bulk_.begin_nodes(elem);
    for (unsigned int in=0; in<bulk_.num_nodes(elem); in++) {
      stk::mesh::Entity enode = elem_nodes[in];
      const double* exyz = stk::mesh::field_data(*coords, enode);
      for (int j=0; j < 3; j++) {
        const int offset = j * nodesPerElem + in;
        elemxyz[offset] = exyz[j];
      }
    }
    const double nearestDistance = meSCS->isInElement(
      elemxyz.data(), info.nodalCoords_.data(), info.isoCoords_.data());
#if 0
    if (nearestDistance > (1.0 + 1.0e-8)) {
        std::cerr
            << "TIOGA WARNING: In pair (" << nodeID << ", " << donorID << "): "
            << "iso-parametric distance is greater than 1.0: " << nearestDistance
            << "; num nodes on element = " << bulk_.num_nodes(elem)
            << std::endl;

        if (!outfile.is_open()) {
            std::string fname = "fringe_mismatch." + std::to_string(nproc)
                + "." + std::to_string(iproc);
            outfile.open(fname, std::ios::out);
        }
        outfile << nodeID << "\t" << donorID << "\t" << nodesPerElem << std::endl;
        for (int j=0; j < 3; j++)
            outfile << info.nodalCoords_[j] << "\t";
        outfile << std::endl;
        for (int j=0; j < 3; j++)
            outfile << info.isoCoords_[j] << "\t";
        outfile << std::endl;
        for (int in=0; in < nodesPerElem; in++) {
            for (int j=0; j < nDim; j++)
                outfile << elemxyz[j * nodesPerElem + in] << "\t";
            outfile << std::endl;
        }
        outfile << std::endl;
    }
#endif
    meSCS->interpolatePoint(3, info.isoCoords_.data(), elemxyz.data(), intxyz.data());

    double error = 0.0;
    for (int i=0; i<3; i++) {
      error += info.nodalCoords_[i] - intxyz[i];
    }
    if (std::fabs(error) > maxError) maxError = error;
  }

#if 1
  stk::parallel_machine_barrier(bulk_.parallel());
  double g_maxError = -1.0e16;
  stk::all_reduce_max(bulk_.parallel(), &maxError, &g_maxError, 1);
  if (iproc == 0)
      std::cout << "Nalu CVFEM interpolation results: max error = " << g_maxError
                << std::endl;
#endif

  if (outfile.is_open()) outfile.close();
#endif
}

void TiogaSTKIface::register_mesh()
{
    auto tmon = get_timer("TiogaSTKIface::register_mesh");
    reset_data_structures();

    // Update the coordinates for TIOGA and register updates to the TIOGA mesh block.
    for (auto& tb: blocks_) {
        tb->update_coords();
        tb->register_block(tg_);
    }
}

void TiogaSTKIface::post_connectivity_work()
{
    auto tmon = get_timer("TiogaSTKIface::post_connectivity_work");
    for (auto& tb : blocks_) {
        // Update IBLANK information at nodes and elements
        tb->update_iblanks();
        tb->update_iblank_cell();
    }

    // Synchronize IBLANK data for shared nodes
    ScalarFieldType* ibf =
        meta_.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "iblank");
    std::vector<const stk::mesh::FieldBase*> pvec{ibf};
    stk::mesh::copy_owned_to_shared(bulk_, pvec);
}

void TiogaSTKIface::register_solution(const int nvars)
{
    auto tmon = get_timer("TiogaSTKIface::register_solution");
    for (auto& tb: blocks_)
        tb->register_solution(tg_, nvars);
}

void TiogaSTKIface::update_solution(const int nvars)
{
    auto tmon = get_timer("TiogaSTKIface::update_solution");
    double maxNorm = -1.0e20;
    double g_maxNorm = -1.0e20;
    for (auto& tb: blocks_)
    {
        double norm = tb->update_solution(nvars);
        maxNorm = std::max(norm, maxNorm);
    }

    stk::all_reduce_max(bulk_.parallel(), &maxNorm, &g_maxNorm, 1);
    if (bulk_.parallel_rank() == 0)
        std::cout << "TIOGA interpolation error (max L2 norm) for STK mesh: "
                  << g_maxNorm << std::endl;
}

}  // tioga
