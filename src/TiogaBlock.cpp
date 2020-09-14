
#include "TiogaBlock.h"
#include "Timer.h"

#include <numeric>
#include <iostream>
#include <cmath>

#include "tioga.h"

namespace tioga_nalu {

TiogaBlock::TiogaBlock(
  stk::mesh::MetaData& meta,
  stk::mesh::BulkData& bulk,
  const YAML::Node& node,
  const std::string& coordsName,
  const int meshtag
) : meta_(meta),
    bulk_(bulk),
    coordsName_(coordsName),
    ndim_(meta_.spatial_dimension()),
    meshtag_(meshtag)
{
  load(node);
}

TiogaBlock::~TiogaBlock()
{
  if (tioga_conn_ != nullptr) {
    delete[] tioga_conn_;
  }
}

void TiogaBlock::load(const YAML::Node& node)
{
  blkNames_ = node["mesh_parts"].as<std::vector<std::string>>();

  if (node["wall_parts"]) {
    wallNames_ = node["wall_parts"].as<std::vector<std::string>>();
  }

  if (node["ovset_parts"]) {
    ovsetNames_ = node["ovset_parts"].as<std::vector<std::string>>();
  }
}

void TiogaBlock::setup()
{
  auto timeMon = get_timer("TiogaBlock::setup");
  names_to_parts(blkNames_, blkParts_);

  if (wallNames_.size() > 0)
    names_to_parts(wallNames_, wallParts_);

  if (ovsetNames_.size() > 0)
    names_to_parts(ovsetNames_, ovsetParts_);

  ScalarFieldType& ibf = meta_.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "iblank");

  ScalarFieldType& ibcell = meta_.declare_field<ScalarFieldType>(
    stk::topology::ELEM_RANK, "iblank_cell");

  for (auto p: blkParts_) {
    stk::mesh::put_field_on_mesh(ibf, *p, nullptr);
    stk::mesh::put_field_on_mesh(ibcell, *p, nullptr);
  }
}

void TiogaBlock::initialize()
{
  auto timeMon = get_timer("TiogaBlock::initialize");
  process_nodes();
  process_wallbc();
  process_ovsetbc();
  process_elements();

  is_init_ = false;
}

void TiogaBlock::update_coords()
{
  auto timeMon = get_timer("TiogaBlock::update_coords");
  stk::mesh::Selector mesh_selector = stk::mesh::selectUnion(blkParts_)
      & (meta_.locally_owned_part() | meta_.globally_shared_part());
  const stk::mesh::BucketVector& mbkts = bulk_.get_buckets(
    stk::topology::NODE_RANK, mesh_selector);
  VectorFieldType* coords = meta_.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, coordsName_);

  auto& ngp_xyz = bdata_.xyz_.h_view;
  int ip = 0;
  for (auto b: mbkts) {
    for (size_t in=0; in < b->size(); in++) {
      stk::mesh::Entity node = (*b)[in];

      double* pt = stk::mesh::field_data(*coords, node);
      for (int i=0; i < ndim_; i++) {
        ngp_xyz(ip * ndim_ + i) = pt[i];
      }
      ip++;
    }
  }

  bdata_.xyz_.sync_to_device();
}

void
TiogaBlock::update_connectivity()
{
  process_nodes();
  process_wallbc();
  process_ovsetbc();
  process_elements();
}

void
TiogaBlock::update_iblanks()
{
  ScalarFieldType* ibf =
    meta_.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "iblank");
  auto timeMon = get_timer("TiogaBlock::update_iblanks");

  stk::mesh::Selector mesh_selector = stk::mesh::selectUnion(blkParts_)
      & (meta_.locally_owned_part() | meta_.globally_shared_part());
  const stk::mesh::BucketVector& mbkts =
    bulk_.get_buckets(stk::topology::NODE_RANK, mesh_selector);

  auto& iblank = bdata_.iblank_.h_view;
  int ip = 0;
  for (auto b : mbkts) {
    double* ib = stk::mesh::field_data(*ibf, *b);
    for (size_t in = 0; in < b->size(); in++) {
      // stk::mesh::Entity node = (*b)[in];
      // stk::mesh::EntityId nid = bulk_.identifier(node);
      ib[in] = iblank(ip++);
    }
  }
}

void TiogaBlock::update_iblank_cell()
{
  ScalarFieldType* ibf = meta_.get_field<ScalarFieldType>(
    stk::topology::ELEM_RANK, "iblank_cell");
  auto timeMon = get_timer("TiogaBlock::update_iblank_cell");

  stk::mesh::Selector mesh_selector = meta_.locally_owned_part() &
    stk::mesh::selectUnion(blkParts_);
  const stk::mesh::BucketVector& mbkts = bulk_.get_buckets(
    stk::topology::ELEM_RANK, mesh_selector);

  auto& iblank_cell = bdata_.iblank_cell_.h_view;
  int ip = 0;
  for (auto b: mbkts) {
    double* ib = stk::mesh::field_data(*ibf, *b);
    for(size_t in=0; in < b->size(); in++) {
      ib[in] = iblank_cell(ip++);
    }
  }
}

void TiogaBlock::get_donor_info(TIOGA::tioga& tg, stk::mesh::EntityProcVec& egvec)
{
  // Nothing to do if we haven't registered this mesh on this proc
  if (num_nodes_ < 1) return;
  auto timeMon = get_timer("TiogaBlock::get_donor_info");

  int dcount, fcount;

  // Call TIOGA API to determine donor info array sizes
  {
      auto timeMon1 = get_timer("TIOGA::getDonorCount");
      tg.getDonorCount(meshtag_, &dcount, &fcount);
  }

  // Receptor info: rProcID, rNodeID, blkID, nFractions
  std::vector<int> receptorInfo(dcount*4);
  // Node index information (the last entry is the donor element ID)
  std::vector<int> inode(fcount);
  // fractions (ignored for now)
  std::vector<double> frac(fcount);

  // Populate the donor information arrays through TIOGA API call
  {
      auto timeMon1 = get_timer("TIOGA::getDonorInfo");
      tg.getDonorInfo(meshtag_,receptorInfo.data(),inode.data(),
                      frac.data(),&dcount);
  }

  int myRank = bulk_.parallel_rank();
  int idx = 0;
  for(int i=0; i<(4*dcount); i += 4) {
    int procid = receptorInfo[i];
    int nweights = receptorInfo[i+3];           // Offset to get the donor element
    int elemid_tmp = inode[idx + nweights]; // Local index for lookup
    auto elemID = bdata_.cell_gid_.h_view[elemid_tmp];       // Global ID of element

    // Move the offset index for next call
    idx += nweights + 1;

    // No ghosting necessary if sharing the same rank
    if (procid == myRank) continue;

    stk::mesh::Entity elem = bulk_.get_entity(stk::topology::ELEM_RANK, elemID);
    stk::mesh::EntityProc elem_proc(elem, procid);
    egvec.push_back(elem_proc);
  }
}

inline void TiogaBlock::names_to_parts(
  const std::vector<std::string>& pnames,
  stk::mesh::PartVector& parts)
{
  parts.resize(pnames.size());
  for(size_t i=0; i < pnames.size(); i++) {
    stk::mesh::Part* p = meta_.get_part(pnames[i]);
    if (nullptr == p) {
      throw std::runtime_error("TiogaBlock: cannot find part named: " + pnames[i]);
    } else {
      parts[i] = p;
    }
  }
}

void TiogaBlock::process_nodes()
{
  stk::mesh::Selector mesh_selector = stk::mesh::selectUnion(blkParts_)
      & (meta_.locally_owned_part() | meta_.globally_shared_part());
  const stk::mesh::BucketVector& mbkts = bulk_.get_buckets(
    stk::topology::NODE_RANK, mesh_selector);
  VectorFieldType* coords = meta_.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, coordsName_);

  int ncount = 0;
  for (auto b: mbkts) ncount += b->size();

  if (is_init_ || ncount != num_nodes_) {
    num_nodes_ = ncount;

    bdata_.xyz_.init("xyz", ndim_ * num_nodes_);
    bdata_.iblank_.init("iblank_node", num_nodes_);
    bdata_.node_res_.init("node_res", num_nodes_);
    bdata_.node_map_.init("stk_to_tioga_id", bulk_.get_size_of_entity_index_space());
    bdata_.node_gid_.init("node_gid", num_nodes_);
  }

  auto& ngp_xyz = bdata_.xyz_.h_view;
  auto& nidmap = bdata_.node_map_.h_view;
  auto& nodegid = bdata_.node_gid_.h_view;
  int ip =0; // Index into the xyz_ array
  for (auto b: mbkts) {
    for (size_t in=0; in < b->size(); in++) {
      stk::mesh::Entity node = (*b)[in];
      stk::mesh::EntityId nid = bulk_.identifier(node);

      double* pt = stk::mesh::field_data(*coords, node);
      for (int i=0; i < ndim_; i++) {
        ngp_xyz(ip * ndim_ + i) = pt[i];
      }
      nidmap(node.local_offset()) = ip + 1; // TIOGA uses 1-based indexing
      nodegid(ip) = nid;
      ip++;
    }
  }

  bdata_.xyz_.sync_to_device();
  bdata_.node_map_.sync_to_device();
  bdata_.node_gid_.sync_to_device();
  Kokkos::deep_copy(bdata_.iblank_.h_view, 1);
  Kokkos::deep_copy(bdata_.iblank_.d_view, 1);
}

void TiogaBlock::process_wallbc()
{
  stk::mesh::Selector mesh_selector = stk::mesh::selectUnion(wallParts_)
      & (meta_.locally_owned_part() | meta_.globally_shared_part());
  const stk::mesh::BucketVector& mbkts = bulk_.get_buckets(
    stk::topology::NODE_RANK, mesh_selector);

  int ncount = 0;
  for (auto b: mbkts) ncount += b->size();

  if (is_init_ || (ncount != num_wallbc_)) {
    num_wallbc_ = ncount;
    bdata_.wallIDs_.init("wall_ids", num_wallbc_);
  }

  auto& wallids = bdata_.wallIDs_.h_view;
  auto& nidmap = bdata_.node_map_.h_view;
  int ip = 0; // Index into the wallIDs array
  for (auto b: mbkts) {
    for (size_t in=0; in < b->size(); in++) {
      stk::mesh::Entity node = (*b)[in];
      wallids(ip++) = nidmap(node.local_offset());
    }
  }
  bdata_.wallIDs_.sync_to_device();
}

void TiogaBlock::process_ovsetbc()
{
  stk::mesh::Selector mesh_selector = stk::mesh::selectUnion(ovsetParts_)
      & (meta_.locally_owned_part() | meta_.globally_shared_part());
  const stk::mesh::BucketVector& mbkts = bulk_.get_buckets(
    stk::topology::NODE_RANK, mesh_selector);

  int ncount = 0;
  for (auto b: mbkts) ncount += b->size();

  if (is_init_ || (ncount != num_ovsetbc_)) {
    num_ovsetbc_ = ncount;
    bdata_.ovsetIDs_.init("overset_ids", num_ovsetbc_);
  }

  auto& ovsetids = bdata_.ovsetIDs_.h_view;
  auto& nidmap = bdata_.node_map_.h_view;
  int ip = 0; // Index into ovsetIDs array
  for (auto b: mbkts) {
    for (size_t in=0; in < b->size(); in++) {
      stk::mesh::Entity node = (*b)[in];
      ovsetids(ip++) = nidmap(node.local_offset());
    }
  }
  bdata_.ovsetIDs_.sync_to_device();
}

void TiogaBlock::process_elements()
{
  stk::mesh::Selector mesh_selector = meta_.locally_owned_part() &
    stk::mesh::selectUnion(blkParts_);
  const stk::mesh::BucketVector& mbkts = bulk_.get_buckets(
    stk::topology::ELEM_RANK, mesh_selector);

  // 1. Determine the number of topologies present in this mesh block. For
  // each topology determine the number of elements associated with it (across
  // all buckets). We will use this for resizing arrays later on.
  for(auto b: mbkts) {
    size_t num_elems = b->size();
    // npe = Nodes Per Elem
    int npe = b->topology().num_nodes();
    auto topo = conn_map_.find(npe);
    if (topo != conn_map_.end()) {
      conn_map_[npe] += num_elems;
    } else {
      conn_map_[npe] = num_elems;
    }
  }

  // 2. Resize arrays used to pass data to TIOGA grid registration interface
  auto ntypes = conn_map_.size();
  bdata_.num_verts_.init("num_verts_per_etype", ntypes);
  bdata_.num_cells_.init("num_cells_per_etype", ntypes);

  connect_.resize(ntypes);
  if (tioga_conn_)
    delete[] tioga_conn_;
  tioga_conn_ = new int*[ntypes];

  std::map<int, int> conn_ids;        // Topo -> array index lookup table
  std::map<int, size_t> conn_offsets; // Topo -> array offset lookup table

  // 3. Populate TIOGA data structures
  int idx = 0;
  int cres_count = 0;
  int tot_elems = 0;
  for (auto kv: conn_map_) {
    connect_[idx].resize(kv.first * kv.second);
    tot_elems += kv.second;
    {
        bdata_.num_verts_.h_view[idx] = kv.first;
        bdata_.num_cells_.h_view[idx] = kv.second;
        bdata_.connect_[idx].init("cell_" + std::to_string(kv.first), kv.first * kv.second);
    }
    conn_ids[kv.first] = idx;
    conn_offsets[kv.first] = 0;
    idx++;
  }

  bdata_.iblank_cell_.init("iblank_cell", tot_elems);
  bdata_.cell_res_.init("cell_res", tot_elems);
  bdata_.cell_gid_.init("cell_gid", tot_elems);

  // 4. Create connectivity map based on local node index (xyz_)
  int ep = 0;
  auto& nidmap = bdata_.node_map_.h_view;
  auto& cellgid = bdata_.cell_gid_.h_view;
  for (auto b: mbkts) {
    const int npe = b->num_nodes(0);
    const int idx = conn_ids[npe];
    int offset = conn_offsets[npe];
    for (size_t in=0; in < b->size(); in++) {
      const stk::mesh::Entity elem = (*b)[in];
      const stk::mesh::EntityId eid = bulk_.identifier(elem);
      cellgid(ep++) = eid;
      const stk::mesh::Entity* enodes = b->begin_nodes(in);
      for (int i=0; i < npe; i++) {
        const stk::mesh::EntityId nid = bulk_.identifier(enodes[i]);
        bdata_.connect_[idx].h_view(offset) = nidmap(enodes[i].local_offset());
        connect_[idx][offset++] = nidmap(enodes[i].local_offset());
      }
    }
    conn_offsets[npe] = offset;
  }

  // TIOGA expects a ptr-to-ptr data structure for connectivity
  for(size_t i=0; i<ntypes; i++) {
    tioga_conn_[i] = connect_[i].data();
  }

  bdata_.cell_gid_.sync_to_device();
  Kokkos::deep_copy(bdata_.iblank_cell_.h_view, 1);
  Kokkos::deep_copy(bdata_.iblank_cell_.d_view, 1);
}

void TiogaBlock::register_block(TIOGA::tioga& tg)
{
  // Do nothing if this mesh block isn't present in this MPI Rank
  if (num_nodes_ < 1) return;
  auto timeMon = get_timer("TiogaBlock::register_block");

  Kokkos::deep_copy(bdata_.iblank_.h_view, 1.0);
  Kokkos::deep_copy(bdata_.iblank_cell_.h_view, 1.0);

  // Register the mesh block information to TIOGA
  tg.registerGridData(
    meshtag_,                        // Unique body tag
    num_nodes_,                      // Number of nodes in this mesh block
    bdata_.xyz_.h_view.data(),       // Nodal coordinates
    bdata_.iblank_.h_view.data(),    // iblank array corresponding to nodes
    num_wallbc_,                     // Number of Wall BC nodes
    num_ovsetbc_,                    // Number of overset BC nodes
    bdata_.wallIDs_.h_view.data(),   // Node IDs of wall BC nodes
    bdata_.ovsetIDs_.h_view.data(),  // Node IDs of overset BC nodes
    bdata_.num_verts_.h_view.size(), // Number of topologies in this mesh block
    bdata_.num_verts_.h_view.data(), // Number of vertices per topology
    bdata_.num_cells_.h_view.data(), // Number of cells for each topology
    tioga_conn_,                     // Element node connectivity information
    bdata_.cell_gid_.h_view.data(),  // Global ID for the element array
    bdata_.node_gid_.h_view.data()   // Global ID for the node array
  );
  // Indicate that we want element IBLANK information returned
  tg.set_cell_iblank(meshtag_, bdata_.iblank_cell_.h_view.data());
  //tg.setResolutions(meshtag_, node_res_.data(), cell_res_.data());
}

void TiogaBlock::register_solution_old(TIOGA::tioga& tg)
{
  if (num_nodes_ < 1) return;
  auto timeMon = get_timer("TiogaBlock::register_solution");

  qsol_.resize(num_nodes_);

  auto& xyz = bdata_.xyz_.h_view;
  for (int i=0, ii=0; i<num_nodes_; i++, ii+=3) {
    qsol_[i] = xyz[ii] + xyz[ii+1] + xyz[ii+2];
  }

  tg.registerSolution(meshtag_, qsol_.data());
}

double TiogaBlock::calculate_residuals_old()
{
  double rnorm = 0.0;

  // Skip block if this is not shared by the proc
  if (num_nodes_ < 1) return rnorm;
  auto timeMon = get_timer("TiogaBlock::calculate_residuals");

  auto& xyz = bdata_.xyz_.h_view;
  for (int i=0, ii=0; i < num_nodes_; i++, ii+=3) {
    double diff = qsol_[i] - (xyz[ii] + xyz[ii+1] + xyz[ii+2]);
    rnorm += diff * diff;
  }

  rnorm /= num_nodes_;
  return std::sqrt(rnorm);
}

void TiogaBlock::register_solution(TIOGA::tioga& tg, const int nvars)
{
    if (num_nodes_ < 1) return;
    auto tmon = get_timer("TiogaBlock::register_solution");

    qsol_.resize(num_nodes_ * nvars);

    auto* qvars = meta_.get_field<GenericFieldType>(
        stk::topology::NODE_RANK, "qvars");
    stk::mesh::Selector sel = stk::mesh::selectUnion(blkParts_)
        & (meta_.locally_owned_part() | meta_.globally_shared_part());
    const auto& bkts = bulk_.get_buckets(
        stk::topology::NODE_RANK, sel);

    int ip = 0;
    for (auto b: bkts) {
        for (size_t in=0; in < b->size(); ++in) {
            const auto node = (*b)[in];
            double* qq = stk::mesh::field_data(*qvars, node);
            for (int i=0; i < nvars; ++i)
                qsol_[ip++] = qq[i];
        }
    }

    tg.register_unstructured_solution(meshtag_, qsol_.data(),nvars,0);
}

double TiogaBlock::update_solution(const int nvars)
{
    double rnorm = 0.0;
    if (num_nodes_ < 1) return rnorm;
    auto tmon = get_timer("TiogaBlock::update_solution");

    auto* qvars = meta_.get_field<GenericFieldType>(
        stk::topology::NODE_RANK, "qvars");
    stk::mesh::Selector sel = stk::mesh::selectUnion(blkParts_)
        & (meta_.locally_owned_part() | meta_.globally_shared_part());
    const auto& bkts = bulk_.get_buckets(
        stk::topology::NODE_RANK, sel);

    int ip = 0;
    for (auto b: bkts) {
        for (size_t in=0; in < b->size(); ++in) {
            const auto node = (*b)[in];
            double* qq = stk::mesh::field_data(*qvars, node);
            for (int i=0; i < nvars; ++i) {
                const double diff = qsol_[ip] - qq[i];
                rnorm += diff * diff;
                qq[i] = qsol_[ip++];
            }
        }
    }

    rnorm /= static_cast<double>(num_nodes_ * nvars);
    return std::sqrt(rnorm);
}


} // namespace tioga
