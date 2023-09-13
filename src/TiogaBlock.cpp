
#include "TiogaBlock.h"
#include "Timer.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpReduceUtils.h"

#include "stk_util/parallel/ParallelReduce.hpp"
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_mesh/base/FieldBLAS.hpp"
#include "stk_mesh/base/GetNgpMesh.hpp"

#include <numeric>
#include <iostream>
#include <cmath>

#include "TiogaMeshInfo.h"
#include "tioga.h"

extern "C" {
double computeCellVolume(double xv[8][3],int nvert);
}

namespace tioga_nalu {
namespace {
template<typename T1, typename T2>
void kokkos_to_tioga_view(T1& lhs, const T2& rhs)
{
    lhs.hptr = rhs.h_view.data();
    lhs.dptr = rhs.d_view.data();
    lhs.sz = rhs.d_view.size();
}

}

TiogaBlock::TiogaBlock(
  stk::mesh::MetaData& meta,
  stk::mesh::BulkData& bulk,
  const YAML::Node& node,
  const std::string& coordsName,
  const int meshtag
) : meta_(meta),
    bulk_(bulk),
    minfo_(new TIOGA::MeshBlockInfo),
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

  if (node["adjust_resolutions"])
    adjust_resolutions_ = node["adjust_resolutions"].as<bool>();
  if (node["cell_res_multiplier"])
      cellResFac_ = node["cell_res_multiplier"].as<bool>();
  if (node["node_res_multiplier"])
      nodeResFac_ = node["node_res_multiplier"].as<bool>();
}

void TiogaBlock::setup()
{
  auto timeMon = get_timer("TiogaBlock::setup");
  names_to_parts(blkNames_, blkParts_);

  if (wallNames_.size() > 0)
    names_to_parts(wallNames_, wallParts_);

  if (ovsetNames_.size() > 0)
    names_to_parts(ovsetNames_, ovsetParts_);

  ScalarFieldType& nodeVol = meta_.declare_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "nodal_volume");

  ScalarFieldType& cellVol = meta_.declare_field<ScalarFieldType>(
      stk::topology::ELEM_RANK, "cell_volume");

  ScalarFieldType& ibf = meta_.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "iblank");

  ScalarFieldType& ibcell = meta_.declare_field<ScalarFieldType>(
    stk::topology::ELEM_RANK, "iblank_cell");

  for (auto p: blkParts_) {
    stk::mesh::put_field_on_mesh(ibf, *p, nullptr);
    stk::mesh::put_field_on_mesh(ibcell, *p, nullptr);
    stk::mesh::put_field_on_mesh(nodeVol, *p, nullptr);
    stk::mesh::put_field_on_mesh(cellVol, *p, nullptr);
  }
}

void TiogaBlock::initialize()
{
  auto timeMon = get_timer("TiogaBlock::initialize");
  process_nodes();
  process_wallbc();
  process_ovsetbc();
  process_elements();

  compute_volumes();
  if (adjust_resolutions_) adjust_cell_resolutions();

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

  compute_volumes();
  if (adjust_resolutions_) adjust_cell_resolutions();
}

void
TiogaBlock::update_iblanks()
{
  ScalarFieldType* ibf =
    meta_.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "iblank");
  auto timeMon = get_timer("TiogaBlock::update_iblanks");

  stk::mesh::Selector mesh_selector = stk::mesh::selectUnion(blkParts_)
      & (meta_.locally_owned_part() | meta_.globally_shared_part());

  using Traits = ngp::NGPMeshTraits<>;
  using CounterType = Kokkos::Sum<ngp::ArrayInt3>;
  typename CounterType::value_type counter;
  CounterType ngp_counter(counter);
  // TODO: move to device view
  bdata_.iblank_.sync_to_device();
  auto& iblarr = bdata_.iblank_.d_view;
  auto& nidmap = bdata_.eid_map_.d_view;
  auto& iblank_ngp = stk::mesh::get_updated_ngp_field<double>(*ibf);
  ngp::run_entity_par_reduce(
      "update_iblanks", stk::mesh::get_updated_ngp_mesh(bulk_),
      //"update_iblanks", bulk_.get_updated_ngp_mesh(),
      stk::topology::NODE_RANK, mesh_selector,
      KOKKOS_LAMBDA(
          const typename Traits::MeshIndex& mi,
          typename CounterType::value_type& pctr) {
          auto node = (*mi.bucket)[mi.bucketOrd];
          const auto idx = nidmap(node.local_offset()) - 1;
          const auto ibval = iblarr(idx);
          iblank_ngp.get(mi, 0) = ibval;

          if(ibval > 0.5) {
              ++pctr.array_[0];
          } else if (ibval < -0.5) {
              ++pctr.array_[1];
          } else {
              ++pctr.array_[2];
          }
      }, ngp_counter);
  ibf->modify_on_device();

  int gcounter[3] = {0, 0, 0};
  stk::all_reduce_sum(bulk_.parallel(), counter.array_, gcounter, 3);
  if (bulk_.parallel_rank() == 0) {
      std::cout << "STK IBLANK " << meshtag_
                << ": field=" << gcounter[0]
                << "; fringe=" << gcounter[1]
                << "; hole=" << gcounter[2]
                << std::endl;
  }
}

void TiogaBlock::update_iblank_cell()
{
  ScalarFieldType* ibf = meta_.get_field<ScalarFieldType>(
    stk::topology::ELEM_RANK, "iblank_cell");
  auto timeMon = get_timer("TiogaBlock::update_iblank_cell");

  stk::mesh::Selector mesh_selector = meta_.locally_owned_part() &
    stk::mesh::selectUnion(blkParts_);

  using Traits = ngp::NGPMeshTraits<>;
  // TODO: move to device view
  bdata_.iblank_cell_.sync_to_device();
  auto& iblarr = bdata_.iblank_cell_.d_view;
  auto& eidmap = bdata_.eid_map_.d_view;
  auto& iblank_ngp = stk::mesh::get_updated_ngp_field<double>(*ibf);
  ngp::run_entity_algorithm(
      "update_iblanks", stk::mesh::get_updated_ngp_mesh(bulk_),
      //"update_iblanks", bulk_.get_updated_ngp_mesh(),
      stk::topology::ELEM_RANK, mesh_selector,
      KOKKOS_LAMBDA(const typename Traits::MeshIndex& mi) {
          auto elem = (*mi.bucket)[mi.bucketOrd];
          const auto idx = eidmap(elem.local_offset()) - 1;
          iblank_ngp.get(mi, 0) = iblarr(idx);
      });
  ibf->modify_on_device();
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
    bdata_.eid_map_.init("stk_to_tioga_id", bulk_.get_size_of_entity_index_space());
    bdata_.node_gid_.init("node_gid", num_nodes_);
  }

  auto& ngp_xyz = bdata_.xyz_.h_view;
  auto& nidmap = bdata_.eid_map_.h_view;
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
  bdata_.eid_map_.sync_to_device();
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
  auto& nidmap = bdata_.eid_map_.h_view;
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
  auto& nidmap = bdata_.eid_map_.h_view;
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

  if (tioga_conn_)
    delete[] tioga_conn_;
  tioga_conn_ = new int*[ntypes];

  std::map<int, int> conn_ids;        // Topo -> array index lookup table
  std::map<int, size_t> conn_offsets; // Topo -> array offset lookup table

  // 3. Populate TIOGA data structures
  int idx = 0;
  int tot_elems = 0;
  for (auto kv: conn_map_) {
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
  auto& eidmap = bdata_.eid_map_.h_view;
  auto& cellgid = bdata_.cell_gid_.h_view;
  for (auto b: mbkts) {
    const int npe = b->num_nodes(0);
    const int idx = conn_ids[npe];
    int offset = conn_offsets[npe];
    for (size_t in=0; in < b->size(); in++) {
      const stk::mesh::Entity elem = (*b)[in];
      const stk::mesh::EntityId eid = bulk_.identifier(elem);
      eidmap(elem.local_offset()) = ep + 1;
      cellgid(ep++) = eid;
      const stk::mesh::Entity* enodes = b->begin_nodes(in);
      for (int i=0; i < npe; i++) {
        const stk::mesh::EntityId nid = bulk_.identifier(enodes[i]);
        bdata_.connect_[idx].h_view(offset++) = eidmap(enodes[i].local_offset());
      }
    }
    conn_offsets[npe] = offset;
  }

  // TIOGA expects a ptr-to-ptr data structure for connectivity
  for(size_t i=0; i<ntypes; i++) {
    tioga_conn_[i] = bdata_.connect_[i].h_view.data();
    bdata_.connect_[i].sync_to_device();
  }

  bdata_.eid_map_.sync_to_device();
  bdata_.cell_gid_.sync_to_device();
  bdata_.num_verts_.sync_to_device();
  bdata_.num_cells_.sync_to_device();
  Kokkos::deep_copy(bdata_.iblank_cell_.h_view, 1);
  Kokkos::deep_copy(bdata_.iblank_cell_.d_view, 1);
}

void TiogaBlock::compute_volumes()
{
  double xv[8][3];
  int inode[8];

  auto& xyz = bdata_.xyz_.h_view;
  auto& cellres = bdata_.cell_res_.h_view;
  auto& noderes = bdata_.node_res_.h_view;
  auto& nodegid = bdata_.node_gid_.h_view;
  auto& cellgid = bdata_.cell_gid_.h_view;
  auto& nv = bdata_.num_verts_.h_view;
  auto& nc = bdata_.num_cells_.h_view;
  auto connect = bdata_.connect_;

  auto* nodal_vol = meta_.get_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "nodal_volume");
  auto* cell_vol = meta_.get_field<ScalarFieldType>(
      stk::topology::ELEM_RANK, "cell_volume");
  stk::mesh::field_fill(0.0, *nodal_vol);

  const size_t ntypes = nv.size();
  int k = 0;
  for (size_t n=0; n < ntypes; ++n) {
    const int nvert = nv[n];
    const auto& vconn = connect[n].h_view;

    for (int i=0; i < nc[n]; ++i) {
      for (int m=0; m < nvert; ++m) {
        inode[m] = vconn[nvert * i + m] - 1;
        const int i3 = 3 * inode[m];
        for (int j=0; j < 3; ++j) xv[m][j] = xyz[i3 + j];
      }

      const double vol = computeCellVolume(xv, nvert);
      {
        const auto eid = cellgid[k];
        const auto elem = bulk_.get_entity(stk::topology::ELEM_RANK, eid);
        double* cVol = stk::mesh::field_data(*cell_vol, elem);
        cVol[0] = vol * cellResFac_;
      }
      cellres[k++] = vol * cellResFac_;

      for (int m=0; m < nvert; ++m) {
        const auto nid = nodegid[inode[m]];
        const auto node = bulk_.get_entity(stk::topology::NODE_RANK, nid);
        const auto nelems = bulk_.num_elements(node);
        double* dVol = stk::mesh::field_data(*nodal_vol, node);
        dVol[0] += (vol * nodeResFac_) / static_cast<double>(nelems);
      }
    }
  }

  // Update nodal volumes across processor boundaries
  stk::mesh::parallel_sum(bulk_, {nodal_vol});

  stk::mesh::Selector sel = stk::mesh::selectUnion(blkParts_)
      & (meta_.locally_owned_part() | meta_.globally_shared_part());
  const auto& mbkts = bulk_.get_buckets(
      stk::topology::NODE_RANK, sel);

  int ip = 0;
  for (auto b: mbkts) {
    double* dVol = stk::mesh::field_data(*nodal_vol, *b);
    for (size_t in=0; in < b->size(); ++in) {
      noderes[ip++] = dVol[in];
    }
  }

  bdata_.cell_res_.sync_to_device();
  bdata_.node_res_.sync_to_device();
}

void TiogaBlock::adjust_node_resolutions()
{
  // Only perform this step if user requested adjust resolutions
  if (!adjust_resolutions_) return;

  constexpr double large_volume = std::numeric_limits<double>::max();
  constexpr double lvol1 = 1.0e15;
  stk::mesh::Selector sel = stk::mesh::selectUnion(blkParts_)
      & (meta_.locally_owned_part() | meta_.globally_shared_part());
  const stk::mesh::BucketVector& mbkts = bulk_.get_buckets(
      stk::topology::NODE_RANK, sel);
  auto* nodal_vol = meta_.get_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "nodal_volume");

  auto& eidmap = bdata_.eid_map_.h_view;
  auto& noderes = bdata_.node_res_.h_view;
  for (auto b: mbkts) {
    double* dVol = stk::mesh::field_data(*nodal_vol, *b);
    for (size_t in = 0; in < b->size(); in++) {
      const auto node = (*b)[in];
      const int nidx = eidmap(node.local_offset()) - 1;
      const double nodevol = dVol[in];
      noderes[nidx] = (nodevol > lvol1) ? large_volume : nodevol;
    }
  }

  bdata_.node_res_.sync_to_device();
}

void TiogaBlock::adjust_cell_resolutions()
{
  // For every face on the sideset, grab the connected element and set its
  // cell resolution to a large value. Also for each node of that element, set
  // the nodal resolution to a large value.

  constexpr double large_volume = std::numeric_limits<double>::max();
  // Paraview freaks out if we set large volume in the field
  constexpr double lvol1 = 1.0e16;
  stk::mesh::Selector sel = stk::mesh::selectUnion(ovsetParts_)
      & (meta_.locally_owned_part() | meta_.globally_shared_part());
  const stk::mesh::BucketVector& mbkts = bulk_.get_buckets(
    meta_.side_rank(), sel);
  auto* nodal_vol = meta_.get_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "nodal_volume");
  auto* cell_vol = meta_.get_field<ScalarFieldType>(
      stk::topology::ELEM_RANK, "cell_volume");

  size_t counter[2] = {0, 0};

  auto& eidmap = bdata_.eid_map_.h_view;
  auto& cellres = bdata_.cell_res_.h_view;
  for (auto b: mbkts) {
    for (size_t fi=0; fi < b->size(); ++fi) {
      const auto face = (*b)[fi];
      const auto* elems = bulk_.begin_elements(face);
      const auto num_elems = bulk_.num_elements(face);

      for (unsigned ie=0; ie < num_elems; ++ie) {
        const auto elem = elems[ie];
        const int eidx = eidmap(elem.local_offset()) - 1;
        cellres[eidx] = large_volume;
        double* cVol = stk::mesh::field_data(*cell_vol, elem);
        cVol[0] = lvol1;
        ++counter[0];

        const auto* nodes = bulk_.begin_nodes(elem);
        const auto num_nodes = bulk_.num_nodes(elem);

        for (unsigned in=0; in < num_nodes; ++in) {
          const auto node = nodes[in];
          double* dVol = stk::mesh::field_data(*nodal_vol, node);
          dVol[0] = lvol1;
          ++counter[1];
        }
      }
    }
  }

  size_t gcounter[2] = {0, 0};
  stk::all_reduce_sum(bulk_.parallel(), counter, gcounter, 2);
  if (bulk_.parallel_rank() == 0) {
      std::cout << "Set resolutions for mandatory fringes:\n    "
                << "Mesh ID = " << meshtag_
                << " Elements = " << gcounter[0]
                << " Nodes = " << gcounter[1]
                << std::endl;
  }

  bdata_.cell_res_.sync_to_device();
}

void TiogaBlock::register_block(TIOGA::tioga& tg, const bool use_ngp_iface)
{
  if (use_ngp_iface)
    register_block_ngp(tg);
  else
    register_block_classic(tg);
}

void TiogaBlock::register_block_ngp(TIOGA::tioga& tg)
{
  if (num_nodes_ < 1) return;
  auto timeMon = get_timer("TiogaBlock::register_block");

  Kokkos::deep_copy(bdata_.iblank_.h_view, 1.0);
  Kokkos::deep_copy(bdata_.iblank_cell_.h_view, 1.0);
  Kokkos::deep_copy(bdata_.iblank_.d_view, 1.0);
  Kokkos::deep_copy(bdata_.iblank_cell_.d_view, 1.0);

  block_info_to_tioga();
  tg.register_unstructured_grid(minfo_.get());
}

void TiogaBlock::register_block_classic(TIOGA::tioga& tg)
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
  tg.setResolutions(
      meshtag_, bdata_.node_res_.h_view.data(), bdata_.cell_res_.h_view.data());
}

void TiogaBlock::register_solution_old(TIOGA::tioga& tg)
{
  if (num_nodes_ < 1) return;
  auto timeMon = get_timer("TiogaBlock::register_solution");

  auto& qsol = bdata_.qsol_;
  if (qsol.size() != num_nodes_) qsol.init("stk_soln_array", num_nodes_);

  auto& qsolarr = qsol.h_view;
  auto& xyz = bdata_.xyz_.h_view;
  for (int i=0, ii=0; i<num_nodes_; i++, ii+=3) {
      qsolarr(i) = xyz[ii] + xyz[ii+1] + xyz[ii+2];
  }

  tg.registerSolution(meshtag_, qsolarr.data());
}

double TiogaBlock::calculate_residuals_old()
{
  double rnorm = 0.0;

  // Skip block if this is not shared by the proc
  if (num_nodes_ < 1) return rnorm;
  auto timeMon = get_timer("TiogaBlock::calculate_residuals");

  auto& qsol = bdata_.qsol_.h_view;
  auto& xyz = bdata_.xyz_.h_view;
  for (int i=0, ii=0; i < num_nodes_; i++, ii+=3) {
      double diff = qsol(i) - (xyz[ii] + xyz[ii+1] + xyz[ii+2]);
      rnorm += diff * diff;
  }

  rnorm /= num_nodes_;
  return std::sqrt(rnorm);
}

void TiogaBlock::register_solution(TIOGA::tioga& tg, const int nvars, const bool use_ngp_iface)
{
    if (num_nodes_ < 1) return;
    auto tmon = get_timer("TiogaBlock::register_solution");

    const size_t qsol_size = num_nodes_ * nvars;
    auto& qsol = bdata_.qsol_;
    if (qsol.size() != qsol_size) qsol.init("stk_soln_array", qsol_size);
    auto& qsolarr = qsol.h_view;

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
                qsolarr(ip++) = qq[i];
        }
    }

    if (use_ngp_iface) {
        bdata_.qsol_.sync_to_device();
        minfo_->num_vars = nvars;
        kokkos_to_tioga_view(minfo_->qnode, bdata_.qsol_);
        tg.register_unstructured_solution();
    }
    else
        tg.register_unstructured_solution(meshtag_, qsolarr.data(),nvars,0);
}

double TiogaBlock::update_solution(const int nvars)
{
    double rnorm = 0.0;
    if (num_nodes_ < 1) return rnorm;
    auto tmon = get_timer("TiogaBlock::update_solution");

    // TODO: sync to host after uppdating TIOGA to update solution on device
    // bdata_.qsol_.sync_to_host();

    auto& qsolarr = bdata_.qsol_.h_view;
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
                const double diff = qsolarr(ip) - qq[i];
                rnorm += diff * diff;
                qq[i] = qsolarr(ip++);
            }
        }
    }

    rnorm /= static_cast<double>(num_nodes_ * nvars);
    return std::sqrt(rnorm);
}

void TiogaBlock::block_info_to_tioga()
{
    // Note solution registration is handled separately. We don't process that here.
    static_assert(NgpTiogaBlock::max_vertex_types == TIOGA::MeshBlockInfo::max_vertex_types,
                  "Invalid vertex types encountered");
    auto& mi = *minfo_;
    auto& bd = bdata_;

    mi.meshtag = meshtag_;
    mi.num_nodes = num_nodes_;
    mi.qtype = TIOGA::MeshBlockInfo::ROW;

    // Array dimensions
    kokkos_to_tioga_view(mi.wall_ids, bd.wallIDs_);
    kokkos_to_tioga_view(mi.overset_ids, bd.ovsetIDs_);
    kokkos_to_tioga_view(mi.num_vert_per_elem, bd.num_verts_);
    kokkos_to_tioga_view(mi.num_cells_per_elem, bd.num_cells_);

    kokkos_to_tioga_view(mi.xyz, bd.xyz_);
    kokkos_to_tioga_view(mi.iblank_node, bd.iblank_);
    kokkos_to_tioga_view(mi.iblank_cell, bd.iblank_cell_);
    kokkos_to_tioga_view(mi.node_gid, bd.node_gid_);
    kokkos_to_tioga_view(mi.cell_gid, bd.cell_gid_);
    kokkos_to_tioga_view(mi.node_res, bd.node_res_);
    kokkos_to_tioga_view(mi.cell_res, bd.cell_res_);

    for (int i=0; i < TIOGA::MeshBlockInfo::max_vertex_types; ++i) {
        kokkos_to_tioga_view(mi.vertex_conn[i], bd.connect_[i]);
    }
}


} // namespace tioga
