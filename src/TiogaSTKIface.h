#ifndef TIOGASTKIFACE_H
#define TIOGASTKIFACE_H

#include "TiogaBlock.h"

#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>

#include "yaml-cpp/yaml.h"

#include <vector>
#include <memory>
#include <array>

namespace TIOGA {
    class tioga;
}

namespace tioga_nalu {

struct OversetInfo
{
  //! Fringe node point
  stk::mesh::Entity node_;

  //! Donor element from other mesh
  stk::mesh::Entity donorElem_;

  std::array<double,3> isoCoords_;
  std::array<double,3> nodalCoords_;
};

/** Nalu interface to TIOGA (Topology Independent Overset Grid Assembly)
 *
 *  This class provides a two-way data transfer interface for TIOGA library and
 *  provides overset connectivity capability for Nalu.
 */
class TiogaSTKIface
{
public:
  /**
   *  @param meta STK MetaData
   *  @param bulk STK BulkData
   *  @param node YAML node containing overset inputs
   */
  TiogaSTKIface(stk::mesh::MetaData&,
                stk::mesh::BulkData&,
                const YAML::Node&,
                const std::string&);

  ~TiogaSTKIface();

  /** Setup block structure information (steps before mesh creation)
   */
  void setup();

  /** Initialize mesh data structure (steps after mesh creation)
   */
  void initialize();

  /** Determine overset connectivity by calling into TIOGA API
   *
   *  This method performs several steps: updates coordinates (if necessary,
   *  during mesh motion), registers the mesh blocks to TIOGA, calculate mesh
   *  connectivity information (hole, fringe, and field point determination),
   *  update the "overset inactive part" for hole elements, create the {fringe
   *  node, donor element} mapping pair data structures for overset simulations.
   */
  void execute();

  /** Check interpolation errors from overset on a linear field function
   *
   */
  void check_soln_norm();

  /** Register mesh to TIOGA
   */
  void register_mesh();

  /** Perform post-connectivity updates
   */
  void post_connectivity_work();

  //! Register solution arrays to TIOGA
  void register_solution(const int);

  //! Update solution field
  void update_solution(const int);

  /** Return the TIOGA interface object */
  TIOGA::tioga& tioga_iface()
  { return tg_; }

private:
  TiogaSTKIface() = delete;
  TiogaSTKIface(const TiogaSTKIface&) = delete;

  /** Process the input parameters and initialize all data structures necessary
   * to call TIOGA.
   */
  void load(const YAML::Node&);

  /** Ghost donor elements to receptor MPI ranks
   */
  void update_ghosting();

  /** Reset all connectivity data structures when recomputing connectivity
   */
  void reset_data_structures();

  /** Determine (receptor, donor) pairs on the MPI rank containing receptors.
   *
   *  Populates the list of donor elements for each receptor on this MPI rank.
   *  Creation of the actual data structures is done in populate_overset_info.
   *  The method is also responsible for determining fringe/field mismatches for
   *  the shared nodes across processor interfaces. The logic used is to signal
   *  all procs sharing the node to take on a fringe status (iblank = -1) if one
   *  of them has status of fringe while others have a status of field point.
   */
  void get_receptor_info();

  /** Populate the datastructures used to perform overset connectivity in Nalu
   */
  void populate_overset_info();

  //! Reference to the STK MetaData object
  stk::mesh::MetaData& meta_;

  //! Reference to the STK BulkData object
  stk::mesh::BulkData& bulk_;

  //! List of TIOGA data structures for each mesh block participating in overset
  //! connectivity
  std::vector<std::unique_ptr<TiogaBlock>> blocks_;

  //! Reference to the TIOGA API interface
  TIOGA::tioga& tg_;

  //! Pointer to STK Custom Ghosting object
  stk::mesh::Ghosting* ovsetGhosting_;

  //! Work array used to hold donor elements that require ghosting to receptor
  //! MPI ranks
  stk::mesh::EntityProcVec elemsToGhost_;

  //! Fringe {receptor, donor} information
  std::vector<std::unique_ptr<OversetInfo>> ovsetInfo_;

  //! List of receptor nodes that are shared entities across MPI ranks. This
  //! information is used to synchronize the field vs. fringe point status for
  //! these shared nodes across processor boundaries.
  std::vector<stk::mesh::EntityId> receptorIDs_;

  //! Donor elements corresponding to TiogaSTKIface::receptorIDs_ that must be
  //! ghosted to another MPI rank to ensure that owned and shared nodes are
  //! consistent.
  std::vector<stk::mesh::EntityId> donorIDs_;

  //! Ghosting exchange information
  std::vector<int> ghostCommProcs_;

  //! Name of the coordinate field sent to TIOGA for OGA
  //!
  //! For static meshes this is coordinates, for moving meshes this is
  //! "current_coordinates". Initialized when this instance is constructed.
  std::string coordsName_;

  bool use_ngp_iface_{false};
};


}  // tioga

#endif /* TIOGASTKIFACE_H */
