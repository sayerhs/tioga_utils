#ifndef TIOGABLOCK_H
#define TIOGABLOCK_H

#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>

#include "yaml-cpp/yaml.h"

#include <vector>
#include <memory>

#include "ngp_utils/NgpTypes.h"

namespace TIOGA {
class tioga;
class MeshBlockInfo;
}

namespace tioga_nalu {

typedef stk::mesh::Field<double, stk::mesh::Cartesian> VectorFieldType;
typedef stk::mesh::Field<double> ScalarFieldType;
typedef stk::mesh::Field<double, stk::mesh::SimpleArrayTag>  GenericFieldType;

struct NgpTiogaBlock
{
    static constexpr int max_vertex_types = 4;

    ngp::NgpDualArray<double*> xyz_;
    ngp::NgpDualArray<double*> node_res_;
    ngp::NgpDualArray<double*> cell_res_;
    ngp::NgpDualArray<double*> qnode_;
    ngp::NgpDualArray<int*> iblank_;
    ngp::NgpDualArray<int*> iblank_cell_;
    ngp::NgpDualArray<int*> wallIDs_;
    ngp::NgpDualArray<int*> ovsetIDs_;
    ngp::NgpDualArray<int*> num_verts_;
    ngp::NgpDualArray<int*> num_cells_;
    ngp::NgpDualArray<int*> connect_[max_vertex_types];

    ngp::NgpDualArray<int*> eid_map_;
    ngp::NgpDualArray<stk::mesh::EntityId*> node_gid_;
    ngp::NgpDualArray<stk::mesh::EntityId*> cell_gid_;

    ngp::NgpDualArray<double*> qsol_;
};

/**
 * Interface to convert STK Mesh Part(s) to TIOGA blocks.
 *
 * This class provides a mapping between STK mesh parts the concept of a TIOGA
 * mesh block. Each TIOGA mesh block is determined by a unique body tag and
 * requires information of all the nodes and elements comprising the mesh block
 * (within this MPI rank). TIOGA determines the global mesh information by
 * looking up the unique body tag across all MPI ranks. TIOGA requires
 * information regarding the volume mesh as well as the wall and overset
 * surfaces.
 *
 * TIOGA communicates overset connectivity via IBLANK (node) and IBLANK_CELL
 * (element) masking arrays that have flags indicating whether a node/element is
 * a hole, fringe, or a field point.
 */
class TiogaBlock
{
public:
  TiogaBlock(stk::mesh::MetaData&,
             stk::mesh::BulkData&,
             const YAML::Node&,
             const std::string&,
             const int);

  ~TiogaBlock();

  /** Setup block structure information (steps before mesh creation)
   */
  void setup();

  /** Initialize mesh data structure (steps after mesh creation)
   */
  void initialize();

  /** Update coordinates upon mesh motion
   *
   *  Update the coordinates sent to TIOGA from STK. This assumes that the mesh
   *  connectivity information itself does not change, i.e., no refinement, etc.
   */
  void update_coords();

  /** Perform full update including connectivity
   *
   */
  void update_connectivity();

  /** Register this block with TIOGA
   *
   *  @param use_ngp_interface Use TIOGA's NGP interface for registration
   */
  void register_block(TIOGA::tioga&, const bool use_ngp_iface = false);

  /** Update iblanks after connectivity updates
   */
  void update_iblanks();

  /** Update element iblanks after connectivity updates
   */
  void update_iblank_cell();

  /** Determine the custom ghosting elements for this mesh block
   *
   *  Calls the TIOGA API and populates the elements that need ghosting to other
   *  MPI ranks.
   *
   *  @param tg Reference to TIOGA API object
   *  @param egvec List of {donorElement, receptorMPIRank} pairs to be populated
   */
  void get_donor_info(TIOGA::tioga&, stk::mesh::EntityProcVec&);

  void register_solution_old(TIOGA::tioga&);

  double calculate_residuals_old();

  /** Register solution for this meshblock to TIOGA
   *
   *  @param tg Reference to TIOGA API object
   *  @param nvars Number of components in the generic field
   *  @param use_ngp_interface Use TIOGA's NGP interface for registration
   */
  void register_solution(TIOGA::tioga&, const int, const bool use_ngp_iface = false);

  //! Update solution field and return error norm
  double update_solution(const int);

  // Accessors

  //! STK Global ID for all the nodes comprising this mesh block
  inline const ngp::NgpDualArray<stk::mesh::EntityId*>::HostArrayType& node_id_map() const
  { return bdata_.node_gid_.h_view; }

private:
  TiogaBlock() = delete;
  TiogaBlock(const TiogaBlock&) = delete;

  void load(const YAML::Node&);

  inline void names_to_parts(
    const std::vector<std::string>&,
    stk::mesh::PartVector&);

  /**
   * Extract nodes from all parts to send to TIOGA
   */
  void process_nodes();

  /** Determine the local indices (into the TIOGA mesh block data structure) of
   * all the wall boundary nodes.
   */
  void process_wallbc();

  /** Determine the local indices (into the TIOGA mesh block data structure) of
   *  all the overset boundary nodes.
   */
  void process_ovsetbc();

  /** Generate the element data structure and connectivity information to send to TIOGA
   */
  void process_elements();

  void block_info_to_tioga();

  void register_block_classic(TIOGA::tioga&);
  void register_block_ngp(TIOGA::tioga&);

  //! Reference to the STK Mesh MetaData object
  stk::mesh::MetaData& meta_;

  //! Reference to the STK Mesh BulkData object
  stk::mesh::BulkData& bulk_;

  //! Part names for the nodes for this mesh block
  std::vector<std::string> blkNames_;

  //! Part names for the wall boundaries
  std::vector<std::string> wallNames_;

  //! Part names for the overset boundaries
  std::vector<std::string> ovsetNames_;

  //! Mesh parts for the nodes
  stk::mesh::PartVector blkParts_;

  //! Wall BC parts
  stk::mesh::PartVector wallParts_;

  //! Overset BC parts
  stk::mesh::PartVector ovsetParts_;

  NgpTiogaBlock bdata_;
  std::unique_ptr<TIOGA::MeshBlockInfo> minfo_;

  /** Connectivity map.
   *
   *  This map holds the number of elements present per topology type (npe ->
   *  num_elements).
   */
  std::map<int, int> conn_map_;

  /** Tioga connectivity data structure
   *
   */
  int** tioga_conn_{nullptr};

  //! Receptor information for this mesh block
  std::vector<int> receptor_info_;

  std::string coordsName_;

  //! Dimensionality of the mesh
  int ndim_;

  //! Global mesh tag identifier
  int meshtag_;

  //! Number of nodes for this mesh
  int num_nodes_{0};

  //! Number of wall BC nodes (in this processor)
  int num_wallbc_{0};

  //! Number of overset BC nodes (in this processor)
  int num_ovsetbc_{0};

  //! Flag to check if we are are already initialized
  bool is_init_ { true };

};

} // namespace tioga

#endif /* TIOGABLOCK_H */
