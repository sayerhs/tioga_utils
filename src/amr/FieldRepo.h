#ifndef FIELDREPO_H
#define FIELDREPO_H

#include <string>
#include <unordered_map>
#include <memory>

#include "AMReX_AmrCore.H"
#include "AMReX_MultiFab.H"
#include "AMReX_iMultiFab.H"
#include "AMReX_Vector.H"

namespace tioga_amr {

/** Possible locations for a field in a structured mesh
 */
enum class FieldLoc : int {
    CELL,  ///< Cell-centered (default)
    NODE,  ///< Node-centered (e.g., for pressure)
    XFACE, ///< Face-centered in x-direction (e.g., face normal velocity)
    YFACE, ///< Face-centered in y-direction
    ZFACE  ///< Face-centered in z-direction
};

/** Object that holds the MultiFab instances at all levels for a given field
 */
struct LevelDataHolder
{
    LevelDataHolder();

    //! real multifabs for all the known fields at this level
    amrex::Vector<amrex::MultiFab> m_mfabs;
    //! Factory for creating new FABs
    std::unique_ptr<amrex::FabFactory<amrex::FArrayBox>> m_factory;

    //! int fabs for all known fields at this level
    amrex::Vector<amrex::iMultiFab> m_int_fabs;
    std::unique_ptr<amrex::FabFactory<amrex::IArrayBox>> m_int_fact;
};

class FieldRepo;

template<typename T>
class FieldBase
{
public:
    friend class FieldRepo;

    FieldBase(const FieldBase&) = delete;
    FieldBase& operator=(const FieldBase&) = delete;
    ~FieldBase() = default;

    inline const std::string& name() { return m_name; }

    //! Unique integer identifier for this field
    inline unsigned id() const { return m_id; }

    //! Number of components for this field
    inline int num_comp() const { return m_ncomp; }

    //! Ghost cells
    inline const amrex::IntVect& num_grow() const { return m_ngrow; }

    inline FieldLoc field_location() const { return m_floc; }

    T& operator()(int lev) noexcept;
    const T& operator()(int lev) const noexcept;

    amrex::Vector<T*> vec_ptrs() noexcept;

    amrex::Vector<const T*> vec_const_ptrs() const noexcept;

    const FieldRepo& repo() const { return m_repo; }

protected:
    FieldBase(
        FieldRepo& repo,
        const std::string& name,
        const unsigned fid,
        const int ncomp = 1,
        const int ngrow = 1,
        const FieldLoc floc = FieldLoc::CELL)
        : m_repo(repo), m_name(name), m_id(fid), m_ncomp(ncomp), m_ngrow(ngrow), m_floc(floc)
    {}

    FieldRepo& m_repo;
    std::string m_name;
    const unsigned m_id;
    int m_ncomp;
    amrex::IntVect m_ngrow;
    FieldLoc m_floc;
};

using Field = FieldBase<amrex::MultiFab>;
using IntField = FieldBase<amrex::iMultiFab>;

class FieldRepo
{
public:
    friend class FieldBase<amrex::MultiFab>;
    friend class FieldBase<amrex::iMultiFab>;

    FieldRepo(const amrex::AmrCore& mesh)
        : m_mesh(mesh)
        , m_leveldata(mesh.maxLevel() + 1)
    {}

    FieldRepo(const FieldRepo&) = delete;
    FieldRepo& operator=(const FieldRepo&) = delete;
    ~FieldRepo() = default;

    //! Perform field data management tasks when
    //! amrex::AmrCore::MakeNewLevelFromScratch is called
    void make_new_level_from_scratch(
        int lev, amrex::Real time,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm);

    void make_new_level_from_coarse(
        int lev, amrex::Real time,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm);

    void remake_level(
        int lev, amrex::Real time,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm);

    void clear_level(int lev);

    /** Create a new field instance
     *
     *  @param name Unique identifier for this field
     *  @param ncomp Number of components in this field (defalt: 1)
     *  @param ngrow Number of ghost cells/nodes for this field (default: 1)
     *  @param nstates Number of time states for this field (default: 1)
     *  @param floc Field location (default: cell-centered)
     */
    Field& declare_field(
        const std::string& name,
        const int ncomp = 1,
        const int ngrow = 0,
        const FieldLoc floc = FieldLoc::CELL);

    inline Field& declare_cc_field(
        const std::string& name,
        const int ncomp = 1,
        const int ngrow = 0)
    {
        return declare_field(name, ncomp, ngrow, FieldLoc::CELL);
    }

    inline Field& declare_nd_field(
        const std::string& name,
        const int ncomp = 1,
        const int ngrow = 0)
    {
        return declare_field(name, ncomp, ngrow, FieldLoc::NODE);
    }

    /** Return a previously created field identified by name and time state
     */
    Field& get_field(const std::string& name) const;

    //! Query if field uniquely identified by name and time state exists in repository
    bool field_exists(const std::string& name) const;

    IntField& declare_int_field(
        const std::string& name,
        const int ncomp = 1,
        const int ngrow = 0,
        const FieldLoc floc = FieldLoc::CELL);

    inline IntField& declare_cc_int_field(
        const std::string& name,
        const int ncomp = 1,
        const int ngrow = 0)
    {
        return declare_int_field(name, ncomp, ngrow, FieldLoc::CELL);
    }

    inline IntField& declare_nd_int_field(
        const std::string& name,
        const int ncomp = 1,
        const int ngrow = 0)
    {
        return declare_int_field(name, ncomp, ngrow, FieldLoc::NODE);
    }

    IntField& get_int_field(const std::string& name) const;

    bool int_field_exists(const std::string& name) const;

    const amrex::AmrCore& mesh() const { return m_mesh; }

    //! Total number of levels currently active in the AMR mesh
    int num_active_levels() const noexcept { return m_mesh.finestLevel() + 1; }

    //! Number of fields registered in the database
    int num_fields() const noexcept { return m_field_vec.size(); }

private:
    /** Return the amrex::MultiFab instance for a field at a given level
     *
     *  \param fid Unique integer field identifier for this field
     *  \param lev AMR level
     */
    inline amrex::MultiFab& get_multifab(const unsigned fid, const int lev) noexcept
    {
        BL_ASSERT(lev <= m_mesh.finestLevel());
        return m_leveldata[lev]->m_mfabs[fid];
    }

    inline amrex::iMultiFab& get_int_fab(const unsigned fid, const int lev) noexcept
    {
        BL_ASSERT(lev <= m_mesh.finestLevel());
        return m_leveldata[lev]->m_int_fabs[fid];
    }

    //! Allocate field data for a single level outside of regrid
    void allocate_field_data(
        int lev,
        Field& field,
        LevelDataHolder& fdata,
        const amrex::FabFactory<amrex::FArrayBox>& factory);

    //! Allocate field data at all levels
    void allocate_field_data(Field& field);

    //! Allocate data at a level during regrid
    void allocate_field_data(
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm,
        LevelDataHolder& fdata,
        const amrex::FabFactory<amrex::FArrayBox>& factory);

    void allocate_field_data(int lev, IntField& field, LevelDataHolder& fdata);

    void allocate_field_data(IntField& field);

    void allocate_field_data(
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm,
        LevelDataHolder& fdata,
        const amrex::FabFactory<amrex::IArrayBox>& factory);

    //! Reference to the mesh instance
    const amrex::AmrCore& m_mesh;

    //! Array (size: nlevels) of data holder that contains another array of
    //! MultiFabs for all fields at that level.
    amrex::Vector<std::unique_ptr<LevelDataHolder>> m_leveldata;

    //! References to field instances identified by unique integer
    mutable amrex::Vector<std::unique_ptr<Field>> m_field_vec;

    //! Reference to integer field instances identified by unique integer
    mutable amrex::Vector<std::unique_ptr<IntField>> m_int_field_vec;

    //! Map of field name to unique integer ID for lookups
    std::unordered_map<std::string, size_t> m_fid_map;

    //! Map of integer field name to unique integer ID for lookups
    std::unordered_map<std::string, size_t> m_int_fid_map;

    //! Flag indicating if mesh is available to allocate field data
    bool m_is_initialized{false};
};

} // tioga_amr


#endif /* FIELDREPO_H */
