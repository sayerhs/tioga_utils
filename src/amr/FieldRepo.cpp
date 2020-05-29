#include "FieldRepo.h"
#include "Timer.h"

namespace tioga_amr {

/** Convert amr_wind::FieldLoc to index type for use with AMReX objects.
 */
inline amrex::IndexType index_type(const FieldLoc floc)
{
    switch (floc) {
    case FieldLoc::CELL:
        return amrex::IndexType::TheCellType();

    case FieldLoc::NODE:
        return amrex::IndexType::TheNodeType();

    case FieldLoc::XFACE:
        return amrex::IndexType(amrex::IntVect::TheDimensionVector(0));

    case FieldLoc::YFACE:
        return amrex::IndexType(amrex::IntVect::TheDimensionVector(1));

    case FieldLoc::ZFACE:
        return amrex::IndexType(amrex::IntVect::TheDimensionVector(2));
    }

    // Suppress warnings when compiling with CUDA
    return amrex::IndexType::TheCellType();
}


LevelDataHolder::LevelDataHolder()
    : m_factory(new amrex::FArrayBoxFactory())
    , m_int_fact(new amrex::DefaultFabFactory<amrex::IArrayBox>())
{}

template<>
amrex::MultiFab& FieldBase<amrex::MultiFab>::operator()(int lev) noexcept
{
    BL_ASSERT(lev < m_repo.num_active_levels());
    return  m_repo.get_multifab(m_id, lev);
}

template<>
const amrex::MultiFab& FieldBase<amrex::MultiFab>::operator()(int lev) const noexcept
{
    BL_ASSERT(lev < m_repo.num_active_levels());
    return m_repo.get_multifab(m_id, lev);
}

template<>
amrex::iMultiFab& FieldBase<amrex::iMultiFab>::operator()(int lev) noexcept
{
    BL_ASSERT(lev < m_repo.num_active_levels());
    return  m_repo.get_int_fab(m_id, lev);
}

template<>
const amrex::iMultiFab& FieldBase<amrex::iMultiFab>::operator()(int lev) const noexcept
{
    BL_ASSERT(lev < m_repo.num_active_levels());
    return m_repo.get_int_fab(m_id, lev);
}

template<>
amrex::Vector<amrex::MultiFab*> FieldBase<amrex::MultiFab>::vec_ptrs() noexcept
{
    const int nlevels = m_repo.num_active_levels();
    amrex::Vector<amrex::MultiFab*> ret;
    ret.reserve(nlevels);
    for (int lev = 0; lev < nlevels; ++lev) {
        ret.push_back(&m_repo.get_multifab(m_id, lev));
    }
    return ret;
}

template<>
amrex::Vector<const amrex::MultiFab*> FieldBase<amrex::MultiFab>::vec_const_ptrs() const noexcept
{
    const int nlevels = m_repo.num_active_levels();
    amrex::Vector<const amrex::MultiFab*> ret;
    ret.reserve(nlevels);
    for (int lev = 0; lev < nlevels; ++lev) {
        ret.push_back(static_cast<const amrex::MultiFab*>(
                          &m_repo.get_multifab(m_id, lev)));
    }
    return ret;
}

template<>
amrex::Vector<amrex::iMultiFab*> FieldBase<amrex::iMultiFab>::vec_ptrs() noexcept
{
    const int nlevels = m_repo.num_active_levels();
    amrex::Vector<amrex::iMultiFab*> ret;
    ret.reserve(nlevels);
    for (int lev = 0; lev < nlevels; ++lev) {
        ret.push_back(&m_repo.get_int_fab(m_id, lev));
    }
    return ret;
}

template<>
amrex::Vector<const amrex::iMultiFab*> FieldBase<amrex::iMultiFab>::vec_const_ptrs() const noexcept
{
    const int nlevels = m_repo.num_active_levels();
    amrex::Vector<const amrex::iMultiFab*> ret;
    ret.reserve(nlevels);
    for (int lev = 0; lev < nlevels; ++lev) {
        ret.push_back(static_cast<const amrex::iMultiFab*>(
                          &m_repo.get_int_fab(m_id, lev)));
    }
    return ret;
}

void FieldRepo::make_new_level_from_scratch(
    int lev, amrex::Real /* time */,
    const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm)
{
    auto tmon = tioga_nalu::get_timer("FieldRepo::make_new_level_from_scratch");
    m_leveldata[lev].reset(new LevelDataHolder());

    allocate_field_data(ba, dm, *m_leveldata[lev], *(m_leveldata[lev]->m_factory));
    allocate_field_data(ba, dm, *m_leveldata[lev], *(m_leveldata[lev]->m_int_fact));

    m_is_initialized = true;
}

void FieldRepo::make_new_level_from_coarse(
    int lev, amrex::Real time,
    const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm)
{
    auto tmon = tioga_nalu::get_timer("FieldRepo::make_level_from_coarse");
    std::unique_ptr<LevelDataHolder> ldata(new LevelDataHolder());

    allocate_field_data(ba, dm, *ldata, *(ldata->m_factory));
    allocate_field_data(ba, dm, *ldata, *(ldata->m_int_fact));

    m_leveldata[lev] = std::move(ldata);
    m_is_initialized = true;
}

void FieldRepo::remake_level(
    int lev, amrex::Real time,
    const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm)
{
    auto tmon = tioga_nalu::get_timer("FieldRepo::remake_level");
    std::unique_ptr<LevelDataHolder> ldata(new LevelDataHolder());

    allocate_field_data(ba, dm, *ldata, *(ldata->m_factory));
    allocate_field_data(ba, dm, *ldata, *(ldata->m_int_fact));

    m_leveldata[lev] = std::move(ldata);
    m_is_initialized = true;
}

void FieldRepo::clear_level(int lev)
{
    auto tmon = tioga_nalu::get_timer("FieldRepo::clear_level");
    m_leveldata[lev].reset();
}

Field& FieldRepo::declare_field(
    const std::string& name,
    const int ncomp,
    const int ngrow,
    const FieldLoc floc)
{
    auto tmon = tioga_nalu::get_timer("FieldRepo::declare_field");
    // If the field is already registered check and return the fields
    {
        auto found = m_fid_map.find(name);
        if (found != m_fid_map.end()) {
            auto& field = *m_field_vec[found->second];

            if ((ncomp != field.num_comp()) ||
                (floc != field.field_location())) {
                amrex::Abort("Attempt to reregister field with inconsistent parameters: "
                             + name);
            }
            return field;
        }
    }

    {
        const int fid = m_field_vec.size();

        std::unique_ptr<Field> field(
            new Field(*this, name, fid, ncomp, ngrow, floc));

        if (m_is_initialized)
            allocate_field_data(*field);

        m_field_vec.emplace_back(std::move(field));
        m_fid_map[name] = fid;
    }

    return *m_field_vec[m_fid_map[name]];
}

IntField& FieldRepo::declare_int_field(
    const std::string& name,
    const int ncomp,
    const int ngrow,
    const FieldLoc floc)
{
    auto tmon = tioga_nalu::get_timer("FieldRepo::declare_int_field");

    // If the field is already registered check and return the fields
    {
        auto found = m_int_fid_map.find(name);
        if (found != m_int_fid_map.end()) {
            auto& field = *m_int_field_vec[found->second];

            if ((ncomp != field.num_comp()) ||
                (floc != field.field_location())) {
                amrex::Abort("Attempt to reregister field with inconsistent parameters: "
                             + name);
            }
            return field;
        }
    }

    {
        const int fid = m_int_field_vec.size();

        std::unique_ptr<IntField> field(
            new IntField(*this, name, fid, ncomp, ngrow, floc));

        if (m_is_initialized)
            allocate_field_data(*field);

        m_int_field_vec.emplace_back(std::move(field));
        m_int_fid_map[name] = fid;
    }

    return *m_int_field_vec[m_int_fid_map[name]];
}

Field& FieldRepo::get_field(const std::string& name) const
{
    auto tmon = tioga_nalu::get_timer("FieldRepo::get_field");
    const auto found = m_fid_map.find(name);
    if (found == m_fid_map.end()) {
        amrex::Abort("Cannot find field: " + name);
    }

    AMREX_ASSERT(found->second < static_cast<unsigned>(m_field_vec.size()));
    return *m_field_vec[found->second];
}

bool FieldRepo::field_exists(const std::string& name) const
{
    const auto found = m_fid_map.find(name);
    return (found != m_fid_map.end());
}

IntField& FieldRepo::get_int_field(const std::string& name) const
{
    auto tmon = tioga_nalu::get_timer("FieldRepo::get_int_field");
    const auto found = m_int_fid_map.find(name);
    if (found == m_int_fid_map.end()) {
        amrex::Abort("Cannot find field: " + name);
    }

    AMREX_ASSERT(found->second < static_cast<unsigned>(m_int_field_vec.size()));
    return *m_int_field_vec[found->second];
}

bool FieldRepo::int_field_exists(const std::string& name) const
{
    const auto found = m_int_fid_map.find(name);
    return (found != m_int_fid_map.end());
}

void FieldRepo::allocate_field_data(
    const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm,
    LevelDataHolder& level_data,
    const amrex::FabFactory<amrex::FArrayBox>& factory)
{
    auto& mfab_vec = level_data.m_mfabs;

    for (auto& field : m_field_vec) {
        auto ba1 =
            amrex::convert(ba, index_type(field->field_location()));

        mfab_vec.emplace_back(
            ba1, dm, field->num_comp(), field->num_grow(), amrex::MFInfo(),
            factory);

        mfab_vec.back().setVal(0.0);
    }
}

void FieldRepo::allocate_field_data(
    int lev,
    Field& field,
    LevelDataHolder& level_data,
    const amrex::FabFactory<amrex::FArrayBox>& factory)
{
    auto& mfab_vec = level_data.m_mfabs;
    AMREX_ASSERT(mfab_vec.size() == field.id());
    const auto ba = amrex::convert(
        m_mesh.boxArray(lev), index_type(field.field_location()));

    mfab_vec.emplace_back(
        ba, m_mesh.DistributionMap(lev), field.num_comp(), field.num_grow(),
        amrex::MFInfo(), factory);

    mfab_vec.back().setVal(0.0);
}

void FieldRepo::allocate_field_data(Field& field)
{
    for (int lev=0; lev <= m_mesh.finestLevel(); ++lev) {
        allocate_field_data(lev, field, *m_leveldata[lev], *(m_leveldata[lev]->m_factory));
    }
}

void FieldRepo::allocate_field_data(
    const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm,
    LevelDataHolder& level_data,
    const amrex::FabFactory<amrex::IArrayBox>& factory)
{
    auto& fab_vec = level_data.m_int_fabs;

    for (auto& field : m_int_field_vec) {
        auto ba1 =
            amrex::convert(ba, index_type(field->field_location()));

        fab_vec.emplace_back(
            ba1, dm, field->num_comp(), field->num_grow(), amrex::MFInfo(),
            factory);
    }
}


void FieldRepo::allocate_field_data(
    int lev,
    IntField& field,
    LevelDataHolder& level_data)
{
    auto& fab_vec = level_data.m_int_fabs;
    AMREX_ASSERT(fab_vec.size() == field.id());

    const auto ba = amrex::convert(
        m_mesh.boxArray(lev), index_type(field.field_location()));

    fab_vec.emplace_back(
        ba, m_mesh.DistributionMap(lev), field.num_comp(), field.num_grow(),
        amrex::MFInfo(), *level_data.m_int_fact);
}

void FieldRepo::allocate_field_data(IntField& field)
{
  for (int lev = 0; lev <= m_mesh.finestLevel(); ++lev) {
    allocate_field_data(lev, field, *m_leveldata[lev]);
  }
}

} // namespace tioga_amr
