#ifndef STRUCTMESH_H
#define STRUCTMESH_H

#include "FieldRepo.h"

#include "AMReX_AmrCore.H"

namespace tioga_amr {

class StructMesh : public amrex::AmrCore
{
public:
    StructMesh();

    virtual ~StructMesh() = default;

    void initialize_mesh(const amrex::Real current_time=0.0);

    int num_levels() const { return finest_level + 1; }

    FieldRepo& repo() { return m_repo; }

protected:
    virtual void MakeNewLevelFromScratch(
        int lev, amrex::Real time, const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm) override;

    virtual void MakeNewLevelFromCoarse(
        int lev, amrex::Real time, const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm) override;

    virtual void RemakeLevel(
        int lev, amrex::Real time, const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm) override;

    virtual void ClearLevel(int lev) override;

    virtual void
    ErrorEst(int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow)
        override;

    FieldRepo m_repo;
};

}

#endif /* STRUCTMESH_H */
