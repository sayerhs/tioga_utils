#include "StkIface.h"
#include "TiogaBlock.h"
#include "Timer.h"

namespace tioga_nalu {

StkIface::StkIface(stk::ParallelMachine& comm)
    : comm_(comm)
    , meta_()
    , bulk_(meta_, comm, stk::mesh::BulkData::NO_AUTO_AURA)
    , stkio_(comm)
{}

void StkIface::load(const YAML::Node& node)
{
    if (!stk::parallel_machine_rank(comm_)) {
        std::cout << "Reading inputs for STK mesh" << std::endl;
    }
    const int nproc = stk::parallel_machine_size(bulk_.parallel());

    if ((nproc > 1) && node["decomposition_method"]) {
        auto decomp_method = node["decomposition_method"].as<std::string>();
        stkio_.property_add(
            Ioss::Property("DECOMPOSITION_METHOD", decomp_method));
    }

    {
        auto timeMon = get_timer("StkIface::init_meta_data");
        const auto inp_mesh = node["input_mesh"].as<std::string>();
        if (!stk::parallel_machine_rank(comm_)) {
            std::cout << "Reading meta data for: " << inp_mesh << std::endl;
        }
        stkio_.add_mesh_database(inp_mesh, stk::io::READ_MESH);
        stkio_.set_bulk_data(bulk_);
        stkio_.create_input_mesh();
        stkio_.add_all_mesh_fields_as_input_fields();
    }

    if (node["motion_info"]) {
        has_motion_ = true;
        const auto& mnode = node["motion_info"];
        motion_.reset(new MeshMotion(meta_, bulk_, mnode));
    }

    const auto& oset_info = node["overset_info"];
    tg_.reset(new TiogaSTKIface(meta_, bulk_, oset_info, coordinates_name()));
}

void StkIface::setup()
{
    if (!stk::parallel_machine_rank(comm_)) {
        std::cout << "Performing STK setup" << std::endl;
    }
    auto timeMon = get_timer("StkIface::setup");
    if (has_motion_)
        motion_->setup();
    tg_->setup();

    if (num_vars() > 0) {
        auto& qvar = meta_.declare_field<GenericFieldType>(stk::topology::NODE_RANK, "qvars");
        stk::mesh::put_field_on_mesh(qvar, meta_.universal_part(), num_vars(), nullptr);
    }
}

void StkIface::initialize()
{
    if (!stk::parallel_machine_rank(comm_)) {
        std::cout << "Performing STK initialize" << std::endl;
    }
    auto timeMon = get_timer("StkIface::initialize");
    if (has_motion_)
        motion_->initialize();
    tg_->initialize();

}

void StkIface::populate_bulk_data()
{
    if (!stk::parallel_machine_rank(comm_)) {
        std::cout << "Reading bulk data" << std::endl;
    }
    auto timeMon = get_timer("StkIface::populate_bulk_data");
    stkio_.populate_bulk_data();
}

void StkIface::load_and_initialize_all(const YAML::Node& node)
{
    load(node);
    setup();
    populate_bulk_data();
    initialize();
}

void StkIface::register_solution()
{
    if (num_vars() < 1) return;

    init_vars();
    tg_->register_solution(num_vars());
}


void StkIface::write_outputs(const YAML::Node& node, const double time)
{
    auto tmon = tioga_nalu::get_timer("TiogaAMRIface::write_outputs");

    bool has_motion = false;
    if (node["motion_info"])
        has_motion = true;

    ScalarFieldType* ibf = meta_.get_field<ScalarFieldType>(
        stk::topology::NODE_RANK, "iblank");
    ScalarFieldType* ibcell = meta_.get_field<ScalarFieldType>(
        stk::topology::ELEM_RANK, "iblank_cell");

    std::string out_mesh = node["output_mesh"].as<std::string>();
    if (bulk_.parallel_rank() == 0)
        std::cout << "Writing STK output file: " << out_mesh << std::endl;
    size_t fh = stkio_.create_output_mesh(out_mesh, stk::io::WRITE_RESTART);
    stkio_.add_field(fh, *ibf);
    stkio_.add_field(fh, *ibcell);

    if (has_motion) {
        VectorFieldType* mesh_disp = meta_.get_field<VectorFieldType>(
            stk::topology::NODE_RANK, "mesh_displacement");
        stkio_.add_field(fh, *mesh_disp);
    }

    if (num_vars() > 0) {
        auto* qvars = meta_.get_field<GenericFieldType>(
            stk::topology::NODE_RANK, "qvars");
        stkio_.add_field(fh, *qvars);
    }

    stkio_.begin_output_step(fh, time);
    stkio_.write_defined_output_fields(fh);
    stkio_.end_output_step(fh);
}

void StkIface::init_vars()
{
    auto tmon = tioga_nalu::get_timer("TiogaAMRIface::init_vars");

    auto* coords = meta_.get_field<VectorFieldType>(
        stk::topology::NODE_RANK, coordinates_name());
    auto* qvars = meta_.get_field<GenericFieldType>(stk::topology::NODE_RANK, "qvars");
    const stk::mesh::Selector sel = stk::mesh::selectField(*qvars);
    const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);

    for (auto b: bkts) {
        for (size_t in=0; in < b->size(); ++in) {
            const auto node = (*b)[in];
            const double* xyz = stk::mesh::field_data(*coords, node);
            double* qq = stk::mesh::field_data(*qvars, node);

            if (ncell_vars_ > 0) {
                switch(ncell_vars_) {
                case 1:
                    qq[0] = xyz[0] + xyz[1] + xyz[2];
                    break;

                case 3:
                    qq[0] = xyz[0] + xyz[1] + xyz[2];
                    qq[1] = xyz[0] * xyz[0] + xyz[1] * xyz[1]  + xyz[2] * xyz[2] ;
                    qq[2] = xyz[0] * xyz[1] * xyz[2];
                    break;

                default:
                    for (int n=0; n < ncell_vars_; ++n) {
                        const int np1 = n+1;
                        qq[n] = np1 * xyz[0] + 2.0 * np1 * xyz[1] + 3.0 * np1 * xyz[2];
                    }
                    break;
                }
            }

            if (nnode_vars_ > 0) {
                const int ii = ncell_vars_;
                switch(nnode_vars_) {
                case 1:
                    qq[ii + 0] = xyz[0] + xyz[1] + xyz[2];
                    break;

                case 3:
                    qq[ii + 1] = xyz[0] + xyz[1] + xyz[2];
                    qq[ii + 2] = xyz[0] * xyz[0] + xyz[1] * xyz[1]  + xyz[2] * xyz[2] ;
                    qq[ii + 3] = xyz[0] * xyz[1] * xyz[2];
                    break;

                default:
                    for (int n=0; n < nnode_vars_; ++n) {
                        const int np1 = n + 1;
                        qq[ii + n] = np1 * xyz[0] + 2.0 * np1 * xyz[1] + 3.0 * np1 * xyz[2];
                    }
                    break;
                }
            }
        }
    }
}

} // namespace tioga_nalu
