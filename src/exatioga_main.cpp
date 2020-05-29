#include <iostream>
#include <fstream>

#include "amrex_yaml.h"
#include "ExaTioga.h"
#include "Timer.h"

#include "stk_util/parallel/Parallel.hpp"
#include "stk_util/environment/OptionsSpecification.hpp"
#include "stk_util/environment/ParseCommandLineArgs.hpp"

#include "AMReX.H"

bool parse_cmdline(
    int& argc, char** argv,
    stk::ParsedOptions& options, std::string& inpfile)
{
    stk::OptionsSpecification opts("TIOGA utils options");
    opts.add_options()
        ("help,h", "Print help message and exit")
        ("input-file,i", "YAML input file",
         stk::DefaultValue<std::string>("exatioga.yaml"),
         stk::TargetPointer<std::string>(&inpfile));

    stk::parse_command_line_args(argc, const_cast<const char**>(argv), opts, options);

    if (options.count("help")) {
        if (!stk::parallel_machine_rank(MPI_COMM_WORLD))
            std::cout << opts << std::endl;
        return true;
    }

    return false;
}

int main(int argc, char** argv)
{
    stk::ParallelMachine comm = stk::parallel_machine_init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    std::string input_file;
    stk::ParsedOptions options;

    const bool do_exit = parse_cmdline(argc, argv, options, input_file);
    if (do_exit) {
        Kokkos::finalize();
        stk::parallel_machine_finalize();
        return 0;
    }

    std::ifstream fin(input_file.c_str());
    if (!fin.good()) {
        if (!stk::parallel_machine_rank(comm))
            std::cerr << "Cannot find input file: " << input_file << std::endl;
        return 1;
    }

    const YAML::Node doc = YAML::LoadFile(input_file);
    if (!stk::parallel_machine_rank(comm)) {
        std::cout << "ExaWind TIOGA overset connectivity\n"
                  << "Processing inputs from file: " << input_file
                  << std::endl << std::endl;
    }
    {
        int targc = 0;
        char** targv = nullptr;
        amrex::Initialize(targc, targv, true, comm, [&]() {
            if (doc["amrex"]) {
                tioga_amr::populate_parameters(doc, "amrex");
            }
        });
    }

    {
        tioga_amr::ExaTioga driver(comm);
        driver.init_amr(doc);
        driver.init_stk(doc);
        // tioga_amr::TiogaAMRIface tg_amr(doc["amr_wind"]);
        // tioga_nalu::StkIface tg_stk(comm);

        // tg_amr.initialize();
        // tg_stk.load_and_initialize_all(doc["nalu_wind"]);
    }

    Teuchos::TimeMonitor::summarize(
        std::cout, false, true, false, Teuchos::Union);
    amrex::Finalize();
    Kokkos::finalize();
    stk::parallel_machine_finalize();
}
