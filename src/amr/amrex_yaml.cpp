#include "amrex_yaml.h"

#include "AMReX_ParmParse.H"

namespace tioga_amr {

void populate_parameters(const YAML::Node& node, const std::string& prefix, const bool required)
{
    if (!node[prefix]) {
        if (required)
            throw std::runtime_error("Cannot find inputs for " + prefix);
        else
            return;
    }

    amrex::ParmParse pp(prefix);

    for (const auto& it : node[prefix]) {
        const auto& key = it.first.as<std::string>();
        const auto& val = it.second;
        if (val.IsSequence()) {
            const auto varr = val.as<amrex::Vector<std::string>>();
            pp.addarr(key.c_str(), varr);
        } else {
            const std::string value = val.as<std::string>();
            pp.add(key.c_str(), value);
        }
    }
}

} // namespace tioga_amr
