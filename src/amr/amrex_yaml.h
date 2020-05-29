#ifndef AMREX_YAML_H
#define AMREX_YAML_H

#include "yaml-cpp/yaml.h"

#include "AMReX_Vector.H"
#include "AMReX_ParmParse.H"

namespace YAML {

template <typename T>
struct convert<amrex::Vector<T>>
{
    static bool decode(const Node& node, amrex::Vector<T>& rhs)
    {
        if (!node.IsSequence()) return false;

        rhs.resize(node.size());
        for (int i = 0; i < node.size(); ++i) {
          rhs[i] = node[i].as<T>();
        }
        return true;
    }
};

} // namespace YAML

namespace tioga_amr {

void populate_parameters(
    const YAML::Node& node,
    const std::string& prefix,
    const bool required = true);

}

#endif /* AMREX_YAML_H */
