#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "param_search/search.hpp"
#include "solvers/ssqp.hpp"
#include "solvers/admm_qp.hpp"

namespace ssqp
{
namespace python
{
namespace nb = nanobind;

void bind_model(nb::module_ &);
void bind_solvers(nb::module_ &);

NB_MODULE(ssqpy, m)
{
    m.doc() = "Python bindings for SSQP";

    auto model   = m.def_submodule("model", R"doc(
Submodule containing classes related to the SSQP model, a respresentation of the
system used for optimization.
        )doc");
    auto solvers = m.def_submodule("solvers", R"doc(
Submodule containing classes related to the SSQP solvers.
        )doc");

    bind_model(model);
    bind_solvers(solvers);

    nb::class_<param_search::ParamSearch>(m, "ParamSearch")
      .def(nb::init<>())
      .def(nb::init<const std::vector<solvers::ssqp::Params> &,
                    const std::vector<model::Model> &>())
      .def("addHyperparams", &param_search::ParamSearch::addHyperparams)
      .def("addProblem", &param_search::ParamSearch::addProblem)
      .def("search", &param_search::ParamSearch::search);
}
}
}
