#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>

#include "param_search/search.hpp"
#include "solvers/ssqp.hpp"
#include "solvers/admm_qp.hpp"

namespace ssqp
{
namespace python
{
namespace nb = nanobind;

void bind_ssqp(nb::module_ &m);
void bind_MPC(nb::module_ &m);
void bind_FlatVec(nb::module_ &m);

void bind_solvers(nb::module_ &m)
{
    bind_ssqp(m);
    bind_MPC(m);
    bind_FlatVec(m);

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