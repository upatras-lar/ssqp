#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/shared_ptr.h>

#include "param_search/search.hpp"
#include "solvers/ssqp.hpp"
#include "solvers/admm_qp.hpp"
#include "model/stagewise/StageDynamics.hpp"

namespace ssqp
{
namespace python
{
namespace nb = nanobind;

using namespace model::stagewise;

void bind_ssqp(nb::module_ &m);
void bind_MPC(nb::module_ &m);
void bind_FlatVec(nb::module_ &m);

void bind_solvers(nb::module_ &m)
{
    bind_ssqp(m);
    bind_MPC(m);
    bind_FlatVec(m);

    nb::class_<param_search::DynamicsParamGaussNewton>(
      m,
      "DynamicsParamGaussNewton")
      .def(nb::init<const std::shared_ptr<StageDynamics> &, const uint>(),
           nb::arg("dynamics"),
           nb::arg("window_size") = 100)
      .def("addData",
           &param_search::DynamicsParamGaussNewton::addData,
           nb::arg("x"),
           nb::arg("u"),
           nb::arg("x_next"));

    nb::class_<param_search::HyperParamSearch>(m, "HyperParamSearch")
      .def(nb::init())
      .def(nb::init<const std::vector<solvers::ssqp::Params> &,
                    const std::vector<model::Model> &>())
      .def("addHyperparams", &param_search::HyperParamSearch::addHyperparams)
      .def("addProblem", &param_search::HyperParamSearch::addProblem)
      .def("search", &param_search::HyperParamSearch::search);
}
}
}