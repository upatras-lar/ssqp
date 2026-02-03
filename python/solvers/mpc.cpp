#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>
#include <nanobind/eigen/dense.h>

#include "solvers/mpc.hpp"
#include "model/Model.hpp"

namespace ssqp
{
namespace python
{
namespace nb = nanobind;

using namespace solvers;
using namespace model;

void bind_MPC(nb::module_ &m)
{
    nb::class_<MPC>(m, "MPC")
      .def(
        "__init__",
        [](MPC                   *self,
           Model                 &model,
           const ssqp::Params    &ssqp_params,
           const admm_qp::Params &admm_params,
           const uint             sqp_iters,
           const uint             admm_iters)
        {
            MPC::Params params;

            params.sqp_iters  = sqp_iters;
            params.admm_iters = admm_iters;

            new (self) MPC(model, params, ssqp_params, admm_params);
        },
        nb::keep_alive<1, 2>(),
        nb::arg("model"),
        nb::arg("ssqp_params") = ssqp::Params::Default(),
        nb::arg("admm_params") = admm_qp::Params::Default(),
        nb::kw_only(),
        nb::arg("sqp_iters")  = 5,
        nb::arg("admm_iters") = 50)
      .def("step",
           static_cast<std::pair<FlatVec, FlatVec> (
             MPC::*)(const std::vector<VecX> &, const std::vector<VecX> &)>(
             &MPC::step),
           nb::arg("states"),
           nb::arg("controls"))
      .def("step", static_cast<VecX (MPC::*)(const VecX &)>(&MPC::step));
}

}
}