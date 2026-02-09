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
    nb::class_<MPC>(m, "MPC", R"doc(
        Helper class which utilizes SSQP to perform MPC for real-time SSQP-based 
        control.

        Repeatedly uses SSQP with low iterations to solve a problem for the current 
        state of a system, applies the first of the generated controls, updates
        the initial guess using feedback from the system and a warm start using the
        previous solution  and repeats the process.
        )doc")
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
        nb::arg("admm_iters") = 50,
        R"doc(
        Initialize an MPC helper for the given model, the given MPC, SSQP and
        ADMM parameters.
        Note that the max iterations used for SSQP and ADMM will be determined
        by the MPC parameters only.
        )doc")
      .def("step",
           static_cast<std::pair<FlatVec, FlatVec> (
             MPC::*)(const std::vector<VecX> &, const std::vector<VecX> &)>(
             &MPC::step),
           nb::arg("states"),
           nb::arg("controls"),
           "Uses the provided states and controls for warm-starting.")
      .def("step", static_cast<VecX (MPC::*)(const VecX &)>(&MPC::step), R"doc(
        Receding horizon initialization step.
        Uses the provided state as feedback and the previous solution for
        warm-starting.
           )doc");
}

}
}