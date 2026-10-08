#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>
#include <nanobind/eigen/dense.h>

#include "solvers/mpc.hpp"
#include "model/Model.hpp"
#include "../model/model_reference.hpp"

namespace ssqp
{
namespace python
{
namespace nb = nanobind;

using namespace solvers;
using namespace model;

void bind_MPC(nb::module_ &m)
{
    nb::class_<MPC>(m, "MPC", nb::dynamic_attr(), R"doc(
        Helper class which utilizes SSQP to perform MPC for real-time SSQP-based 
        control.

        Repeatedly uses SSQP with low iterations to solve a problem for the current 
        state of a system, applies the first of the generated controls, updates
        the initial guess using feedback from the system and a warm start using the
        previous solution  and repeats the process.
        )doc")
      .def(
        "__init__",
        [](MPC                *self,
           Model              &model,
           const ssqp::Params &ssqp_params,
           const uint          sqp_iters,
           const uint          qp_iters,
           const double        max_time,
           const bool          keep_convergence_data)
        {
            MPC::Params params;

            params.sqp_iters             = sqp_iters;
            params.qp_iters              = qp_iters;
            params.max_time              = max_time;
            params.keep_convergence_data = keep_convergence_data;

            new (self) MPC(model, params, ssqp_params);
        },
        nb::call_policy<keep_model>(),
        nb::arg("model"),
        nb::arg("ssqp_params") = ssqp::Params::Default(),
        nb::kw_only(),
        nb::arg("sqp_iters")             = 5,
        nb::arg("qp_iters")              = 50,
        nb::arg("max_time")              = -1.0,
        nb::arg("keep_convergence_data") = false,
        R"doc(
        Initialize an MPC helper for the given model, the given MPC, SSQP and
        QP parameters.
        Note that the max iterations used for SSQP and the QP solver, and the
        SSQP time limit (max_time, in seconds), will be determined by the MPC
        parameters only.
        )doc")
      .def("reinitialize",
           static_cast<void (MPC::*)(const VecX &)>(&MPC::reinitialize),
           nb::arg("state"),
           "Sets MPC warm-start to the given state and zero controls")
      .def("reinitialize",
           static_cast<void (MPC::*)(const FlatVec &, const FlatVec &)>(
             &MPC::reinitialize),
           nb::arg("states"),
           nb::arg("controls"),
           "Sets MPC warm-start to the given states and controls")
      .def("reinitialize",
           static_cast<void (MPC::*)(const std::vector<VecX> &,
                                     const std::vector<VecX> &)>(
             &MPC::reinitialize),
           nb::arg("states"),
           nb::arg("controls"),
           "Sets MPC warm-start to the given states and controls")
      .def("step",
           static_cast<std::pair<FlatVec, FlatVec> (
             MPC::*)(const std::vector<VecX> &, const std::vector<VecX> &)>(
             &MPC::step),
           nb::arg("states"),
           nb::arg("controls"),
           nb::call_guard<nb::gil_scoped_release>(),
           "Uses the provided states and controls for warm-starting.")
      .def("step",
           static_cast<std::pair<FlatVec, FlatVec> (MPC::*)(const VecX &)>(
             &MPC::step),
           R"doc(
        Receding horizon initialization step
        Uses the provided state as feedback and the previous solution for
        warm-starting

        Other Python threads keep running while it solves
           )doc",
           nb::arg("state"),
           nb::call_guard<nb::gil_scoped_release>())
      .def("getConvergenceData",
           &MPC::getConvergenceData,
           "Returns stored convergence data")
      .def("feedbackGain",
           &MPC::feedbackGain,
           nb::arg("stage"),
           R"doc(
        Feedback gain K (nu-by-nx) of the last step at the given stage, giving
        the change du = K dx of the stage's controls for a change dx of its
        state. dx is in the tangent space of the dynamics, as returned by
        state_right_minus. Between steps, it corrects the first stage's
        controls for the measured state:
        u = u.stage(0) + K @ dynamics.state_right_minus(state, x.stage(0))

        HPIPM returns the Riccati gain of its last interior point iteration.
        ADMM returns the gain of its final LQR, which also penalizes inactive
        constraints, so it is stiffer.
           )doc");
}

}
}