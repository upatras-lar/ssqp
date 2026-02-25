#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "solvers/ssqp.hpp"
#include "solvers/admm_qp.hpp"
#include "solvers/hpipm_qp.hpp"
#include "model/Model.hpp"

namespace ssqp
{
namespace python
{
namespace nb = nanobind;

using namespace solvers;
using namespace model;

void bind_ssqp(nb::module_ &m)
{
    nb::enum_<SolverState>(m, "SolverState", R"doc(
        The state of the solver.
        States:
         - Success: Convergence was achieved.
         - Loose Success: Solver stopped after loose success tolerance was reached.
         - Primal Infeasible: The problem is primal infeasible (currently not used).
         - Dual Infeasible: The problem is dual infeasible (currently not used).
         - Max Iter Reached: Maximum number of iterations reached before convergence was achieved.
         - Ongoing: The solver is ongoing.
        )doc")
      .value("Success", SolverState::SUCCESS)
      .value("LooseSuccess", SolverState::LOOSE_SUCCESS)
      .value("PrimalInfeasible", SolverState::PRIMAL_INFEASIBLE)
      .value("DualInfeasible", SolverState::DUAL_INFEASIBLE)
      .value("MaxIterReached", SolverState::MAX_ITER_REACHED)
      .value("Ongoing", SolverState::ONGOING)
      .export_values();

    nb::enum_<ssqp::LineSearchMethod>(m, "LineSearchMethod", R"doc(
        The line search method to use for the SSQP update step.
        Available line search methods:
         - Filter line search: Accepts an update if there is an improvement in
           any of the three criteria functions (norm, gap or violation)
         - Merit function line search: Uses a merit function that is a weighted
           sum of each of the three criteria functions and its gradient to
           determine the update step
        )doc")
      .value("Filter", ssqp::LineSearchMethod::Filter)
      .value("Merit", ssqp::LineSearchMethod::Merit)
      .export_values();

    nb::enum_<ssqp::QPSolverType>(m, "QPSolverType", R"doc(
        The solver type to use to solve the QP at each step.
        Available solvers:
         - ADMM: The custom ADMM implementation.
         - HPIPM: The HPIPM interior point solver, using the HPIPM library.
        )doc")
      .value("ADMM", ssqp::QPSolverType::ADMM)
      .value("HPIPM", ssqp::QPSolverType::HPIPM)
      .export_values();

    nb::enum_<hpipm_mode>(m, "HPIPMMode", R"doc(
        The HPIPM solver mode.
        Available modes:
         - Robust
         - Speed
         - Balance
        )doc")
      .value("Robust", hpipm_mode::ROBUST)
      .value("Speed", hpipm_mode::SPEED)
      .value("Balance", hpipm_mode::BALANCE);

    nb::class_<ssqp::IterationConvergenceData>(
      m,
      "ssqpIterationConvergenceData",
      "SSQP iteration convergence data.")
      .def_ro("qp_iters",
              &ssqp::IterationConvergenceData::qp_iters,
              "Number of QP solver iterations.")
      .def_ro("qp_duration",
              &ssqp::IterationConvergenceData::qp_duration,
              "QP solver runtime.")
      .def_ro("step_size",
              &ssqp::IterationConvergenceData::step_size,
              "Line search step size.")
      .def_ro("norm_dx",
              &ssqp::IterationConvergenceData::norm_dx,
              "Norm of lagrangian's derivative with respect to x after update.")
      .def_ro("norm_du",
              &ssqp::IterationConvergenceData::norm_du,
              "Norm of lagrangian's derivative with respect to u after update.")

      .def_ro("first_violation_stage",
              &ssqp::IterationConvergenceData::first_violation_stage,
              "First stage where a gap/constraint is violated.")
      .def_ro("violation",
              &ssqp::IterationConvergenceData::violation,
              "Violation at first violation stage.")
      .def_ro("gap_violations",
              &ssqp::IterationConvergenceData::gap_violations,
              "Gap violations per stage.")
      .def_ro("constraint_violations",
              &ssqp::IterationConvergenceData::constraint_violations,
              "Constraint violations per stage.");

    nb::class_<ssqp::ConvergenceData>(m,
                                      "ssqpConvergenceData",
                                      "SSQP convergence data")
      .def_ro("iteration_data",
              &ssqp::ConvergenceData::iteration_data,
              "Iteration convergence data.")
      .def_ro("total_duration",
              &ssqp::ConvergenceData::total_duration,
              "SSQP solver runtime.")
      .def_ro("iterations",
              &ssqp::ConvergenceData::iterations,
              "Number of SSQP solver iterations.")
      .def_ro(
        "loose_success",
        &ssqp::ConvergenceData::loose_success,
        "Whether termination occurred due to loose criteria satisfaction.")
      .def_ro("trajectory_cost",
              &ssqp::ConvergenceData::trajectory_cost,
              "Total final trajectory cost")
      .def_ro("trajectory_stage_gap",
              &ssqp::ConvergenceData::trajectory_stage_gap,
              "Total final trajectory stage gap.")
      .def_ro("trajectory_constraint_violation",
              &ssqp::ConvergenceData::trajectory_constraint_violation,
              "Total final trajectory constraint violation.");

    nb::class_<ssqp::Params>(m, "ssqpParams", "SSQP solver parameters.")
      .def(nb::init<>())
      .def(nb::init<const std::string &>())
      .def("print", &ssqp::Params::print)
      .def("write", &ssqp::Params::write)
      .def_rw("max_iter",
              &ssqp::Params::max_iter,
              "Maximum number of SSQP iterations.")
      .def_rw("tolerance",
              &ssqp::Params::tolerance,
              "(Initial) Tolerance for lagrangian jacobian norms.")
      .def_rw("tolerance_final", &ssqp::Params::tolerance_final, R"doc(
        Final tolerance for lagrangian jacobian norms
        This value will be reached after `tolerance_update_steps` SSQP steps
        Use -1.0 to ignore
        )doc")
      .def_rw("tolerance_update_steps",
              &ssqp::Params::tolerance_update_steps,
              R"doc(
        Number of steps to go from `tolerance` to `tolerance_final`
        Updates are done linearly per step
        Ignored if `tolerance_final` is -1.0
        )doc")
      .def_rw("loose_success", &ssqp::Params::loose_success, R"doc(
        Whether to enable loose success
        Loose success will terminate if there are no updates for
        `loose_success_stale_iters` that are larger than
        `loose_success_tolerance`
        )doc")
      .def_rw("loose_success_tolerance",
              &ssqp::Params::loose_success_tolerance,
              "Tolerance for change in cost jacobian norms.")
      .def_rw("loose_success_stale_iters",
              &ssqp::Params::loose_success_stale_iters,
              "Number of iterations for stale cost jacobian norms.")
      .def_rw("line_search_method", &ssqp::Params::line_search_method, R"doc(
        Which type of line search to use
        Choose between:
         - Filter line search
         - Merit function line search
        )doc")
      .def_rw("filter_size",
              &ssqp::Params::filter_size,
              "Size of filter history.")
      .def_rw("merit_mu_gamma", &ssqp::Params::merit_mu_gamma, R"doc(
        Weight for stage gap term
        )doc")
      .def_rw("merit_mu_c",
              &ssqp::Params::merit_mu_c,
              "Weight for constraint violation term.")
      .def_rw("merit_armijo_beta",
              &ssqp::Params::merit_armijo_beta,
              "Factor for gradient of merit function.")
      .def_rw("inv_alpha_scale_factor",
              &ssqp::Params::inv_alpha_scale_factor,
              "Factor to divide α (step size) by each line search step.")
      .def_rw("min_alpha",
              &ssqp::Params::min_alpha,
              "Minimum α (step size) value.")
      .def_rw("accept_worse_update",
              &ssqp::Params::accept_worse_update,
              R"doc(
        Whether to update with min_alpha if it is reached
        Automatically also skips regularization
        )doc")
      .def_rw("skip_regularization",
              &ssqp::Params::skip_regularization,
              R"doc(
        Whether to skip regularization step if line search fails
        Will throw an exception instead
        )doc")
      .def_rw("qp_solver_type",
              &ssqp::Params::qp_solver_type,
              "The type of QP solver to use.")
      .def_rw("admm_params",
              &ssqp::Params::admm_params,
              "Parameters for the ADMM QP solver.")
      .def_rw("hpipmParams",
              &ssqp::Params::hpipm_params,
              "Parameters for the HPIPM QP solver.");

    nb::class_<admm_qp::Params>(m, "admmParams")
      .def(nb::init<>())
      .def("print", &admm_qp::Params::print)
      .def_rw("init_rho", &admm_qp::Params::init_rho, "Initial rho value.")
      .def_rw("alpha", &admm_qp::Params::alpha, "Over-relaxation parameter.")
      .def_rw("sigma",
              &admm_qp::Params::sigma,
              "Proximal regularization parameter.")
      .def_rw("rho_update",
              &admm_qp::Params::rho_update,
              "Whether to perform rho update.")
      .def_rw("rho_min",
              &admm_qp::Params::rho_min,
              "Minimum possible rho value.")
      .def_rw("rho_max",
              &admm_qp::Params::rho_max,
              "Maximum possible rho value.")
      .def_rw("adaptive_rho_tolerance",
              &admm_qp::Params::adaptive_rho_tolerance,
              "Factor by which rho must change to actually perform rho update.")
      .def_rw("rho_update_iter_freq",
              &admm_qp::Params::rho_update_iter_freq,
              R"doc(
        Number of iterations to wait before updating rho
        Also affects checking for termination
        )doc")
      .def_rw("max_iter",
              &admm_qp::Params::max_iter,
              "Maximum number of ADMM iterations.")
      .def_rw("abs_tolerance",
              &admm_qp::Params::abs_tolerance,
              "Absolute ADMM tolerance.")
      .def_rw("abs_tolerance_final",
              &admm_qp::Params::abs_tolerance_final,
              R"doc(
        Final absolute ADMM tolerance
        This value will be reached after `tolerance_update_steps` ADMM steps
        Use -1.0 to ignore
        )doc")
      .def_rw("rel_tolerance",
              &admm_qp::Params::rel_tolerance,
              "Relative ADMM tolerance.")
      .def_rw("rel_tolerance_final",
              &admm_qp::Params::rel_tolerance_final,
              R"doc(
        Final relative ADMM tolerance
        This value will be reached after `tolerance_update_steps` ADMM steps
        Use -1.0 to ignore
        )doc")
      .def_rw("tolerance_update_steps",
              &admm_qp::Params::tolerance_update_steps,
              R"doc(
        Number of steps to go from initial to final tolerances
        Updates are done linearly per step
        Ignored (per tolerance) if tolerance is -1.0
              )doc")
      .def_rw("warm_start", &admm_qp::Params::warm_start, R"doc(
        Whether to warm start ADMM with the unconstrained solution
        Note: set to false if using Inverse Dynamics
        )doc")
      .def_rw("reset_rho", &admm_qp::Params::reset_rho, R"doc(
        Whether to reset the rho parameter between each new ADMM formulation
        Note: setting to false may often cause oscillations
        )doc")
      .def_rw("reset_y", &admm_qp::Params::reset_y, R"doc(
        Whether to reset the y parameter between each new ADMM formulation
        Note: setting to false may often cause oscillations
        )doc");

    nb::class_<hpipm_qp::Params>(m, "hpipmParams")
      .def(nb::init<>())
      .def("print", &hpipm_qp::Params::print)
      .def_rw("mode", &hpipm_qp::Params::mode, "The HPIPM solver mode.")
      .def_rw("alpha_min",
              &hpipm_qp::Params::alpha_min,
              "The minimum step size.")
      .def_rw("mu0", &hpipm_qp::Params::mu0, "The initial barrier parameter.")
      .def_rw("max_iter",
              &hpipm_qp::Params::max_iter,
              "Maximum number of HPIPM iterations.")
      .def_rw("tol_stat", &hpipm_qp::Params::tol_stat)
      .def_rw("tol_eq", &hpipm_qp::Params::tol_eq)
      .def_rw("tol_ineq", &hpipm_qp::Params::tol_ineq)
      .def_rw("tol_comp", &hpipm_qp::Params::tol_comp)
      .def_rw("reg_prim",
              &hpipm_qp::Params::reg_prim,
              "The regularization term.")
      .def_rw("warm_start",
              &hpipm_qp::Params::warm_start,
              "Whether to warm start.")
      .def_rw("prediction_correction",
              &hpipm_qp::Params::prediction_correction,
              "Whether to perform prediction correction.")
      .def_rw("square_root_riccati",
              &hpipm_qp::Params::square_root_riccati,
              "Whether to use the square root riccati algorithm.")
      .def_rw("split_step",
              &hpipm_qp::Params::split_step,
              "Whether to use different step for primal and dual variables.");

    nb::class_<ssqp>(m, "ssqp", R"doc(
        Defines an SSQP problem solver based on a given SSQP model and the desired
        parameters.

        SSQP is a Structure-exploiting Sequential Quadratic Programming solver.
        The solver starts with an initial guess (either provided, or zero) for the
        states (x) and controls (u) of an optimization problem over a time horizon H.

        At each step, SSQP constructs a Quadratic Program at the current SSQP guess
        with a quadratic cost function and linearized constraints. This is done using
        the SSQP model, a model of the system being optimized which provides the
        necessary methods to linearize at each guess, like gradients and hessians.

        The solver then utilizes a QP solver to produce a solution to the constructed
        QP, which can act as an update step for the current SSQP guess. Based on this
        solution, a line search method is used to update the current guess.

        Termination criteria are then checked for the new SSQP guess. The process is
        repeated until convergence is reached.   
           )doc")
      .def(nb::init<model::Model &, const ssqp::Params &>(),
           nb::keep_alive<1, 2>(),
           nb::arg("model"),
           nb::arg("params") = ssqp::Params::Default(),
           "Initialize an SSQP solver based on the given model and SSQP "
           "parameter objects.")
      .def(
        "__init__",
        [](ssqp *self, Model &model, nb::kwargs kwargs)
        {
            ssqp::Params params;

            if (kwargs.contains("max_iter"))
            {
                params.max_iter = nb::cast<uint>(kwargs["max_iter"]);
            }
            if (kwargs.contains("tolerance"))
            {
                params.tolerance = nb::cast<double>(kwargs["tolerance"]);
            }
            if (kwargs.contains("tolerance_final"))
            {
                params.tolerance_final =
                  nb::cast<double>(kwargs["tolerance_final"]);
            }
            if (kwargs.contains("tolerance_update_steps"))
            {
                params.tolerance_update_steps =
                  nb::cast<uint>(kwargs["tolerance_update_steps"]);
            }
            if (kwargs.contains("loose_success"))
            {
                params.loose_success = nb::cast<bool>(kwargs["loose_success"]);
            }
            if (kwargs.contains("loose_success_tolerance"))
            {
                params.loose_success_tolerance =
                  nb::cast<double>(kwargs["loose_success_tolerance"]);
            }
            if (kwargs.contains("loose_success_stale_iters"))
            {
                params.loose_success_stale_iters =
                  nb::cast<uint>(kwargs["loose_success_stale_iters"]);
            }
            if (kwargs.contains("line_search_method"))
            {
                params.line_search_method = nb::cast<ssqp::LineSearchMethod>(
                  kwargs["line_search_method"]);
            }
            if (kwargs.contains("filter_size"))
            {
                params.filter_size = nb::cast<uint>(kwargs["filter_size"]);
            }
            if (kwargs.contains("merit_mu_gamma"))
            {
                params.merit_mu_gamma =
                  nb::cast<double>(kwargs["merit_mu_gamma"]);
            }
            if (kwargs.contains("merit_mu_c"))
            {
                params.merit_mu_c = nb::cast<double>(kwargs["merit_mu_c"]);
            }
            if (kwargs.contains("merit_armijo_beta"))
            {
                params.merit_armijo_beta =
                  nb::cast<double>(kwargs["merit_armijo_beta"]);
            }
            if (kwargs.contains("inv_alpha_scale_factor"))
            {
                params.inv_alpha_scale_factor =
                  nb::cast<double>(kwargs["inv_alpha_scale_factor"]);
            }
            if (kwargs.contains("min_alpha"))
            {
                params.min_alpha = nb::cast<double>(kwargs["min_alpha"]);
            }
            if (kwargs.contains("skip_regularization"))
            {
                params.skip_regularization =
                  nb::cast<bool>(kwargs["regularization"]);
            }
            if (kwargs.contains("accept_worse_update"))
            {
                params.accept_worse_update =
                  nb::cast<bool>(kwargs["accept_worse_update"]);
            }
            if (kwargs.contains("qp_solver_type"))
            {
                params.qp_solver_type =
                  nb::cast<ssqp::QPSolverType>(kwargs["qp_solver_type"]);
            }

            admm_qp::Params admm_params;

            if (kwargs.contains("admm_init_rho"))
            {
                admm_params.init_rho =
                  nb::cast<double>(kwargs["admm_init_rho"]);
            }
            if (kwargs.contains("admm_alpha"))
            {
                admm_params.alpha = nb::cast<double>(kwargs["admm_alpha"]);
            }
            if (kwargs.contains("admm_sigma"))
            {
                admm_params.sigma = nb::cast<double>(kwargs["admm_sigma"]);
            }
            if (kwargs.contains("admm_rho_update"))
            {
                admm_params.rho_update =
                  nb::cast<bool>(kwargs["admm_rho_update"]);
            }
            if (kwargs.contains("admm_rho_min"))
            {
                admm_params.rho_min = nb::cast<double>(kwargs["admm_rho_min"]);
            }
            if (kwargs.contains("admm_rho_max"))
            {
                admm_params.rho_max = nb::cast<double>(kwargs["admm_rho_max"]);
            }
            if (kwargs.contains("admm_adaptive_rho_tolerance"))
            {
                admm_params.adaptive_rho_tolerance =
                  nb::cast<double>(kwargs["admm_adaptive_rho_tolerance"]);
            }
            if (kwargs.contains("admm_rho_update_iter_freq"))
            {
                admm_params.rho_update_iter_freq =
                  nb::cast<uint>(kwargs["admm_rho_update_iter_freq"]);
            }
            if (kwargs.contains("admm_max_iter"))
            {
                admm_params.max_iter = nb::cast<uint>(kwargs["admm_max_iter"]);
            }
            if (kwargs.contains("admm_abs_tolerance"))
            {
                admm_params.abs_tolerance =
                  nb::cast<double>(kwargs["admm_abs_tolerance"]);
            }
            if (kwargs.contains("admm_abs_tolerance_final"))
            {
                admm_params.abs_tolerance_final =
                  nb::cast<double>(kwargs["admm_abs_tolerance_final"]);
            }
            if (kwargs.contains("admm_rel_tolerance"))
            {
                admm_params.rel_tolerance =
                  nb::cast<double>(kwargs["admm_rel_tolerance"]);
            }
            if (kwargs.contains("admm_rel_tolerance_final"))
            {
                admm_params.rel_tolerance_final =
                  nb::cast<double>(kwargs["admm_rel_tolerance_final"]);
            }
            if (kwargs.contains("admm_tolerance_update_steps"))
            {
                admm_params.tolerance_update_steps =
                  nb::cast<uint>(kwargs["admm_tolerance_update_steps"]);
            }
            if (kwargs.contains("admm_prim_infs_tolerance"))
            {
                admm_params.primal_infs_tolerance =
                  nb::cast<double>(kwargs["admm_prim_infs_tolerance"]);
            }
            if (kwargs.contains("admm_warm_start"))
            {
                admm_params.warm_start =
                  nb::cast<bool>(kwargs["admm_warm_start"]);
            }
            if (kwargs.contains("admm_reset_rho"))
            {
                admm_params.reset_rho =
                  nb::cast<bool>(kwargs["admm_reset_rho"]);
            }
            if (kwargs.contains("admm_reset_y"))
            {
                admm_params.reset_y = nb::cast<bool>(kwargs["admm_reset_y"]);
            }

            params.admm_params = admm_params;

            hpipm_qp::Params hpipm_params;

            if (kwargs.contains("hpipm_mode"))
            {
                hpipm_params.mode = nb::cast<hpipm_mode>(kwargs["hpipm_mode"]);
            }
            if (kwargs.contains("hpipm_max_iter"))
            {
                hpipm_params.max_iter = nb::cast<int>(kwargs["hpipm_max_iter"]);
            }
            if (kwargs.contains("hpipm_alpha_min"))
            {
                hpipm_params.alpha_min =
                  nb::cast<double>(kwargs["hpipm_alpha_min"]);
            }
            if (kwargs.contains("hpipm_mu0"))
            {
                hpipm_params.mu0 = nb::cast<double>(kwargs["hpipm_mu0"]);
            }
            if (kwargs.contains("hpipm_tol_stat"))
            {
                hpipm_params.tol_stat =
                  nb::cast<double>(kwargs["hpipm_tol_stat"]);
            }
            if (kwargs.contains("hpipm_tol_eq"))
            {
                hpipm_params.tol_eq = nb::cast<double>(kwargs["hpipm_tol_eq"]);
            }
            if (kwargs.contains("hpipm_tol_ineq"))
            {
                hpipm_params.tol_ineq =
                  nb::cast<double>(kwargs["hpipm_tol_ineq"]);
            }
            if (kwargs.contains("hpipm_tol_comp"))
            {
                hpipm_params.tol_comp =
                  nb::cast<double>(kwargs["hpipm_tol_comp"]);
            }
            if (kwargs.contains("hpipm_reg_prim"))
            {
                hpipm_params.reg_prim =
                  nb::cast<double>(kwargs["hpipm_reg_prim"]);
            }
            if (kwargs.contains("hpipm_warm_start"))
            {
                hpipm_params.warm_start =
                  nb::cast<bool>(kwargs["hpipm_warm_start"]);
            }
            if (kwargs.contains("hpipm_prediction_correction"))
            {
                hpipm_params.prediction_correction =
                  nb::cast<bool>(kwargs["hpipm_prediction_correction"]);
            }
            if (kwargs.contains("hpipm_square_root_riccati"))
            {
                hpipm_params.square_root_riccati =
                  nb::cast<bool>(kwargs["hpipm_square_root_riccati"]);
            }
            if (kwargs.contains("hpipm_split_step"))
            {
                hpipm_params.split_step =
                  nb::cast<bool>(kwargs["hpipm_split_step"]);
            }

            params.hpipm_params = hpipm_params;

            new (self) ssqp(model, params);
        },
        nb::keep_alive<1, 2>(),
        R"doc(
        Initialize an SSQP solver based on the given model and SSQP parameters. 
        Parameters, as defined in ssqpParams, can be passed as keyword arguments
        to this constructor. ADMM parameters, as defined in admmParams, can be 
        passed as keyword arguments with the `admm_` prefix. Similarly, HPIPM
        parameters, as defined in hpipmParams, can be passed as keyword
        arguments with the `hpipm_` prefix.
        )doc")
      .def(
        "solve",
        [](ssqp &self)
        {
            const Model &model = self.model();
            ssqp::Guess  solution(model.dims(), model.linear_dims());

            SolverState state = self.solve(solution);

            return nb::make_tuple(state, solution.x, solution.u);
        },
        "Solve the defined problem using SSQP.")
      .def("write",
           &ssqp::write,
           nb::arg("filename"),
           "Write the current solution to the output file with the provided "
           "name.")
      .def("getConvergenceData",
           &ssqp::convergenceData,
           "Get the SSQP solver's convergence data (after solving).");
}
}
}
