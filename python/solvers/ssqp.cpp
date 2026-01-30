#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "solvers/ssqp.hpp"
#include "solvers/admm_qp.hpp"
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
    nb::enum_<SolverState>(m, "SolverState")
      .value("Success", SolverState::SUCCESS)
      .value("LooseSuccess", SolverState::LOOSE_SUCCESS)
      .value("PrimalInfeasible", SolverState::PRIMAL_INFEASIBLE)
      .value("DualInfeasible", SolverState::DUAL_INFEASIBLE)
      .value("MaxIterReached", SolverState::MAX_ITER_REACHED)
      .value("Ongoing", SolverState::ONGOING)
      .export_values();

    nb::class_<ssqp>(m, "ssqp")
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
            if (kwargs.contains("use_filter"))
            {
                params.use_filter = nb::cast<bool>(kwargs["use_filter"]);
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

            new (self) ssqp(model, params, admm_params);
        },
        nb::keep_alive<1, 2>())
      .def("solve",
           [](ssqp &self)
           {
               const Model &model = self.model();
               ssqp::Guess  solution(model.dims(), model.linear_dims());

               SolverState state = self.solve(solution);

               return nb::make_tuple(state, solution.x, solution.u);
           })
      .def("write", &ssqp::write, nb::arg("filename"));
}
}
}
