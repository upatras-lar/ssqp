import json

import ssqpy

from common.ProblemWrapper import ProblemWrapper


class SSQPProblem(ProblemWrapper):
    def __init__(self, horizon, dt, urdf_path, parameters):
        super().__init__(horizon, dt, urdf_path)

        self.model = ssqpy.model.Model(
            horizon, dt, urdf_path=urdf_path, **parameters
        )

    def addSquaredJointAngleCost(self, k, weight):
        cost = ssqpy.model.costs.SquaredJointAngleCost(self.model, weight**0.5)
        self.model.addCost(k, cost)

    def addSquaredJointVelocityCost(self, k, weight):
        cost = ssqpy.model.costs.SquaredJointVelocityCost(
            self.model, weight**0.5
        )
        self.model.addCost(k, cost)

    def addSquaredControlCost(self, k, weight):
        cost = ssqpy.model.costs.SquaredControlCost(self.model, weight**0.5)
        self.model.addCost(k, cost)

    def addSquaredForceCost(self, k, weight):
        cost = ssqpy.model.costs.SquaredForceCost(self.model, weight**0.5)
        self.model.addCost(k, cost)

    def addSquaredConfigurationErrorCost(self, k, desired_config, weight):
        cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
            self.model, desired_config, weight**0.5
        )
        self.model.addCost(k, cost)

    def addSquaredFramePoseErrorCost(
        self,
        k,
        frame,
        desired_translation,
        desired_rotation,
        translation_weight,
        rotation_weight,
    ):
        cost = ssqpy.model.costs.FrameSquaredPoseErrorCost(
            self.model,
            frame,
            desired_translation,
            desired_rotation,
            translation_weight**0.5,
            rotation_weight**0.5,
        )
        self.model.addCost(k, cost)

    def addFrameTranslationBound(self, k, frame, lb, ub):
        constraint = ssqpy.model.constraints.FrameTranslationBound(
            self.model, frame, lb, ub
        )
        self.model.addConstraints(k, constraint)

    def addFrameLinearVelocityBound(self, k, frame, lb, ub):
        constraint = ssqpy.model.constraints.FrameLinearVelocityBound(
            self.model,
            frame,
            lb,
            ub,
        )
        self.model.addConstraints(k, constraint)

    def addContactConstraint(self, k, frame, max_normal_force, eq_tol):
        self.model.addContactConstraint(
            k,
            frame,
            max_normal_force=max_normal_force,
            eq_tol=eq_tol,
        )

    def finalize(self, parameter_dict):
        self.model.finalize(**parameter_dict)

    def solve(self, parameters):
        if not self.model.finalized:
            self.model.finalize()

        if not parameters.get("verbose", False):
            ssqpy.setSilentMode()

        ssqp_params = ssqpy.solvers.ssqpParams()

        ssqp_params.max_iter = parameters.get("max_iter", 1000)
        ssqp_params.tolerance = parameters.get("tolerance", 1e-2)

        if "qp_solver" in parameters and parameters["qp_solver"] == "hpipm":
            ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM
            ssqp_params.hpipmParams.tol_comp = parameters.get(
                "hpipm_tol", 1e-3
            )
            ssqp_params.hpipmParams.tol_stat = parameters.get(
                "hpipm_tol", 1e-3
            )
            ssqp_params.hpipmParams.tol_eq = parameters.get("hpipm_tol", 1e-3)
            ssqp_params.hpipmParams.tol_ineq = parameters.get(
                "hpipm_tol", 1e-3
            )
            # Left at hpipmParams()'s own default unless explicitly passed
            if "hpipm_split_step" in parameters:
                ssqp_params.hpipmParams.split_step = parameters[
                    "hpipm_split_step"
                ]
            if "hpipm_prediction_correction" in parameters:
                ssqp_params.hpipmParams.prediction_correction = parameters[
                    "hpipm_prediction_correction"
                ]
            if "hpipm_warm_start" in parameters:
                ssqp_params.hpipmParams.warm_start = parameters[
                    "hpipm_warm_start"
                ]
            if "hpipm_square_root_riccati" in parameters:
                ssqp_params.hpipmParams.square_root_riccati = parameters[
                    "hpipm_square_root_riccati"
                ]
        else:
            ssqp_params.admmParams.max_iter = parameters.get(
                "admm_max_iter", 100
            )
            ssqp_params.admmParams.abs_tolerance = parameters.get(
                "admm_tol", 1e-3
            )
            ssqp_params.admmParams.rel_tolerance = parameters.get(
                "admm_tol", 1e-3
            )
            if "admm_adaptive_rho_tolerance" in parameters:
                ssqp_params.admmParams.adaptive_rho_tolerance = parameters[
                    "admm_adaptive_rho_tolerance"
                ]
            if "admm_rho_update_iter_freq" in parameters:
                ssqp_params.admmParams.rho_update_iter_freq = parameters[
                    "admm_rho_update_iter_freq"
                ]
            if "admm_warm_start" in parameters:
                ssqp_params.admmParams.warm_start = parameters.get(
                    "admm_warm_start", True
                )
            # Left at admmParams()'s own default unless explicitly passed
            if "admm_reset_rho" in parameters:
                ssqp_params.admmParams.reset_rho = parameters["admm_reset_rho"]
            if "admm_reset_y" in parameters:
                ssqp_params.admmParams.reset_y = parameters["admm_reset_y"]
            if "admm_alpha" in parameters:
                ssqp_params.admmParams.alpha = parameters["admm_alpha"]

        self.solver = ssqpy.solvers.ssqp(self.model, ssqp_params)
        self._benchmark_solver = (
            "ssqp-hpipm"
            if ssqp_params.qp_solver_type == ssqpy.solvers.QPSolverType.HPIPM
            else "ssqp-admm"
        )
        self.status, _, _ = self.solver.solve()

        convergence_data = self.solver.getConvergenceData()

        print(
            f"[SSQPY] Solve time: {convergence_data.total_duration / 1000} ms"
        )

    def save(self, output_json):
        if not self.solver.write(output_json):
            raise OSError(f"Could not write benchmark result to {output_json}")

        # Keep the native SSQP convergence record intact while adding the
        # small common envelope consumed by the cross-solver visualizer
        with open(output_json) as f:
            output = json.load(f)
        convergence = output.get("convergence", {})
        # The native writer historically stores the zero-based loop counter
        # in convergence.iterations, and the trace has one row per attempted
        # SQP iteration, so it is the unambiguous count for this common
        # envelope while preserving the native record verbatim
        iteration_data = convergence.get("iteration_data", [])
        iterations = len(iteration_data) or convergence.get("iterations")
        successful_states = {
            ssqpy.solvers.SolverState.Success,
            ssqpy.solvers.SolverState.LooseSuccess,
            ssqpy.solvers.SolverState.ConstraintSuccess,
        }
        output["benchmark"] = {
            "format_version": 1,
            "solver": self._benchmark_solver,
            "success": self.status in successful_states,
            "iterations": iterations,
            "runtime_ms": convergence.get("runtime_ms"),
            "objective": convergence.get("trajectory_cost"),
            "constraint_violation": convergence.get(
                "trajectory_constraint_violation"
            ),
            "stage_gap": convergence.get("trajectory_stage_gap"),
        }
        with open(output_json, "w") as f:
            json.dump(output, f, indent=4)

        print(f"[SSQPY] Saved solution to {output_json}")
