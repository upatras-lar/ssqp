from ProblemWrapper import ProblemWrapper
import ssqpy


class SSQPProblem(ProblemWrapper):
    def __init__(self, horizon, dt, urdf_path, parameters):
        super().__init__(horizon, dt, urdf_path)

        self.model = ssqpy.model.Model(
            horizon, dt, urdf_path=urdf_path, **parameters
        )

    def addSquaredJointAngleCost(self, k, weight):
        cost = ssqpy.model.costs.SquaredJointAngleCost(self.model, weight)
        self.model.addCost(k, cost)

    def addSquaredJointVelocityCost(self, k, weight):
        cost = ssqpy.model.costs.SquaredJointVelocityCost(self.model, weight)
        self.model.addCost(k, cost)

    def addSquaredControlCost(self, k, weight):
        cost = ssqpy.model.costs.SquaredControlCost(self.model, weight)
        self.model.addCost(k, cost)

    def addSquaredForceCost(self, k, weight):
        cost = ssqpy.model.costs.SquaredForceCost(self.model, weight)
        self.model.addCost(k, cost)

    def addSquaredConfigurationErrorCost(self, k, desired_config, weight):
        cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
            self.model, desired_config, weight
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
            translation_weight,
            rotation_weight,
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

        self.solver = ssqpy.solvers.ssqp(self.model, ssqp_params)

        self.solver.solve()

    def save(self, output_json):
        self.solver.write(output_json)
