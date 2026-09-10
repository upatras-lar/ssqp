import json
import time

# 1. Import pinocchio first to avoid C++ symbol/allocator crashes
import pinocchio as pin

import crocoddyl
import mim_solvers
import numpy as np
from ProblemWrapper import ProblemWrapper


class PendubotActuationModel(crocoddyl.ActuationModelAbstract):
    """One-input pendubot actuation, matching SSQP's undamped ABA model."""

    def __init__(self, state, model):
        super().__init__(state, nu=1)
        self.model = model
        self.B = np.array([[1.0], [0.0]])
        self.u_lb = np.array([-model.effortLimit[0]])
        self.u_ub = np.array([model.effortLimit[0]])

    def calc(self, data, x, u):
        data.tau[:] = self.B @ u

    def calcDiff(self, data, x, u):
        data.dtau_dx[:] = 0.0
        data.dtau_du[:] = 0.0
        data.dtau_du[:] = self.B[:, 0]


class MimSolversProblem(ProblemWrapper):

    def __init__(self, horizon, dt, urdf_path, parameters=None):
        super().__init__(horizon, dt, urdf_path)

        parameters = parameters or {}

        self.H = horizon
        self.dt = dt
        self.urdf_path = urdf_path

        # ---------------------------------------------------------------------
        # 1. Detect & Load Pinocchio Model
        # ---------------------------------------------------------------------
        # Check if explicitly requested as floating-base or detect from parameter
        is_floating_base = parameters.get("floating_base", True)

        try:
            if is_floating_base:
                self.model = pin.buildModelFromUrdf(
                    urdf_path, pin.JointModelFreeFlyer()
                )
            else:
                self.model = pin.buildModelFromUrdf(urdf_path)
        except RuntimeError:
            # Fallback if building with FreeFlyer fails on fixed-base URDFs
            self.model = pin.buildModelFromUrdf(urdf_path)
            is_floating_base = False

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.state = crocoddyl.StateMultibody(self.model)

        # ---------------------------------------------------------------------
        # 2. Dynamic Actuation Handling (Custom vs. Floating Base vs. Full)
        # ---------------------------------------------------------------------
        if "actuation_model" in parameters:
            # Pass custom user-defined actuation (e.g. pendubot underactuation)
            self.actuation = parameters["actuation_model"]
        elif (
            is_floating_base
            and self.model.joints[1].shortname() == "JointModelFreeFlyer"
        ):
            self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        else:
            self.actuation = crocoddyl.ActuationModelFull(self.state)

        self.nu = self.actuation.nu

        # Default initial state vector
        self.x0 = parameters.get("x0", self.state.zero())

        # Forward kinematics reference data for frame locations/contacts
        self.pin_data = self.model.createData()

        # ---------------------------------------------------------------------
        # 3. Stage Containers (0 to H)
        # ---------------------------------------------------------------------
        self.stage_costs = [
            crocoddyl.CostModelSum(self.state, self.nu)
            for _ in range(self.H + 1)
        ]
        self.stage_constraints = [
            crocoddyl.ConstraintModelManager(self.state, self.nu)
            for _ in range(self.H + 1)
        ]
        self.stage_contacts = [
            crocoddyl.ContactModelMultiple(self.state, self.nu)
            for _ in range(self.H + 1)
        ]
        # The shared benchmark registers force costs before registering the
        # contacts they belong to, so keep those costs pending until both
        # objects exist
        self._force_cost_weights = {}
        self._force_cost_count = {}
        self.friction_coeff = float(parameters.get("friction_coefficient", 0.6))

        # Pre-allocate URDF-derived position, velocity, and torque bounds
        self._setup_default_bounds()

        self.solver = None

    def _setup_default_bounds(self):
        """Builds default velocity, position, and control limit constraints."""
        state_residual = crocoddyl.ResidualModelState(
            self.state, self.state.zero(), self.nu
        )

        # Velocity bounds
        velocity_limit = np.asarray(
            self.model.velocityLimit, dtype=float
        ).reshape(-1)
        lb_dq = -np.inf * np.ones(self.nv)
        ub_dq = np.inf * np.ones(self.nv)
        if self.model.joints[1].shortname() == "JointModelFreeFlyer":
            lb_dq[6:] = -velocity_limit[6:]
            ub_dq[6:] = velocity_limit[6:]
        else:
            lb_dq[:] = -velocity_limit
            ub_dq[:] = velocity_limit
        nq_ndx = self.state.ndx - self.state.nv
        lb_v = np.concatenate([-np.inf * np.ones(nq_ndx), lb_dq])
        ub_v = np.concatenate([np.inf * np.ones(nq_ndx), ub_dq])
        self.velocity_constraint = crocoddyl.ConstraintModelResidual(
            self.state, state_residual, lb_v, ub_v
        )

        # State residuals are expressed in tangent coordinates, so leave the
        # floating-base coordinates unbounded, and apply URDF position limits
        # only to the joint coordinates, just like SSQP/CasADi
        lb_q = -np.inf * np.ones(self.state.ndx)
        ub_q = np.inf * np.ones(self.state.ndx)
        if self.model.joints[1].shortname() == "JointModelFreeFlyer":
            q_lb = np.asarray(self.model.lowerPositionLimit, dtype=float)[7:]
            q_ub = np.asarray(self.model.upperPositionLimit, dtype=float)[7:]
            lb_q[6 : 6 + self.nv - 6] = q_lb
            ub_q[6 : 6 + self.nv - 6] = q_ub
        else:
            q_lb = np.asarray(self.model.lowerPositionLimit, dtype=float)[: self.nv]
            q_ub = np.asarray(self.model.upperPositionLimit, dtype=float)[: self.nv]
            lb_q[: self.nv] = q_lb
            ub_q[: self.nv] = q_ub
        self.position_constraint = crocoddyl.ConstraintModelResidual(
            self.state, state_residual, lb_q, ub_q
        )

        # Control bounds
        if self.nu > 0:
            effort = np.asarray(self.model.effortLimit, dtype=float).reshape(-1)
            if self.model.joints[1].shortname() == "JointModelFreeFlyer":
                effort = effort[-self.nu :]
            else:
                effort = effort[: self.nu]
            lb_u = -effort
            ub_u = effort
            control_residual = crocoddyl.ResidualModelControl(
                self.state, self.nu
            )
            self.control_constraint = crocoddyl.ConstraintModelResidual(
                self.state, control_residual, lb_u, ub_u
            )

    # -------------------------------------------------------------------------
    # Setup / Finalize API
    # -------------------------------------------------------------------------

    def finalize(self, parameters=None):
        parameters = parameters or {}
        if "init_states" in parameters:
            self.x0 = np.asarray(parameters["init_states"], dtype=float)

        q0 = self.x0[: self.nq]
        pin.forwardKinematics(self.model, self.pin_data, q0)
        pin.updateFramePlacements(self.model, self.pin_data)
        # Contact registration happens before finalize() in the shared
        # problem definitions, so refresh the stabilization reference from
        # the actual initial state instead of retaining the constructor-time
        # placeholder; gains are zero for these benchmarks, but keeping the
        # reference correct makes the wrapper safe for nonzero gains too
        for contacts in self.stage_contacts:
            for item in contacts.contacts.todict().values():
                frame_id = item.contact.id
                item.contact.reference = self.pin_data.oMf[
                    frame_id
                ].translation.copy()

    # -------------------------------------------------------------------------
    # Cost & Constraint Interfaces
    # -------------------------------------------------------------------------

    def addSquaredConfigurationErrorCost(self, k, desired_config, weight):
        # crocoddyl's IntegratedActionModelEuler multiplies a running node's
        # (k < H) total cost by dt, but leaves the terminal node's (k == H)
        # cost unscaled (see euler.hxx: the calc(x) overload used for the
        # terminal node never applies time_step_), so pre-dividing by dt on
        # running nodes cancels that scaling and this ends up matching the
        # other solver formulations' shared 0.5 * weight * ||e||^2 convention
        if k < self.H:
            weight = weight / self.dt
        xref = self.x0.copy()
        if (
            len(desired_config) == self.nv - 6
            and self.model.joints[1].shortname() == "JointModelFreeFlyer"
        ):
            xref[7 : self.nq] = desired_config
            config_weights = np.concatenate(
                [np.zeros(6), weight * np.ones(self.nv - 6), np.zeros(self.nv)]
            )
        else:
            xref[: len(desired_config)] = desired_config
            config_weights = np.concatenate(
                [weight * np.ones(self.nv), np.zeros(self.nv)]
            )

        activation = crocoddyl.ActivationModelWeightedQuad(config_weights)
        cost = crocoddyl.CostModelResidual(
            self.state,
            activation,
            crocoddyl.ResidualModelState(self.state, xref, self.nu),
        )
        self.stage_costs[k].addCost(f"config_err_{k}", cost, 1.0)

    def addSquaredJointVelocityCost(self, k, weight):
        # See the dt-scaling note in addSquaredConfigurationErrorCost
        if k < self.H:
            weight = weight / self.dt
        velocity_weights = np.zeros(2 * self.nv)
        velocity_start = self.nv + (
            6
            if self.model.joints[1].shortname() == "JointModelFreeFlyer"
            else 0
        )
        velocity_weights[velocity_start:] = weight
        activation = crocoddyl.ActivationModelWeightedQuad(velocity_weights)
        cost = crocoddyl.CostModelResidual(
            self.state,
            activation,
            crocoddyl.ResidualModelState(self.state, self.x0, self.nu),
        )
        self.stage_costs[k].addCost(f"vel_{k}", cost, 1.0)

    def addSquaredJointAngleCost(self, k, weight):
        # See the dt-scaling note in addSquaredConfigurationErrorCost
        if k < self.H:
            weight = weight / self.dt
        is_floating_base = (
            self.model.joints[1].shortname() == "JointModelFreeFlyer"
        )
        if is_floating_base:
            # This is a zero-reference regularizer in SSQP, and the desired
            # configuration is supplied by the separate tracking cost
            xref = self.x0.copy()
            xref[7 : self.nq] = 0.0
            config_weights = np.concatenate(
                [np.zeros(6), weight * np.ones(self.nv - 6), np.zeros(self.nv)]
            )
        else:
            xref = np.zeros_like(self.x0)
            config_weights = np.concatenate(
                [weight * np.ones(self.nv), np.zeros(self.nv)]
            )
        activation = crocoddyl.ActivationModelWeightedQuad(config_weights)
        cost = crocoddyl.CostModelResidual(
            self.state,
            activation,
            crocoddyl.ResidualModelState(self.state, xref, self.nu),
        )
        self.stage_costs[k].addCost(f"joint_angle_{k}", cost, 1.0)

    def addSquaredFramePoseErrorCost(
        self,
        k,
        frame_name,
        desired_translation,
        desired_rotation,
        translation_weight,
        rotation_weight,
    ):
        # See the dt-scaling note in addSquaredConfigurationErrorCost
        if k < self.H:
            translation_weight = translation_weight / self.dt
            rotation_weight = rotation_weight / self.dt
        frame_id = self.model.getFrameId(frame_name)
        M_des = pin.SE3(
            np.asarray(desired_rotation, dtype=float),
            np.asarray(desired_translation, dtype=float),
        )
        residual = crocoddyl.ResidualModelFramePlacement(
            self.state, frame_id, M_des, self.nu
        )
        weights = np.concatenate(
            [
                translation_weight * np.ones(3),
                rotation_weight * np.ones(3),
            ]
        )
        activation = crocoddyl.ActivationModelWeightedQuad(weights)
        cost = crocoddyl.CostModelResidual(self.state, activation, residual)
        self.stage_costs[k].addCost(f"frame_pose_{frame_name}_{k}", cost, 1.0)

    def addSquaredControlCost(self, k, weight):
        if k >= self.H:
            return
        # Always a running node here (see the dt-scaling note in
        # addSquaredConfigurationErrorCost)
        weight = weight / self.dt
        cost = crocoddyl.CostModelResidual(
            self.state, crocoddyl.ResidualModelControl(self.state, self.nu)
        )
        self.stage_costs[k].addCost(f"u_{k}", cost, weight)

    def addSquaredForceCost(self, k, weight):
        if k >= self.H:
            return
        self._force_cost_weights.setdefault(k, []).append(weight)
        for name, item in self.stage_contacts[k].contacts.todict().items():
            self._add_force_cost(k, name, item.contact, weight)

    def _add_force_cost(self, k, name, contact, weight):
        """Add a force cost with Crocoddyl's running-node dt factor removed."""
        force_residual = crocoddyl.ResidualModelContactForce(
            self.state,
            contact.id,
            pin.Force.Zero(),
            3,
            self.nu,
            True,
        )
        force_cost = crocoddyl.CostModelResidual(
            self.state, crocoddyl.ActivationModelQuad(3), force_residual
        )
        count = self._force_cost_count.get(k, 0)
        self._force_cost_count[k] = count + 1
        self.stage_costs[k].addCost(
            f"{name}_force_{k}_{count}",
            force_cost,
            weight / self.dt,
        )

    def addContactConstraint(self, k, frame_name, max_normal_force, tol):
        frame_id = self.model.getFrameId(frame_name)
        contact_pos = self.pin_data.oMf[frame_id].translation.copy()
        contact = crocoddyl.ContactModel3D(
            self.state,
            frame_id,
            contact_pos,
            pin.LOCAL_WORLD_ALIGNED,
            self.nu,
        )
        self.stage_contacts[k].addContact(frame_name, contact)

        if k < self.H:
            # Match SSQP's four-facet Coulomb pyramid in world-aligned axes,
            # since crocoddyl's inner approximation scales mu by cos(pi/4),
            # so use the outer/raw coefficient to preserve the shared
            # constraint
            cone = crocoddyl.FrictionCone(
                np.eye(3), self.friction_coeff, 4, False, 0.0, max_normal_force
            )
            cone_residual = crocoddyl.ResidualModelContactFrictionCone(
                self.state,
                frame_id,
                cone,
                self.nu,
                True,
            )
            cone_lb = np.asarray(
                cone.lb if hasattr(cone, "lb") else cone.get_lb(), dtype=float
            )
            cone_ub = np.asarray(
                cone.ub if hasattr(cone, "ub") else cone.get_ub(), dtype=float
            )
            friction_constraint = crocoddyl.ConstraintModelResidual(
                self.state, cone_residual, cone_lb, cone_ub
            )
            self.stage_constraints[k].addConstraint(
                f"{frame_name}_friction_constraint_{k}", friction_constraint
            )

        # The shared problem adds force costs before this contact call
        for weight in self._force_cost_weights.get(k, []):
            self._add_force_cost(k, frame_name, contact, weight)

        # `tol` (SSQP/CasADi's contact-height band) is intentionally unused
        # here, since ContactModel3D enforces the registered contact through
        # the constrained/reduced forward dynamics itself; there is no
        # separate KKT-level height equality for a tolerance to loosen, and
        # the paired linear-velocity constraint remains an exact zero bound,
        # matching SSQP, CasADi, and Aligator

    def addFrameLinearVelocityBound(self, k, frame_name, lb_vel, ub_vel):
        frame_id = self.model.getFrameId(frame_name)
        residual = crocoddyl.ResidualModelFrameVelocity(
            self.state,
            frame_id,
            pin.Motion.Zero(),
            pin.LOCAL_WORLD_ALIGNED,
            self.nu,
        )

        # Pinocchio Motion vectors are ordered [linear, angular], so bound
        # the translational slice and leave the angular residual
        # unconstrained
        lb_full = np.array(
            [lb_vel[0], lb_vel[1], lb_vel[2], -np.inf, -np.inf, -np.inf]
        )
        ub_full = np.array(
            [ub_vel[0], ub_vel[1], ub_vel[2], np.inf, np.inf, np.inf]
        )
        constraint = crocoddyl.ConstraintModelResidual(
            self.state, residual, lb_full, ub_full
        )

        self.stage_constraints[k].addConstraint(
            f"{frame_name}_vel_constraint_{k}", constraint
        )

    def addFrameTranslationBound(self, k, frame_name, lb_pos, ub_pos):
        frame_id = self.model.getFrameId(frame_name)
        residual = crocoddyl.ResidualModelFrameTranslation(
            self.state, frame_id, np.zeros(3), self.nu
        )
        constraint = crocoddyl.ConstraintModelResidual(
            self.state,
            residual,
            np.asarray(lb_pos, dtype=float),
            np.asarray(ub_pos, dtype=float),
        )
        self.stage_constraints[k].addConstraint(
            f"{frame_name}_translation_constraint_{k}", constraint
        )

    # -------------------------------------------------------------------------
    # Solve Routing
    # -------------------------------------------------------------------------

    def solve(self, parameters=None):
        parameters = parameters or {}
        running_models = []

        # Determine whether to use Contact Action Model or Free Forward Dynamics
        has_contacts = any(
            len(c.contacts) > 0 for c in self.stage_contacts[: self.H]
        )

        for k in range(self.H):
            self.stage_constraints[k].addConstraint(
                "q_bounds", self.position_constraint
            )
            self.stage_constraints[k].addConstraint(
                "vel_bounds", self.velocity_constraint
            )
            if self.nu > 0:
                self.stage_constraints[k].addConstraint(
                    "u_bounds", self.control_constraint
                )

            if has_contacts:
                dam = crocoddyl.DifferentialActionModelContactFwdDynamics(
                    self.state,
                    self.actuation,
                    self.stage_contacts[k],
                    self.stage_costs[k],
                    self.stage_constraints[k],
                    0.0,
                    True,
                )
            else:
                dam = crocoddyl.DifferentialActionModelFreeFwdDynamics(
                    self.state,
                    self.actuation,
                    self.stage_costs[k],
                    self.stage_constraints[k],
                )

            running_models.append(
                crocoddyl.IntegratedActionModelEuler(dam, self.dt)
            )

        # Terminal stage
        self.stage_constraints[self.H].addConstraint(
            "q_bounds", self.position_constraint
        )
        self.stage_constraints[self.H].addConstraint(
            "vel_bounds", self.velocity_constraint
        )

        if has_contacts:
            terminal_dam = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state,
                self.actuation,
                self.stage_contacts[self.H],
                self.stage_costs[self.H],
                self.stage_constraints[self.H],
                0.0,
                True,
            )
        else:
            terminal_dam = crocoddyl.DifferentialActionModelFreeFwdDynamics(
                self.state,
                self.actuation,
                self.stage_costs[self.H],
                self.stage_constraints[self.H],
            )

        terminal_model = crocoddyl.IntegratedActionModelEuler(
            terminal_dam, 0.0
        )

        # Solve via mim_solvers CSQP
        self.problem = crocoddyl.ShootingProblem(
            self.x0, running_models, terminal_model
        )
        self.solver = mim_solvers.SolverCSQP(self.problem)
        self.solver.termination_tolerance = parameters.get("tolerance", 1e-2)
        self.solver.max_qp_iters = parameters.get("max_qp_iters", 100)

        self.solver.eps_abs = parameters.get("eps_abs", 1e-3)
        self.solver.eps_rel = parameters.get("eps_rel", 1e-3)
        self.solver.adaptive_rho_tolerance = parameters.get(
            "adaptive_rho_tolerance", 1
        )
        self.solver.rho_update_interval = parameters.get(
            "rho_update_interval", 25
        )
        self.solver.use_filter_line_search = parameters.get(
            "use_filter_line_search", True
        )

        # Callbacks add per-iteration logging and bookkeeping to the timed
        # solve, so keep them opt-in for benchmark runs, while retaining the
        # original diagnostics for interactive troubleshooting
        if parameters.get("callbacks", False):
            self.solver.setCallbacks(
                [
                    mim_solvers.CallbackVerbose(),
                    mim_solvers.CallbackLogger(),
                ]
            )

        xs_init = [self.x0.copy() for _ in range(self.H + 1)]
        us_init = [np.zeros(self.nu) for _ in range(self.H)]

        start = time.perf_counter()
        success = self.solver.solve(
            xs_init, us_init, parameters.get("max_iter", 1000)
        )
        end = time.perf_counter()

        self.solve_time = end - start
        self.converged = bool(success)
        self.iterations = int(self.solver.iter)
        self.constraint_violation = self._evaluate_violation()

        print(f"[mim_solvers] Solve time: {self.solve_time * 1000:.3f} ms")
        print(f"[mim_solvers] Final cost: {self.solver.cost}")
        print(f"[mim_solvers] Iterations: {self.solver.iter}")
        print(f"[mim_solvers] Success: {success}")

    def _evaluate_violation(self):
        """Sum, over stages, of the worst single bound violation at that
        stage -- matching SSQP's trajectory_constraint_violation (see
        ssqp::_cf in src/solvers/ssqp.cpp): per stage, take the infinity
        norm of how far outside [lb, ub] the (possibly vector-valued)
        constraint sits, then sum that per-stage worst case across stages.

        Crocoddyl's shooting rollout satisfies the dynamics exactly by
        construction, so unlike CasadiProblem there is no dynamics-defect
        term to fold in here -- only the path constraints each
        ConstraintModelManager already aggregates into g/g_lb/g_ub.
        Recomputed explicitly on the reported xs/us rather than trusting
        whatever trial the solver's internal data was last evaluated at.
        """
        self.problem.calc(list(self.solver.xs), list(self.solver.us))

        total = 0.0
        for k in range(self.H + 1):
            if k < self.H:
                diff_data = self.problem.runningDatas[k].differential
            else:
                diff_data = self.problem.terminalData.differential
            cm = self.stage_constraints[k]
            if cm.ng == 0 and cm.nh == 0:
                continue
            g = np.asarray(diff_data.constraints.g, dtype=float)
            lb = np.asarray(cm.g_lb, dtype=float)
            ub = np.asarray(cm.g_ub, dtype=float)
            violation = np.maximum(0.0, np.maximum(lb - g, g - ub))
            if violation.size:
                total += float(violation.max())
        return total

    # -------------------------------------------------------------------------
    # JSON Serialization Method
    # -------------------------------------------------------------------------

    def save(self, output_json):
        """Saves trajectory and contact forces to JSON matching the visualizer schema."""
        if self.solver is None:
            raise RuntimeError("Must call solve() before saving the solution.")

        x_sol = np.asarray(self.solver.xs)
        u_sol = np.asarray(self.solver.us)

        is_floating = self.model.joints[1].shortname() == "JointModelFreeFlyer"
        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        running_datas = self.problem.runningDatas

        stage_data = []
        stage_force_frames = []
        for k in range(self.H + 1):
            stage_forces = []
            force_frames = []
            if k < self.H:
                diff_data = running_datas[k].differential

                # Extract forces dynamically if contact data is present
                if hasattr(diff_data, "multibody") and hasattr(
                    diff_data.multibody, "contacts"
                ):
                    contact_data = diff_data.multibody.contacts
                    for name in foot_names:
                        if name in contact_data.contacts:
                            foot_data = contact_data.contacts[name]
                            force_world = np.asarray(
                                foot_data.f.linear, dtype=float
                            ).copy()
                            q_k = np.asarray(x_sol[k][: self.nq])
                            pin.forwardKinematics(
                                self.model, self.pin_data, q_k
                            )
                            pin.updateFramePlacements(
                                self.model, self.pin_data
                            )
                            R_world_local = self.pin_data.oMf[
                                self.model.getFrameId(name)
                            ].rotation
                            force_local = R_world_local.T @ force_world
                            stage_forces.extend(force_local.tolist())
                            force_frames.append(name)

            if is_floating:
                stage = {
                    "position": x_sol[k][:3].tolist(),
                    "orientation": x_sol[k][3:7].tolist(),
                    "linear_velocity": x_sol[k][
                        self.nq : self.nq + 3
                    ].tolist(),
                    "angular_velocity": x_sol[k][
                        self.nq + 3 : self.nq + 6
                    ].tolist(),
                    "joint_configuration": x_sol[k][7 : self.nq].tolist(),
                    "joint_velocity": x_sol[k][
                        self.nq + 6 : self.nq + self.nv
                    ].tolist(),
                    "forces": stage_forces,
                }
                if k < self.H:
                    stage["control"] = u_sol[k][: self.nu].tolist()
            else:
                # Fixed-base fallback format (e.g. Pendubot)
                stage = {
                    "state": x_sol[k].tolist(),
                }
                if k < self.H:
                    stage["control"] = u_sol[k][: self.nu].tolist()
            stage_data.append(stage)
            stage_force_frames.append(force_frames)

        output = {
            "benchmark": {
                "format_version": 1,
                "solver": "mim_solvers",
                "success": self.converged,
                "iterations": self.iterations,
                "runtime_ms": self.solve_time * 1000.0,
                "objective": float(self.solver.cost),
                "constraint_violation": self.constraint_violation,
            },
            "properties": {
                "horizon": self.H,
                "dt": self.dt,
                "solver_mode": "ForwardDynamics",
                "xml": True,
                "nq": self.nq,
                "nv": self.nv,
                "nu": self.nu,
                "njoints": self.model.njoints - 1,
                "total_joints": self.model.njoints,
                "floating_base": is_floating,
                "state_joints": [
                    self.model.names[i]
                    for i in range(2 if is_floating else 1, self.model.njoints)
                ],
                "joint_types": [
                    "revolute"
                    for _ in range(
                        self.model.njoints - (2 if is_floating else 1)
                    )
                ],
                "stages": [],
            },
            "stage_data": stage_data,
        }

        for k in range(self.H + 1):
            force_frames = stage_force_frames[k]
            output["properties"]["stages"].append(
                {
                    "size": self.nq + self.nv,
                    "control": self.nu if k < self.H else 0,
                    "nforces": len(force_frames),
                    "ordered_force_frames": force_frames,
                }
            )

        with open(output_json, "w") as f:
            json.dump(output, f, indent=4)

        print(f"[mim_solvers] Saved solution to {output_json}")
