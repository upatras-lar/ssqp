import json
import time

# 1. ALWAYS import pinocchio first to avoid C++ symbol/allocator crashes
import pinocchio as pin

import crocoddyl
import mim_solvers
import numpy as np
from ProblemWrapper import ProblemWrapper


class PendubotActuationModel(crocoddyl.ActuationModelAbstract):
    """Actuation model with viscous damping matching the URDF specification."""

    def __init__(self, state, model):
        super().__init__(state, nu=1)
        self.model = model
        self.B = np.array([[1.0], [0.0]])
        self.damping = np.asarray(model.damping, dtype=float).reshape(-1)
        self.u_lb = np.array([-model.effortLimit[0]])
        self.u_ub = np.array([model.effortLimit[0]])

    def calc(self, data, x, u):
        v = x[self.model.nq :]
        data.tau[:] = self.B @ u - self.damping * v

    def calcDiff(self, data, x, u):
        data.dtau_dx[:] = 0.0
        data.dtau_du[:] = 0.0
        for i in range(self.model.nv):
            data.dtau_dx[i, self.model.nq + i] = -self.damping[i]
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
            # Pass custom user-defined actuation (e.g., Pendubot viscous damping)
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

        # Pre-allocate URDF-derived position, velocity, and torque bounds
        self._setup_default_bounds()

        self.solver = None

    def _setup_default_bounds(self):
        """Builds default velocity, position, and control limit constraints."""
        state_residual = crocoddyl.ResidualModelState(
            self.state, self.state.zero(), self.nu
        )

        # Velocity bounds
        lb_dq = -np.asarray(self.model.velocityLimit, dtype=float).reshape(-1)
        ub_dq = np.asarray(self.model.velocityLimit, dtype=float).reshape(-1)
        nq_ndx = self.state.ndx - self.state.nv
        lb_v = np.concatenate([-np.inf * np.ones(nq_ndx), lb_dq])
        ub_v = np.concatenate([np.inf * np.ones(nq_ndx), ub_dq])
        self.velocity_constraint = crocoddyl.ConstraintModelResidual(
            self.state, state_residual, lb_v, ub_v
        )

        # Control bounds
        if self.nu > 0:
            lb_u = -np.asarray(self.model.effortLimit, dtype=float).reshape(
                -1
            )[: self.nu]
            ub_u = np.asarray(self.model.effortLimit, dtype=float).reshape(-1)[
                : self.nu
            ]
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

    # -------------------------------------------------------------------------
    # Cost & Constraint Interfaces
    # -------------------------------------------------------------------------

    def addSquaredConfigurationErrorCost(self, k, desired_config, weight):
        # crocoddyl's IntegratedActionModelEuler multiplies a running node's
        # (k < H) total cost by dt, but leaves the terminal node's (k == H)
        # cost unscaled (see euler.hxx: the calc(x) overload used for the
        # terminal node never applies time_step_). Pre-dividing by dt on
        # running nodes cancels that scaling so this ends up matching the
        # other solver formulations' shared 0.5 * weight * ||e||^2 convention.
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
        # See the dt-scaling note in addSquaredConfigurationErrorCost.
        if k < self.H:
            weight = weight / self.dt
        velocity_weights = np.concatenate(
            [np.zeros(self.nv), weight * np.ones(self.nv)]
        )
        activation = crocoddyl.ActivationModelWeightedQuad(velocity_weights)
        cost = crocoddyl.CostModelResidual(
            self.state,
            activation,
            crocoddyl.ResidualModelState(self.state, self.x0, self.nu),
        )
        self.stage_costs[k].addCost(f"vel_{k}", cost, 1.0)

    def addSquaredJointAngleCost(self, k, weight):
        # See the dt-scaling note in addSquaredConfigurationErrorCost.
        if k < self.H:
            weight = weight / self.dt
        is_floating_base = (
            self.model.joints[1].shortname() == "JointModelFreeFlyer"
        )
        if is_floating_base:
            config_weights = np.concatenate(
                [np.zeros(6), weight * np.ones(self.nv - 6), np.zeros(self.nv)]
            )
        else:
            config_weights = np.concatenate(
                [weight * np.ones(self.nv), np.zeros(self.nv)]
            )
        activation = crocoddyl.ActivationModelWeightedQuad(config_weights)
        cost = crocoddyl.CostModelResidual(
            self.state,
            activation,
            crocoddyl.ResidualModelState(self.state, self.x0, self.nu),
        )
        self.stage_costs[k].addCost(f"joint_angle_{k}", cost, 1.0)

    def addSquaredControlCost(self, k, weight):
        if k >= self.H:
            return
        # Always a running node here (see the dt-scaling note in
        # addSquaredConfigurationErrorCost).
        weight = weight / self.dt
        cost = crocoddyl.CostModelResidual(
            self.state, crocoddyl.ResidualModelControl(self.state, self.nu)
        )
        self.stage_costs[k].addCost(f"u_{k}", cost, weight)

    def addSquaredForceCost(self, k, weight):
        if k >= self.H:
            return
        weight = weight / self.dt
        for name, contact in self.stage_contacts[k].contacts:
            force_residual = crocoddyl.ResidualModelContactForce(
                self.state,
                contact.contact.id,
                pin.Force.Zero(),
                3,
                self.nu,
                True,
            )
            force_cost = crocoddyl.CostModelResidual(
                self.state, crocoddyl.ActivationModelQuad(3), force_residual
            )
            self.stage_costs[k].addCost(
                f"{name}_force_{k}", force_cost, weight
            )

    def addContactConstraint(self, k, frame_name, max_normal_force, tol):
        frame_id = self.model.getFrameId(frame_name)
        contact_pos = self.pin_data.oMf[frame_id].translation.copy()
        contact = crocoddyl.ContactModel3D(
            self.state, frame_id, contact_pos, pin.LOCAL, self.nu
        )
        self.stage_contacts[k].addContact(frame_name, contact)

        if k < self.H:
            force_residual = crocoddyl.ResidualModelContactForce(
                self.state, frame_id, pin.Force.Zero(), 3, self.nu, True
            )
            force_constraint = crocoddyl.ConstraintModelResidual(
                self.state,
                force_residual,
                np.array([-np.inf, -np.inf, 0.0]),
                np.array([np.inf, np.inf, max_normal_force]),
            )
            self.stage_constraints[k].addConstraint(
                f"{frame_name}_force_constraint_{k}", force_constraint
            )

    def addFrameLinearVelocityBound(self, k, frame_name, lb_vel, ub_vel):
        frame_id = self.model.getFrameId(frame_name)
        residual = crocoddyl.ResidualModelFrameVelocity(
            self.state,
            frame_id,
            pin.Motion.Zero(),
            pin.LOCAL_WORLD_ALIGNED,
            self.nu,
        )

        if np.allclose(lb_vel, 0.0) and np.allclose(ub_vel, 0.0):
            tol = 0.01
            constraint = crocoddyl.ConstraintModelResidual(
                self.state, residual, -tol * np.ones(6), tol * np.ones(6)
            )
        else:
            lb_full = np.array(
                [-np.inf, -np.inf, -np.inf, lb_vel[0], lb_vel[1], lb_vel[2]]
            )
            ub_full = np.array(
                [np.inf, np.inf, np.inf, ub_vel[0], ub_vel[1], ub_vel[2]]
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

        self.solver.setCallbacks(
            [
                mim_solvers.CallbackVerbose(),
                mim_solvers.CallbackLogger(),
            ]
        )

        xs_init = [self.x0.copy() for _ in range(self.H + 1)]
        us_init = [np.zeros(self.nu) for _ in range(self.H)]

        start = time.time()
        success = self.solver.solve(
            xs_init, us_init, parameters.get("max_iter", 1000)
        )
        end = time.time()

        print(f"Solve time: {(end - start) * 1000:.3f} ms")
        print(f"Final cost: {self.solver.cost}")
        print(f"Iterations: {self.solver.iter}")
        print(f"Success: {success}")

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
        for k in range(self.H):
            diff_data = running_datas[k].differential

            # Extract forces dynamically if contact data is present
            stage_forces = []
            if hasattr(diff_data, "multibody") and hasattr(
                diff_data.multibody, "contacts"
            ):
                contact_data = diff_data.multibody.contacts
                for name in foot_names:
                    if name in contact_data.contacts:
                        foot_data = contact_data.contacts[name]
                        force = np.asarray(
                            foot_data.f.linear, dtype=float
                        ).copy()
                        stage_forces.extend(force.tolist())

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
                    "control": u_sol[k][: self.nu].tolist(),
                    "forces": stage_forces,
                }
            else:
                # Fixed-base fallback format (e.g. Pendubot)
                stage = {
                    "state": x_sol[k].tolist(),
                    "control": u_sol[k][: self.nu].tolist(),
                }
            stage_data.append(stage)

        output = {
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

        nforces = len(stage_data[0].get("forces", [])) if stage_data else 0
        for k in range(self.H):
            output["properties"]["stages"].append(
                {
                    "size": self.nq + self.nv,
                    "control": self.nu,
                    "nforces": nforces,
                    "ordered_force_frames": foot_names if nforces > 0 else [],
                }
            )

        with open(output_json, "w") as f:
            json.dump(output, f, indent=4)

        print(f"Saved solution to {output_json}")
