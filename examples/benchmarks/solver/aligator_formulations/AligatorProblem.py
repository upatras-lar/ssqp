"""
AligatorProblem: a ProblemWrapper implementation backed by the Aligator
trajectory-optimization solver (proxddp).

  - Fixed-base, underactuated systems (e.g. the pendubot) are handled with
    a custom forward-dynamics ODE that runs `pin.aba` on an "effective"
    torque `B @ u - damping * v`, for an arbitrary actuation matrix `B`.
  - Floating-base systems with rigid foot contacts (e.g. Go2 standing) are
    handled with aligator's `MultibodyConstraintFwdDynamics`, built per
    stage from whichever contact frames were registered via
    `addContactConstraint` at that stage.

ProblemWrapper's API is incremental: costs/constraints are registered one
timestep at a time, so AligatorProblem just *records* every call against
its node index `k` and only assembles the actual aligator StageModels /
TrajOptProblem lazily, inside finalize(). solve() calls finalize()
automatically (with an all-zero initial state) if it hasn't been called
explicitly yet.

Two modeling choices are not carried by the ProblemWrapper API and are
taken from `constructor_parameters` / sensible defaults instead:
  - the friction cone coefficient (`friction_coefficient`, default 0.6),
  - the target world-Z contact height (`foot_target_height`, default 0.0).
"""

import json
import os
from time import time

import numpy as np
import pinocchio as pin

import aligator
from aligator import constraints, dynamics, manifolds

from ProblemWrapper import ProblemWrapper


class _UnderactuatedForwardDynamics(dynamics.ODEAbstract):
    """Forward dynamics for an arbitrary actuation matrix B and (possibly
    zero) joint damping.

        ddq = ABA(q, v, B @ u - damping * v)
    """

    def __init__(self, space, pin_model, B, damping):
        super().__init__(space, B.shape[1])
        self.pin_model = pin_model
        self.pin_data = pin_model.createData()
        self.B = B
        self.damping = damping

    def __deepcopy__(self, memo):
        # All mutable state lives on the `data` object aligator passes
        # into forward()/dForward(), not on self, so sharing the same
        # instance across copies is safe -- default deepcopy would instead
        # try to reconstruct this pybind11-derived object via a no-arg
        # constructor.
        return self

    def forward(self, x, u, data):
        try:
            nq_, nv_ = self.pin_model.nq, self.pin_model.nv
            q, v = x[:nq_], x[nq_:]
            tau_eff = self.B @ u - self.damping * v
            ddq = pin.aba(self.pin_model, self.pin_data, q, v, tau_eff)
            data.xdot[:nv_] = v
            data.xdot[nv_:] = ddq
        except Exception:
            import traceback

            traceback.print_exc()
            raise

    def dForward(self, x, u, data):
        try:
            nq_, nv_ = self.pin_model.nq, self.pin_model.nv
            q, v = x[:nq_], x[nq_:]
            tau_eff = self.B @ u - self.damping * v

            pin.computeABADerivatives(
                self.pin_model, self.pin_data, q, v, tau_eff
            )
            ddq_dq = self.pin_data.ddq_dq
            ddq_dv_partial = self.pin_data.ddq_dv
            ddq_dtau = self.pin_data.Minv

            # chain rule: tau_eff depends on v through -damping * v
            ddq_dv = ddq_dv_partial - ddq_dtau @ np.diag(self.damping)
            ddq_du = ddq_dtau @ self.B

            data.Jx[:nv_, :nv_] = 0.0
            data.Jx[:nv_, nv_:] = np.eye(nv_)
            data.Jx[nv_:, :nv_] = ddq_dq
            data.Jx[nv_:, nv_:] = ddq_dv

            # boost.python squeezes data.Ju down to a 1-D array of shape
            # (ndx,) when nu == 1 (as for the pendubot's single actuated
            # joint), but keeps it 2-D, shape (ndx, nu), for nu > 1 (as
            # for the Go2's 12 actuated legs) -- handle both.
            if data.Ju.ndim == 1:
                data.Ju[:nv_] = 0.0
                data.Ju[nv_:] = ddq_du[:, 0] if ddq_du.ndim > 1 else ddq_du
            else:
                data.Ju[:nv_, :] = 0.0
                data.Ju[nv_:, :] = ddq_du
        except Exception:
            import traceback

            traceback.print_exc()
            raise


class AligatorProblem(ProblemWrapper):
    """ProblemWrapper implementation using the Aligator (proxddp) solver."""

    DEFAULT_MU = 0.6
    DEFAULT_FOOT_HEIGHT = 0.0

    def __init__(self, horizon, dt, urdf_path, constructor_parameters=None):
        super().__init__(horizon, dt, urdf_path)
        constructor_parameters = constructor_parameters or {}

        self.free_flyer = bool(constructor_parameters.get("free_flyer", False))
        self.actuated_joints = constructor_parameters.get("actuated_joints")
        self.mu = constructor_parameters.get(
            "friction_coefficient", self.DEFAULT_MU
        )
        self.foot_target_height = constructor_parameters.get(
            "foot_target_height", self.DEFAULT_FOOT_HEIGHT
        )

        if self.free_flyer:
            self.model = pin.buildModelFromUrdf(
                urdf_path, pin.JointModelFreeFlyer()
            )
        else:
            self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        self.nq, self.nv = self.model.nq, self.model.nv
        damping = np.asarray(self.model.damping).reshape(-1)
        self.damping = (
            damping if damping.size == self.nv else np.zeros(self.nv)
        )

        # Actuation matrix B (nv x nu)
        if self.free_flyer:
            self.nu = self.nv - 6
            self.B = np.zeros((self.nv, self.nu))
            self.B[6:, :] = np.eye(self.nu)
        elif self.actuated_joints is not None:
            self.nu = len(self.actuated_joints)
            self.B = np.zeros((self.nv, self.nu))
            for col, row in enumerate(self.actuated_joints):
                self.B[row, col] = 1.0
        else:
            # Fully-actuated fallback for problems that specify neither
            # "free_flyer" nor "actuated_joints".
            self.nu = self.nv
            self.B = np.eye(self.nv)

        self.space = manifolds.MultibodyPhaseSpace(self.model)
        self.ndx = self.space.ndx

        # --- per-node bookkeeping populated by the add*() calls below ---
        # self.H is the number of *stages* (control intervals); there are
        # H + 1 nodes, 0..H, with node H being the terminal one.
        self._node_costs = {k: [] for k in range(self.H + 1)}
        self._frame_vel_bounds = {k: [] for k in range(self.H + 1)}
        self._frame_pos_bounds = {k: [] for k in range(self.H + 1)}
        self._contacts = {k: [] for k in range(self.H + 1)}

        self._contact_model_cache = {}  # frame name -> RigidConstraintModel

        # Caches for objects that don't depend on k, so finalize() builds
        # each cost/constraint/dynamics object once and reuses the same
        # instance across every stage, rather than rebuilding ~H fresh
        # copies (wasteful, and adds real per-iteration solver overhead).
        self._joint_limit_cache = None
        self._vel_residual_cache = None
        self._ctrl_residual_cache = None
        self._quad_cost_cache = (
            {}
        )  # (kind, weight, dim) -> QuadraticResidualCost
        self._unconstrained_dynamics_cache = None
        self._frame_vel_residual_cache = {}
        self._frame_pos_residual_cache = {}
        self._contact_dynamics_cache = {}  # (frames, eq_tol) -> (models, prox)
        self._last_contact_dynamics = {}  # k -> cache_key, for save()

        self._built = False
        self.x0 = None
        self.problem = None
        self.solver = None
        self.xs_opt = None
        self.us_opt = None
        self.solve_time = None
        self.converged = None

    # ------------------------------------------------------------------
    # Cost / constraint registration (ProblemWrapper API)
    # ------------------------------------------------------------------
    def addSquaredJointVelocityCost(self, k, weight):
        self._node_costs[k].append(("vel", weight))

    def addSquaredControlCost(self, k, weight):
        self._node_costs[k].append(("ctrl", weight))

    def addSquaredForceCost(self, k, weight):
        self._node_costs[k].append(("force", weight))

    def addSquaredConfigurationErrorCost(self, k, desired_config, weight):
        self._node_costs[k].append(
            ("config", np.asarray(desired_config, dtype=float).copy(), weight)
        )

    def addSquaredJointAngleCost(self, k, weight):
        self._node_costs[k].append(("joint_angle", weight))

    def addSquaredFramePoseErrorCost(
        self,
        k,
        frame,
        desired_translation,
        desired_rotation,
        translation_weight,
        rotation_weight,
    ):
        self._node_costs[k].append(
            (
                "frame_pose",
                frame,
                np.asarray(desired_translation, dtype=float).copy(),
                np.asarray(desired_rotation, dtype=float).copy(),
                translation_weight,
                rotation_weight,
            )
        )

    def addFrameLinearVelocityBound(self, k, frame, lb, ub):
        self._frame_vel_bounds[k].append(
            (
                frame,
                np.asarray(lb, dtype=float).copy(),
                np.asarray(ub, dtype=float).copy(),
            )
        )

    def addFrameTranslationBound(self, k, frame, lb, ub):
        self._frame_pos_bounds[k].append(
            (
                frame,
                np.asarray(lb, dtype=float).copy(),
                np.asarray(ub, dtype=float).copy(),
            )
        )

    def addContactConstraint(self, k, frame, max_normal_force, eq_tol):
        self._contacts[k].append(
            (frame, float(max_normal_force), float(eq_tol))
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_contact_model(self, frame):
        cm = self._contact_model_cache.get(frame)
        if cm is None:
            frame_id = self.model.getFrameId(frame)
            cm = pin.RigidConstraintModel(
                pin.ContactType.CONTACT_3D,
                self.model,
                self.model.frames[frame_id].parentJoint,
                self.model.frames[frame_id].placement,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            cm.name = frame
            cm.corrector.Kp[:] = np.zeros(3)
            cm.corrector.Kd[:] = np.zeros(3)
            self._contact_model_cache[frame] = cm
        return cm

    def _build_dynamics_for_stage(self, k):
        contacts_k = self._contacts.get(k, [])
        if not contacts_k:
            # No contact set to account for -> the dynamics are identical
            # at every such stage, so build the ODE/integrator once and
            # hand back the same instance every time.
            if self._unconstrained_dynamics_cache is None:
                ode = _UnderactuatedForwardDynamics(
                    self.space, self.model, self.B, self.damping
                )
                # SemiImplEuler (v_next = v + dt*a, q_next = q (+) v_next*dt)
                # matches the other three formulations' discretization
                # (ssqpy's default SemiImplicitEuler, CasadiProblem's
                # hand-rolled v_next/q_next update, and crocoddyl's
                # IntegratedActionModelEuler) -- plain IntegratorEuler uses
                # the *old* v for the position update instead, which is a
                # different (lower-order) discrete dynamics model.
                self._unconstrained_dynamics_cache = (
                    dynamics.IntegratorSemiImplEuler(ode, self.dt)
                )
            return self._unconstrained_dynamics_cache, None, None

        frames = [c[0] for c in contacts_k]
        eq_tol = max(c[2] for c in contacts_k)
        cache_key = (tuple(sorted(frames)), eq_tol)
        cached = self._contact_dynamics_cache.get(cache_key)
        if cached is None:
            constraint_models = [self._get_contact_model(f) for f in frames]
            prox_settings = pin.ProximalSettings(eq_tol, 1e-10, 10)
            cached = (constraint_models, prox_settings)
            self._contact_dynamics_cache[cache_key] = cached
        constraint_models, prox_settings = cached

        ode = dynamics.MultibodyConstraintFwdDynamics(
            self.space, self.B, constraint_models, prox_settings
        )
        # Remember the exact (constraint_models, prox_settings) pair used
        # for this node so save() can recover forces with the *same*
        # proximal tolerance the trajectory was actually solved with,
        # rather than fabricating a different one -- re-running the
        # contact-force recovery at a tighter tolerance than what
        # produced xs_opt/us_opt makes the proximal solver do extra
        # constraint-drift correction that shows up as a spurious extra
        # force component on top of the real support force.
        self._last_contact_dynamics[k] = cache_key

        return (
            dynamics.IntegratorSemiImplEuler(ode, self.dt),
            constraint_models,
            prox_settings,
        )

    def _velocity_error_residual(self):
        if self._vel_residual_cache is not None:
            return self._vel_residual_cache
        x_ref = np.zeros(self.nq + self.nv)
        if self.free_flyer:
            x_ref[6] = 1.0
        state_err = aligator.StateErrorResidual(self.space, self.nu, x_ref)
        if self.free_flyer:
            res = state_err[self.nv + 6 : self.nv + 6 + self.nu]
        else:
            res = state_err[self.nq : self.nq + self.nv]
        self._vel_residual_cache = res
        return res

    def _control_error_residual(self):
        if self._ctrl_residual_cache is None:
            self._ctrl_residual_cache = aligator.ControlErrorResidual(
                self.ndx, self.nu
            )
        return self._ctrl_residual_cache

    def _config_error_residual(self, desired_config):
        x_ref = np.zeros(self.nq + self.nv)
        if self.free_flyer:
            x_ref[6] = 1.0  # identity quaternion w-component
            x_ref[7 : 7 + self.nu] = desired_config
            state_err = aligator.StateErrorResidual(self.space, self.nu, x_ref)
            return state_err[6 : 6 + self.nu]
        else:
            x_ref[: self.nq] = desired_config
            state_err = aligator.StateErrorResidual(self.space, self.nu, x_ref)
            return state_err[: self.nq]

    def _joint_limit_constraints(self):
        """Box constraints on velocity / torque (and, for floating-base
        models, configuration) taken directly from the URDF, applied at
        every stage -- these aren't driven by any explicit ProblemWrapper
        call. Position is never bounded. No torque constraint is added at
        the terminal node, since there's no control there to constrain
        (see `_terminal_joint_limit_constraints` below).

        Returns a dict {"config": (err, bound) | None, "vel": (err,
        bound), "torque": (err, bound)}. Cached: none of this depends on
        k, so it's built once and reused rather than reconstructed H
        times.
        """
        if self._joint_limit_cache is not None:
            return self._joint_limit_cache

        zero_state = np.zeros(self.nq + self.nv)
        if self.free_flyer:
            zero_state[6] = 1.0
        zero_residual = aligator.StateErrorResidual(
            self.space, self.nu, zero_state
        )

        eff = np.asarray(self.model.effortLimit).reshape(-1)
        if self.free_flyer:
            u_lim = eff[-self.nu :]
        elif self.actuated_joints is not None:
            u_lim = eff[self.actuated_joints]
        else:
            u_lim = eff

        torque_err = self._control_error_residual()
        torque_bound = constraints.BoxConstraint(-u_lim, u_lim)

        if self.free_flyer:
            q_lo = np.asarray(self.model.lowerPositionLimit)[-self.nu :]
            q_hi = np.asarray(self.model.upperPositionLimit)[-self.nu :]
            config_err = zero_residual[6 : 6 + self.nu]
            config_bound = constraints.BoxConstraint(q_lo, q_hi)

            v_lim = np.asarray(self.model.velocityLimit).reshape(-1)[
                -self.nu :
            ]
            vel_err = zero_residual[self.nv + 6 : self.nv + 6 + self.nu]
            vel_bound = constraints.BoxConstraint(-v_lim, v_lim)

            result = {
                "config": (config_err, config_bound),
                "vel": (vel_err, vel_bound),
                "torque": (torque_err, torque_bound),
            }
        else:
            v_lim = np.asarray(self.model.velocityLimit).reshape(-1)
            vel_err = zero_residual[self.nq : self.nq + self.nv]
            vel_bound = constraints.BoxConstraint(-v_lim, v_lim)

            result = {
                "config": None,
                "vel": (vel_err, vel_bound),
                "torque": (torque_err, torque_bound),
            }

        self._joint_limit_cache = result
        return result

    def _stage_joint_limit_constraints(self):
        """All URDF-derived box constraints (config if floating-base,
        velocity, torque), for use on ordinary stages -- these all have
        a well-defined control to constrain."""
        limits = self._joint_limit_constraints()
        return [pair for pair in limits.values() if pair is not None]

    def _terminal_joint_limit_constraints(self):
        """URDF-derived box constraints valid at the terminal node.
        Deliberately excludes "torque": the terminal node has no
        associated control variable (no transition leaves it), so a
        control-error-residual constraint there is ill-posed -- it forces
        aligator to fabricate a "control" for a node that doesn't have
        one, adding a spurious/degenerate constraint and multiplier that
        muddies the terminal KKT block and can measurably slow or worsen
        convergence."""
        limits = self._joint_limit_constraints()
        return [
            limits[name]
            for name in ("config", "vel")
            if limits[name] is not None
        ]

    def _build_cost_stack(self, k, constraint_models, prox_settings):
        cost_stack = aligator.CostStack(self.space, self.nu)
        contacts_k = self._contacts.get(k, [])

        for entry in self._node_costs.get(k, []):
            kind = entry[0]
            if kind == "vel":
                weight = entry[1]
                dim = self.nu if self.free_flyer else self.nv
                cache_key = ("vel", weight, dim)
                cost = self._quad_cost_cache.get(cache_key)
                if cost is None:
                    vel_err = self._velocity_error_residual()
                    cost = aligator.QuadraticResidualCost(
                        self.space, vel_err, weight * np.eye(dim)
                    )
                    self._quad_cost_cache[cache_key] = cost
                cost_stack.addCost(cost)
            elif kind == "ctrl":
                weight = entry[1]
                cache_key = ("ctrl", weight, self.nu)
                cost = self._quad_cost_cache.get(cache_key)
                if cost is None:
                    control_err = self._control_error_residual()
                    cost = aligator.QuadraticResidualCost(
                        self.space, control_err, weight * np.eye(self.nu)
                    )
                    self._quad_cost_cache[cache_key] = cost
                cost_stack.addCost(cost)
            elif kind == "config":
                _, desired_config, weight = entry
                # desired_config can legitimately differ per node (it
                # doesn't here, but the API allows it), so this one isn't
                # cached beyond the residual construction itself.
                config_err = self._config_error_residual(desired_config)
                dim = len(desired_config)
                cost_stack.addCost(
                    aligator.QuadraticResidualCost(
                        self.space, config_err, weight * np.eye(dim)
                    )
                )
            elif kind == "joint_angle":
                weight = entry[1]
                if self.free_flyer:
                    desired_config = self.x0[7 : 7 + self.nu]
                else:
                    desired_config = self.x0[: self.nq]
                dim = len(desired_config)
                cache_key = ("joint_angle", weight, dim)
                cost = self._quad_cost_cache.get(cache_key)
                if cost is None:
                    config_err = self._config_error_residual(desired_config)
                    cost = aligator.QuadraticResidualCost(
                        self.space, config_err, weight * np.eye(dim)
                    )
                    self._quad_cost_cache[cache_key] = cost
                cost_stack.addCost(cost)
            elif kind == "frame_pose":
                _, frame, p_des, R_des, t_wgt, r_wgt = entry
                fp_res = self._frame_placement_residual(frame, p_des, R_des)
                weight = np.diag([t_wgt] * 3 + [r_wgt] * 3)
                cost_stack.addCost(
                    aligator.QuadraticResidualCost(self.space, fp_res, weight)
                )
            elif kind == "force":
                weight = entry[1]
                if constraint_models is None:
                    # No active contacts at this node -- nothing to
                    # penalize.
                    continue
                for frame, _, _ in contacts_k:
                    force_res = aligator.ContactForceResidual(
                        self.ndx,
                        self.model,
                        self.B,
                        constraint_models,
                        prox_settings,
                        np.zeros(3),
                        frame,
                    )
                    cost_stack.addCost(
                        aligator.QuadraticResidualCost(
                            self.space, force_res, weight * np.eye(3)
                        )
                    )
        return cost_stack

    def _frame_lin_vel_residual(self, frame):
        cached = self._frame_vel_residual_cache.get(frame)
        if cached is not None:
            return cached
        frame_id = self.model.getFrameId(frame)
        fv = aligator.FrameVelocityResidual(
            self.ndx,
            self.nu,
            self.model,
            pin.Motion.Zero(),
            frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        res = fv[:3]
        self._frame_vel_residual_cache[frame] = res
        return res

    def _frame_placement_residual(self, frame, p_des, R_des):
        # Not cached: unlike _frame_translation_residual (built once with a
        # zero reference and sliced/offset by callers), the desired pose
        # here legitimately varies per node (it's a point on a trajectory),
        # so a fresh residual is built for each (frame, k) pair.
        frame_id = self.model.getFrameId(frame)
        M_des = pin.SE3(R_des, p_des)
        return aligator.FramePlacementResidual(
            self.ndx, self.nu, self.model, M_des, frame_id
        )

    def _frame_translation_residual(self, frame):
        cached = self._frame_pos_residual_cache.get(frame)
        if cached is not None:
            return cached
        frame_id = self.model.getFrameId(frame)
        fp = aligator.FrameTranslationResidual(
            self.ndx, self.nu, self.model, np.zeros(3), frame_id
        )
        self._frame_pos_residual_cache[frame] = fp
        return fp

    def _add_stage_constraints(
        self, stage, k, constraint_models, prox_settings
    ):
        for cerr, cbound in self._stage_joint_limit_constraints():
            stage.addConstraint(cerr, cbound)

        for frame, lb, ub in self._frame_vel_bounds.get(k, []):
            fv = self._frame_lin_vel_residual(frame)
            stage.addConstraint(fv, constraints.BoxConstraint(lb, ub))

        for frame, lb, ub in self._frame_pos_bounds.get(k, []):
            fp = self._frame_translation_residual(frame)
            stage.addConstraint(fp, constraints.BoxConstraint(lb, ub))

        for frame, max_normal_force, _ in self._contacts.get(k, []):
            force_res = aligator.ContactForceResidual(
                self.ndx,
                self.model,
                self.B,
                constraint_models,
                prox_settings,
                np.zeros(3),
                frame,
            )
            A_cone = np.array(
                [
                    [0.0, 0.0, -1.0],  # -f_n <= 0        -> f_n >= 0
                    [0.0, 0.0, 1.0],  # f_n <= max_normal_force
                    [1.0, 0.0, -self.mu],  # f1 - mu*f_n <= 0
                    [-1.0, 0.0, -self.mu],  # -f1 - mu*f_n <= 0
                    [0.0, 1.0, -self.mu],  # f2 - mu*f_n <= 0
                    [0.0, -1.0, -self.mu],  # -f2 - mu*f_n <= 0
                ]
            )
            b_cone = np.array([0.0, max_normal_force, 0.0, 0.0, 0.0, 0.0])
            cone_res = aligator.LinearFunctionComposition(
                force_res, A_cone, np.zeros(6)
            )
            stage.addConstraint(
                cone_res,
                constraints.BoxConstraint(-np.inf * np.ones(6), b_cone),
            )

            frame_id = self.model.getFrameId(frame)
            p_ref = np.array([0.0, 0.0, self.foot_target_height])
            fp = aligator.FrameTranslationResidual(
                self.ndx, self.nu, self.model, p_ref, frame_id
            )[2:3]
            stage.addConstraint(
                fp, constraints.BoxConstraint(np.zeros(1), np.zeros(1))
            )

    # ------------------------------------------------------------------
    # ProblemWrapper API: finalize / solve / save
    # ------------------------------------------------------------------
    def finalize(self, parameter_dict=None):
        parameter_dict = parameter_dict or {}

        init_states = parameter_dict.get("init_states")
        if init_states is not None:
            self.x0 = np.asarray(init_states, dtype=float).copy()
        elif self.x0 is None:
            self.x0 = np.zeros(self.nq + self.nv)
            if self.free_flyer:
                self.x0[6] = 1.0

        stages = []
        for k in range(self.H):
            disc_dynamics, constraint_models, prox_settings = (
                self._build_dynamics_for_stage(k)
            )
            cost_stack = self._build_cost_stack(
                k, constraint_models, prox_settings
            )
            stage = aligator.StageModel(cost_stack, disc_dynamics)
            self._add_stage_constraints(
                stage, k, constraint_models, prox_settings
            )
            stages.append(stage)

        term_cost_stack = self._build_cost_stack(self.H, None, None)
        self.problem = aligator.TrajOptProblem(
            self.x0, stages, term_cost_stack
        )

        for frame, lb, ub in self._frame_vel_bounds.get(self.H, []):
            fv = self._frame_lin_vel_residual(frame)
            self.problem.addTerminalConstraint(
                fv, constraints.BoxConstraint(lb, ub)
            )

        for frame, lb, ub in self._frame_pos_bounds.get(self.H, []):
            fp = self._frame_translation_residual(frame)
            self.problem.addTerminalConstraint(
                fp, constraints.BoxConstraint(lb, ub)
            )

        for cerr, cbound in self._terminal_joint_limit_constraints():
            self.problem.addTerminalConstraint(cerr, cbound)

        self._built = True

    def solve(self, parameter_dict=None):
        parameter_dict = parameter_dict or {}
        if not self._built:
            self.finalize({})

        tol = parameter_dict.get("tol", parameter_dict.get("tolerance", 1e-2))
        mu_init = parameter_dict.get("mu_init", 1e-2)
        max_iters = parameter_dict.get("max_iters", 1000)
        verbose = parameter_dict.get("verbose", aligator.VerboseLevel.VERBOSE)

        self.solver = aligator.SolverProxDDP(tol, mu_init, verbose=verbose)
        self.solver.max_iters = max_iters
        self.solver.setup(self.problem)

        xs_init = [self.x0.copy() for _ in range(self.H + 1)]
        us_init = [np.zeros(self.nu) for _ in range(self.H)]

        start = time()
        self.converged = self.solver.run(self.problem, xs_init, us_init)
        self.solve_time = time() - start

        results = self.solver.results
        self.xs_opt = results.xs
        self.us_opt = results.us

        print(
            f"[AligatorProblem] Solve time: {self.solve_time:.3f} s, "
            f"converged: {self.converged}"
        )
        return self.converged

    def save(self, output_json):
        if self.xs_opt is None:
            raise RuntimeError("save() called before solve()")

        stage_data = []
        if self.free_flyer:
            # Build one ContactForceResidual set per distinct
            # (frames, eq_tol) combination actually used during solve --
            # reusing the *exact* constraint_models/prox_settings objects
            # cached by _build_dynamics_for_stage, not fresh ones. Forces
            # must be recovered with the same proximal tolerance the
            # trajectory was solved with; a different tolerance makes the
            # proximal solver perform extra constraint-drift correction
            # that shows up as a spurious extra force component.
            all_frames = sorted(self._contact_model_cache.keys())
            force_residuals_by_key = {}
            for cache_key, (
                constraint_models,
                prox_settings,
            ) in self._contact_dynamics_cache.items():
                frames_k = list(cache_key[0])
                force_residuals_by_key[cache_key] = {
                    f: aligator.ContactForceResidual(
                        self.ndx,
                        self.model,
                        self.B,
                        constraint_models,
                        prox_settings,
                        np.zeros(3),
                        f,
                    )
                    for f in frames_k
                }

            for k in range(self.H + 1):
                xk = self.xs_opt[k]
                stage = {
                    "position": [float(v) for v in xk[:3]],
                    "orientation": [float(v) for v in xk[3:7]],
                    "linear_velocity": [
                        float(v) for v in xk[self.nq : self.nq + 3]
                    ],
                    "angular_velocity": [
                        float(v) for v in xk[self.nq + 3 : self.nq + 6]
                    ],
                    "joint_configuration": [float(v) for v in xk[7 : self.nq]],
                    "joint_velocity": [
                        float(v) for v in xk[self.nq + 6 : self.nq + self.nv]
                    ],
                }
                if k < self.H:
                    uk = self.us_opt[k]
                    stage["control"] = [float(v) for v in uk]
                    cache_key = self._last_contact_dynamics.get(k)
                    forces_k = []
                    if cache_key is not None:
                        residuals_k = force_residuals_by_key[cache_key]

                        # ContactForceResidual reports the force in
                        # LOCAL_WORLD_ALIGNED (world-aligned axes),
                        # because that's the frame the friction cone and
                        # foot-height constraint were built in and need
                        # to stay physically meaningful. But the saved
                        # JSON convention (matching the other solvers) is
                        # each foot's own LOCAL frame, so rotate it there
                        # for reporting: f_local = R_world_local^T @
                        # f_world, using this stage's actual foot
                        # orientation.
                        q_k = np.asarray(xk[: self.nq])
                        pin.forwardKinematics(self.model, self.data, q_k)
                        pin.updateFramePlacements(self.model, self.data)

                        for f in cache_key[0]:
                            res = residuals_k[f]
                            res_data = res.createData()
                            res.evaluate(xk, uk, res_data)
                            f_world_aligned = np.asarray(res_data.value)
                            frame_id = self.model.getFrameId(f)
                            R_world_local = self.data.oMf[frame_id].rotation
                            f_local = R_world_local.T @ f_world_aligned
                            forces_k.extend(f_local.tolist())
                    stage["forces"] = forces_k
                stage_data.append(stage)

            properties = {
                "horizon": self.H,
                "dt": self.dt,
                "solver_mode": "ForwardDynamics",
                "xml": True,
                "nq": self.nq,
                "nv": self.nv,
                "nu": self.nu,
                "njoints": self.model.njoints - 1,
                "total_joints": self.model.njoints,
                "floating_base": True,
                "state_joints": [
                    self.model.names[i] for i in range(2, self.model.njoints)
                ],
                "joint_types": [
                    "revolute" for _ in range(self.model.njoints - 2)
                ],
                "contact_frames": all_frames,
                "stages": [],
            }
            for k in range(self.H + 1):
                cache_key = self._last_contact_dynamics.get(k)
                frames_k = list(cache_key[0]) if cache_key is not None else []
                properties["stages"].append(
                    {
                        "size": self.ndx,
                        "control": self.nu if k < self.H else 0,
                        "nforces": len(frames_k),
                        "ordered_force_frames": (
                            frames_k if k < self.H else None
                        ),
                    }
                )
        else:
            for k in range(self.H + 1):
                stage = {"state": [float(v) for v in self.xs_opt[k]]}
                if k < self.H:
                    stage["control"] = [float(v) for v in self.us_opt[k]]
                stage_data.append(stage)

            properties = {
                "horizon": self.H,
                "dt": self.dt,
                "solver_mode": "ForwardDynamics",
                "xml": True,
                "nq": self.nq,
                "nv": self.nv,
                "nu": self.nu,
                "njoints": self.model.njoints - 1,
                "total_joints": self.model.njoints,
                "floating_base": False,
                "state_joints": [
                    self.model.names[i] for i in range(1, self.model.njoints)
                ],
                "joint_types": [
                    "revolute" for _ in range(self.model.njoints - 1)
                ],
                "stages": [],
            }
            for k in range(self.H + 1):
                properties["stages"].append(
                    {
                        "size": self.ndx,
                        "control": self.nu if k < self.H else 0,
                        "nforces": 0,
                        "ordered_force_frames": [] if k < self.H else None,
                    }
                )

        output = {"properties": properties, "stage_data": stage_data}

        out_dir = os.path.dirname(output_json)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with open(output_json, "w") as f:
            json.dump(output, f, indent=4)

        print(f"[AligatorProblem] Saved solution to {output_json}")
