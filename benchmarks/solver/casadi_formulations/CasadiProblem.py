"""
CasadiProblem: a ProblemWrapper implementation built on top of CasADi's
``Opti`` stack, using Pinocchio's CasADi bindings (``pinocchio.casadi``)
for the rigid-body dynamics.

Notes
-----
* ``horizon`` (``self.H``) is the number of control intervals; there are
  ``horizon + 1`` nodes, indexed ``k = 0 .. horizon`` inclusive.
* Joint position/velocity/torque limits are not part of the
  ``ProblemWrapper`` interface -- they're pulled directly from the URDF
  and enforced at every stage, for every actuated joint.
* ``addContactConstraint(k, frame, max_normal_force, eq_tol)`` adds a
  linearized friction cone plus an equality-with-tolerance constraint
  holding the foot's height at its initial-state value. It does not pin
  horizontal position or velocity -- velocity is handled separately via
  ``addFrameLinearVelocityBound``.
* The benchmark convention is undamped rigid-body dynamics.  This matches
  SSQP's XML model, which calls Pinocchio ABA/RNEA without adding the URDF
  damping coefficients as an extra generalized force.
"""

from __future__ import annotations

import json
import time

import casadi as ca
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin

from ProblemWrapper import ProblemWrapper

# Optional JIT compilation (run_benchmark.py's --jit flag, threaded
# through as constructor_parameters["jit"]) of each cached low-level
# ca.Function individually, rather than the whole assembled NLP via
# opti.solver(..., {"jit": True, ...}): each is built once and reused
# across all H stages, and this is where nearly all of
# nlp_g/nlp_jac_g/nlp_hess_l's cost lives, so it captures most of
# codegen's per-iteration speedup without asking gcc to compile one
# function covering the entire (H-stage) NLP at once
_JIT_OPTIONS = {
    "jit": True,
    "compiler": "shell",
    "jit_options": {"compiler": "gcc", "flags": ["-O3"]},
}
_NO_JIT_OPTIONS = {}


class CasadiProblem(ProblemWrapper):
    def __init__(self, horizon, dt, urdf_path, constructor_parameters=None):
        super().__init__(horizon, dt, urdf_path)

        cp = dict(constructor_parameters or {})

        self.free_flyer = bool(cp.get("free_flyer", False))
        self.actuated_joints = cp.get("actuated_joints", None)
        self.friction_coeff = float(cp.get("friction_coeff", 0.6))
        self._jit_opts = (
            _JIT_OPTIONS if cp.get("jit", False) else _NO_JIT_OPTIONS
        )
        self.casadi_plugin = cp.get("casadi_plugin", "ipopt")
        self.inverse_dynamics = bool(cp.get("inverse_dynamics", False))

        # Number of OCP nodes: k = 0 .. horizon inclusive
        self.N = horizon + 1

        if self.free_flyer:
            self.model = pin.buildModelFromUrdf(
                urdf_path, pin.JointModelFreeFlyer()
            )
            self._base_nq = 7  # position(3) + quaternion_xyzw(4)
            self._base_nv = 6  # spatial base twist
        else:
            self.model = pin.buildModelFromUrdf(urdf_path)
            self._base_nq = 0
            self._base_nv = 0

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nx = self.nq + self.nv

        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()  # used for the ABA pass
        # Separate Data used for kinematics-only residuals (frame
        # velocities/positions) so its forwardKinematics calls don't
        # clobber whatever the ABA pass left behind in ``self.cdata``
        self.cdata_kin = self.cmodel.createData()

        # Which nv-space indices are actuated (i.e. which rows of the
        # torque vector are free NLP decision variables)
        if self.actuated_joints is not None:
            self._act_idx = list(self.actuated_joints)
        else:
            self._act_idx = list(range(self._base_nv, self.nv))
        self.n_act = len(self._act_idx)

        # Selection matrix mapping the actuated-torque vector into the
        # full nv-length torque vector used by ABA
        S = np.zeros((self.nv, self.n_act))
        for col, row in enumerate(self._act_idx):
            S[row, col] = 1.0
        self._act_selection = ca.DM(S)

        # Model limits (only meaningful for the non-base dofs)
        self._q_lb = np.asarray(self.model.lowerPositionLimit).reshape(-1)
        self._q_ub = np.asarray(self.model.upperPositionLimit).reshape(-1)
        self._dq_max = np.asarray(self.model.velocityLimit).reshape(-1)
        self._tau_max = np.asarray(self.model.effortLimit).reshape(-1)

        # --- Incremental problem description, filled in by addXXX calls ---
        self._vel_cost_wgt = {}  # k -> accumulated weight
        self._u_cost_wgt = {}  # k -> accumulated weight
        self._force_cost_wgt = {}  # k -> accumulated weight
        self._q_cost_wgt = {}  # k -> accumulated weight (joint-angle cost)
        self._config_cost = {}  # k -> [(desired_config, weight), ...]
        self._pose_cost = {}  # k -> [(frame, p_des, R_des, t_wgt, r_wgt), ...]
        self._vel_bounds = {}  # k -> [(frame, lb, ub), ...]
        self._pos_bounds = {}  # k -> [(frame, lb, ub), ...]
        self._contacts = {}  # k -> [(frame, max_normal_force, eq_tol), ...]

        self._contact_frame_order = []  # stable global frame ordering

        # Caches of pre-built SX ca.Function objects (see the note at the
        # top of the "Helpers" section below for why these exist)
        self._dyn_fn_cache = {}
        self._invdyn_dyn_fn_cache = None
        self._invdyn_effort_fn_cache = {}
        self._frame_kin_fn_cache = {}
        self._frame_pos_fn_cache = {}
        self._frame_placement_fn_cache = {}
        self._log6_fn_cache = None
        self._cfg_err_fn_cache = {}

        self.init_states = None
        self.parameter_dict = {}

        # Populated by solve()
        self._opti = None
        self._x = None
        self._u = None
        self._sol = None
        self._nu = None  # per-stage control dimension, k = 0..H-1
        self._contact_frames_at = None  # k -> [frame, ...] (active feet)
        self.solve_time = None
        self.converged = None
        self.iterations = None
        self.objective = None
        self.constraint_violation = None

    # ------------------------------------------------------------------
    # ProblemWrapper interface: incremental problem construction
    # ------------------------------------------------------------------
    def addSquaredJointVelocityCost(self, k, weight):
        self._vel_cost_wgt[k] = self._vel_cost_wgt.get(k, 0.0) + weight

    def addSquaredControlCost(self, k, weight):
        self._u_cost_wgt[k] = self._u_cost_wgt.get(k, 0.0) + weight

    def addSquaredForceCost(self, k, weight):
        self._force_cost_wgt[k] = self._force_cost_wgt.get(k, 0.0) + weight

    def addSquaredConfigurationErrorCost(self, k, desired_config, weight):
        self._config_cost.setdefault(k, []).append(
            (np.asarray(desired_config, dtype=float).reshape(-1), weight)
        )

    def addSquaredJointAngleCost(self, k, weight):
        self._q_cost_wgt[k] = self._q_cost_wgt.get(k, 0.0) + weight

    def addSquaredFramePoseErrorCost(
        self,
        k,
        frame,
        desired_translation,
        desired_rotation,
        translation_weight,
        rotation_weight,
    ):
        self._pose_cost.setdefault(k, []).append(
            (
                frame,
                np.asarray(desired_translation, dtype=float).reshape(-1),
                np.asarray(desired_rotation, dtype=float).reshape(3, 3),
                float(translation_weight),
                float(rotation_weight),
            )
        )

    def addFrameLinearVelocityBound(self, k, frame, lb, ub):
        self._vel_bounds.setdefault(k, []).append(
            (
                frame,
                np.asarray(lb, dtype=float).reshape(-1),
                np.asarray(ub, dtype=float).reshape(-1),
            )
        )

    def addFrameTranslationBound(self, k, frame, lb, ub):
        self._pos_bounds.setdefault(k, []).append(
            (
                frame,
                np.asarray(lb, dtype=float).reshape(-1),
                np.asarray(ub, dtype=float).reshape(-1),
            )
        )

    def addContactConstraint(self, k, frame, max_normal_force, eq_tol):
        if frame not in self._contact_frame_order:
            self._contact_frame_order.append(frame)
        self._contacts.setdefault(k, []).append(
            (frame, float(max_normal_force), float(eq_tol))
        )

    def finalize(self, parameter_dict):
        self.parameter_dict = dict(parameter_dict or {})
        if "init_states" in self.parameter_dict:
            self.init_states = np.asarray(
                self.parameter_dict["init_states"], dtype=float
            ).reshape(-1)

    # ------------------------------------------------------------------
    # Helpers
    #
    # NOTE ON SX vs. MX: ``opti.variable(...)`` returns MX expressions, but
    # ``cpin.Model``/``cpin.Data`` built via ``cpin.Model(model)`` are typed
    # over SX (casadi's ``SXElem`` scalar), so Pinocchio-CasADi functions
    # like ``cpin.aba``/``cpin.forwardKinematics``/``cpin.integrate`` only
    # accept SX arguments, and passing the MX opti variables straight into
    # them raises a Boost.Python ArgumentError, which is avoided by
    # building each piece of SX-only graph exactly once with fresh
    # ``ca.SX.sym`` symbols, wrapping it in a ``ca.Function``, and then
    # *calling* that Function with the MX opti variables, since casadi
    # embeds the compiled SX graph as a black-box node inside the MX
    # graph, which is exactly what Opti needs
    # ------------------------------------------------------------------
    def _config_error(self, q_mx, desired_config):
        """Config-error residual (MX expr): a full-nq desired config uses
        the proper manifold difference (needed for e.g. the pendubot's
        revolute joints, and would be required for quaternion joints
        too); a leg/joint-only desired config (length nq - base_nq) is
        subtracted directly, since it never references the base pose.
        """
        if desired_config.shape[0] == self.nq:
            mode = "full"
        elif desired_config.shape[0] == self.nq - self._base_nq:
            mode = "partial"
        else:
            raise ValueError(
                f"desired_config has length {desired_config.shape[0]}, "
                f"expected {self.nq} (full q) or {self.nq - self._base_nq} "
                "(non-base q)."
            )
        fn = self._get_config_error_fn(mode)
        return fn(q_mx, ca.DM(desired_config))

    def _get_config_error_fn(self, mode):
        if mode in self._cfg_err_fn_cache:
            return self._cfg_err_fn_cache[mode]

        q_s = ca.SX.sym("q", self.nq)
        if mode == "full":
            d_s = ca.SX.sym("d", self.nq)
            err = cpin.difference(self.cmodel, q_s, d_s)
        else:
            d_s = ca.SX.sym("d", self.nq - self._base_nq)
            err = q_s[self._base_nq :] - d_s
        fn = ca.Function(
            f"f_cfg_err_{mode}", [q_s, d_s], [err], self._jit_opts
        )
        self._cfg_err_fn_cache[mode] = fn
        return fn

    def _get_frame_kin_fn(self, frame):
        """Returns a ca.Function (q, v) -> (v_frame(3,), p_frame_z) for the
        given frame, built once (in SX) and cached.
        """
        if frame in self._frame_kin_fn_cache:
            return self._frame_kin_fn_cache[frame]

        fid = self._frame_id(frame)
        q_s = ca.SX.sym("q", self.nq)
        v_s = ca.SX.sym("v", self.nv)
        cpin.forwardKinematics(self.cmodel, self.cdata_kin, q_s, v_s)
        cpin.updateFramePlacements(self.cmodel, self.cdata_kin)
        v_frame = cpin.getFrameVelocity(
            self.cmodel,
            self.cdata_kin,
            fid,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        ).linear
        p_z = self.cdata_kin.oMf[fid].translation[2]
        fn = ca.Function(
            f"f_kin_{frame}", [q_s, v_s], [v_frame, p_z], self._jit_opts
        )
        self._frame_kin_fn_cache[frame] = fn
        return fn

    def _get_frame_pos_fn(self, frame):
        """Returns a ca.Function (q,) -> p_frame(3,) for the given frame,
        built once (in SX) and cached.
        """
        if frame in self._frame_pos_fn_cache:
            return self._frame_pos_fn_cache[frame]

        fid = self._frame_id(frame)
        q_s = ca.SX.sym("q", self.nq)
        cpin.forwardKinematics(self.cmodel, self.cdata_kin, q_s)
        cpin.updateFramePlacements(self.cmodel, self.cdata_kin)
        p_frame = self.cdata_kin.oMf[fid].translation
        fn = ca.Function(f"f_pos_{frame}", [q_s], [p_frame], self._jit_opts)
        self._frame_pos_fn_cache[frame] = fn
        return fn

    def _get_frame_placement_fn(self, frame):
        """Returns a ca.Function (q,) -> (R_frame(3,3), p_frame(3,)) for the
        given frame, built once (in SX) and cached.
        """
        if frame in self._frame_placement_fn_cache:
            return self._frame_placement_fn_cache[frame]

        fid = self._frame_id(frame)
        q_s = ca.SX.sym("q", self.nq)
        cpin.forwardKinematics(self.cmodel, self.cdata_kin, q_s)
        cpin.updateFramePlacements(self.cmodel, self.cdata_kin)
        oMf = self.cdata_kin.oMf[fid]
        fn = ca.Function(
            f"f_placement_{frame}",
            [q_s],
            [oMf.rotation, oMf.translation],
            self._jit_opts,
        )
        self._frame_placement_fn_cache[frame] = fn
        return fn

    def _get_log6_fn(self):
        """Returns a ca.Function (R, p, R_des, p_des) -> err(6,) computing
        the SE(3) log-map pose error log6(T_des^-1 . T), matching ssqpy's
        FrameSquaredPoseErrorCost convention (see Costs.cpp): a single
        combined 6-vector (translation-then-rotation) geodesic error, not
        a naive (p - p_des, log3(R_des^T R)) pair. Frame-independent, so
        built once and cached.
        """
        if self._log6_fn_cache is not None:
            return self._log6_fn_cache

        R_s = ca.SX.sym("R", 3, 3)
        p_s = ca.SX.sym("p", 3)
        Rd_s = ca.SX.sym("Rd", 3, 3)
        pd_s = ca.SX.sym("pd", 3)
        T = cpin.SE3(R_s, p_s)
        Td = cpin.SE3(Rd_s, pd_s)
        err = cpin.log6(Td.inverse() * T).vector
        fn = ca.Function(
            "f_log6", [R_s, p_s, Rd_s, pd_s], [err], self._jit_opts
        )
        self._log6_fn_cache = fn
        return fn

    def _get_dynamics_fn(self, frames):
        """Returns a ca.Function (q, v, tau_act, f_feet) -> x_next for the
        given tuple of *active* contact frames, built once (in SX) per
        distinct frame tuple and cached.
        """
        frames = tuple(frames)
        if frames in self._dyn_fn_cache:
            return self._dyn_fn_cache[frames]

        q_s = ca.SX.sym("q", self.nq)
        v_s = ca.SX.sym("v", self.nv)
        tau_act_s = ca.SX.sym("tau_act", self.n_act)
        nf = 3 * len(frames)
        f_feet_s = ca.SX.sym("f_feet", nf)

        tau_full = self._act_selection @ tau_act_s
        tau_eff = tau_full

        fext = [cpin.Force.Zero() for _ in range(self.model.njoints)]
        for i, frame in enumerate(frames):
            fid = self._frame_id(frame)
            joint_id = self.cmodel.frames[fid].parentJoint
            f_i = f_feet_s[3 * i : 3 * i + 3]
            f_local = self.cmodel.frames[fid].placement.act(
                cpin.Force(f_i, ca.SX.zeros(3))
            )
            fext[joint_id] = fext[joint_id] + f_local

        ddq = cpin.aba(self.cmodel, self.cdata, q_s, v_s, tau_eff, fext)
        v_next = v_s + self.dt * ddq
        q_next = cpin.integrate(self.cmodel, q_s, v_next * self.dt)
        x_next = ca.vertcat(q_next, v_next)

        name = "f_dyn_" + ("_".join(frames) if frames else "free")
        fn = ca.Function(
            name, [q_s, v_s, tau_act_s, f_feet_s], [x_next], self._jit_opts
        )
        self._dyn_fn_cache[frames] = fn
        return fn

    def _get_invdyn_dynamics_fn(self):
        """Returns a ca.Function (q, v, a) -> x_next for the inverse-dynamics
        transition: a trivial double integrator (v_next = v + dt*a, q_next
        integrated from v_next), since under inverse dynamics the
        acceleration *is* the decision variable -- no cpin.aba pass needed
        for the transition itself. Frame-independent (contact forces don't
        enter the dynamics, only the effort constraint below), so built
        once and cached.
        """
        if self._invdyn_dyn_fn_cache is not None:
            return self._invdyn_dyn_fn_cache

        q_s = ca.SX.sym("q", self.nq)
        v_s = ca.SX.sym("v", self.nv)
        a_s = ca.SX.sym("a", self.nv)

        v_next = v_s + self.dt * a_s
        q_next = cpin.integrate(self.cmodel, q_s, v_next * self.dt)
        x_next = ca.vertcat(q_next, v_next)

        fn = ca.Function(
            "f_invdyn", [q_s, v_s, a_s], [x_next], self._jit_opts
        )
        self._invdyn_dyn_fn_cache = fn
        return fn

    def _get_invdyn_effort_fn(self, frames):
        """Returns a ca.Function (q, v, a, f_feet) -> tau(nv,) for the
        given tuple of *active* contact frames: the RNEA-derived generalized
        torque, exactly the "required generalized force" whose actuated rows get
        boxed by the effort limit and whose unactuated rows get boxed to
        zero (the under-actuation condition) in solve(). Built once per
        distinct frame tuple and cached. Uses the same LOCAL-frame
        external-force convention as _get_dynamics_fn's fext.
        """
        frames = tuple(frames)
        if frames in self._invdyn_effort_fn_cache:
            return self._invdyn_effort_fn_cache[frames]

        q_s = ca.SX.sym("q", self.nq)
        v_s = ca.SX.sym("v", self.nv)
        a_s = ca.SX.sym("a", self.nv)
        nf = 3 * len(frames)
        f_feet_s = ca.SX.sym("f_feet", nf)

        fext = [cpin.Force.Zero() for _ in range(self.model.njoints)]
        for i, frame in enumerate(frames):
            fid = self._frame_id(frame)
            joint_id = self.cmodel.frames[fid].parentJoint
            f_i = f_feet_s[3 * i : 3 * i + 3]
            f_local = self.cmodel.frames[fid].placement.act(
                cpin.Force(f_i, ca.SX.zeros(3))
            )
            fext[joint_id] = fext[joint_id] + f_local

        tau = cpin.rnea(self.cmodel, self.cdata, q_s, v_s, a_s, fext)
        tau_eff = tau

        name = "f_effort_" + ("_".join(frames) if frames else "free")
        fn = ca.Function(
            name, [q_s, v_s, a_s, f_feet_s], [tau_eff], self._jit_opts
        )
        self._invdyn_effort_fn_cache[frames] = fn
        return fn

    def _frame_id(self, frame):
        if not self.model.existFrame(frame):
            raise ValueError(f"Unknown frame '{frame}' in URDF model.")
        return self.model.getFrameId(frame)

    def _friction_cone(self, f_i, max_normal_force):
        mu = self.friction_coeff
        fx, fy, fz = f_i[0], f_i[1], f_i[2]

        # All bounds are constants, and the symbolic dependence is entirely
        # in the residual expression
        return [
            (0.0, fz, max_normal_force),
            (0.0, fx + mu * fz, ca.inf),
            (0.0, mu * fz - fx, ca.inf),
            (0.0, fy + mu * fz, ca.inf),
            (0.0, mu * fz - fy, ca.inf),
        ]

    # ------------------------------------------------------------------
    # ProblemWrapper interface: transcription + solve
    # ------------------------------------------------------------------
    def solve(self, parameter_dict):
        """Build and solve the direct-transcription OCP.

        The construction below deliberately follows the structure Fatrop's
        ``structure_detection: manual`` mode requires (see the Fatrop
        CasADi cookbook, https://meco-group.github.io/fatrop/
        fatrop_casadi_cookbook.html):

        1. Opti decision variables are declared **interleaved**, in the
           order x0, u0, x1, u1, ..., x_{N-1}, u_{N-1} (a single loop over
           k that allocates x[k] then u[k], never all the x's first).
        2. Constraints are added in the order dynamics_0, path_0,
           dynamics_1, path_1, ..., dynamics_{N-2}, path_{N-2}, path_{N-1}
           -- i.e. for each stage the dynamics-transition constraint (if
           any) is added *before* that stage's path constraints, and the
           per-stage constraint count is tracked in ``ng`` as we go.
        3. When solving with Fatrop, ``nx``/``nu``/``ng``/``N`` are handed
           to the solver so it can detect the block structure exactly
           rather than guessing at it (``structure_detection: 'auto'`` is
           not reliable for a problem this size/shape).

        This same construction is also valid for Ipopt (which doesn't care
        about the structure), so it's used unconditionally rather than
        having two separate code paths.
        """
        solve_params = dict(parameter_dict or {})
        plugin = solve_params.pop("casadi_plugin", self.casadi_plugin)
        solve_params.pop("tolerance", None)  # ssqpy-only knob, ignored here

        if self.init_states is None:
            x0 = np.zeros(self.nx)
        else:
            x0 = self.init_states
            if x0.shape[0] != self.nx:
                raise ValueError(
                    f"init_states has length {x0.shape[0]}, expected {self.nx}"
                )

        H = self.H  # number of control intervals; N = H + 1 nodes

        # Reference contact heights: the z-height each contact frame had
        # at the initial state, computed once with a plain (non-symbolic)
        # Pinocchio pass
        contact_heights = {}
        if self._contact_frame_order:
            num_data = self.model.createData()
            q0_num = x0[: self.nq]
            pin.forwardKinematics(self.model, num_data, q0_num)
            pin.updateFramePlacements(self.model, num_data)
            for frame in self._contact_frame_order:
                fid = self._frame_id(frame)
                contact_heights[frame] = float(
                    num_data.oMf[fid].translation[2]
                )

        # Per-stage dimensions, needed up front for both the interleaved
        # variable declaration and (for Fatrop) the manual structure spec,
        # with no control variables at all on the terminal node (k = H),
        # matching problems.py, though we still allocate a (size-0)
        # opti variable for u[H] to keep the x0,u0,x1,u1,...,x_{N-1},u_{N-1}
        # interleaving pattern Fatrop expects intact
        contact_frames_at = {}
        nx_list = [self.nx] * self.N
        nu_list = [0] * self.N
        for k in range(H):
            frames_k = sorted(
                {c[0] for c in self._contacts.get(k, [])},
                key=self._contact_frame_order.index,
            )
            contact_frames_at[k] = frames_k
            n_ctrl = self.nv if self.inverse_dynamics else self.n_act
            nu_list[k] = n_ctrl + 3 * len(frames_k)

        opti = ca.Opti()

        # -- 1. interleaved variable declaration: x0, u0, x1, u1, ... --
        x = []
        u = []
        for k in range(self.N):
            x.append(opti.variable(nx_list[k]))
            u.append(opti.variable(nu_list[k]))

        lb_q = ca.DM(self._q_lb[self._base_nq :])
        ub_q = ca.DM(self._q_ub[self._base_nq :])
        lb_dq = ca.DM(-self._dq_max[self._base_nv :])
        ub_dq = ca.DM(self._dq_max[self._base_nv :])
        lb_tau = ca.DM(-self._tau_max[self._act_idx])
        ub_tau = ca.DM(self._tau_max[self._act_idx])

        if self.inverse_dynamics:
            # Effort limit on actuated rows, and the under-actuation
            # condition (zero generalized force, since there's no actuator
            # to supply it) on unactuated rows, combined into one nv-dim
            # box on the RNEA-derived generalized torque (see
            # _get_invdyn_effort_fn), and unlike mim_solvers.SolverCSQP,
            # CasADi solvers (Ipopt/Fatrop) natively support a plain
            # equality-as-zero-width-box constraint with no special wiring
            # needed
            lb_tau_full = np.zeros(self.nv)
            ub_tau_full = np.zeros(self.nv)
            lb_tau_full[self._act_idx] = -self._tau_max[self._act_idx]
            ub_tau_full[self._act_idx] = self._tau_max[self._act_idx]
            lb_tau_full = ca.DM(lb_tau_full)
            ub_tau_full = ca.DM(ub_tau_full)

        # -- 2. dynamics_0, path_0, dynamics_1, path_1, ..., path_{N-1} --
        ng = []
        # Every bound added via _add() below is also recorded here (bucketed
        # by stage) so the post-solve constraint-violation summary can
        # re-evaluate them numerically, and the dynamics-transition equality
        # is folded in the same way: for a direct-transcription solver, a
        # dynamics defect is just another per-stage bound (lb == ub == the
        # predicted next state), not a separate metric
        violation_terms = {k: [] for k in range(self.N)}
        for k in range(self.N):
            q_k = x[k][: self.nq]
            v_k = x[k][self.nq :]
            u_k = u[k]
            frames_k = contact_frames_at.get(k, [])
            # nu_list[k] == 0 at the terminal node (no control/forces
            # there), so u_k is a size-0 opti variable, guard against
            # slicing into it rather than always taking u_k[:n_act]
            n_ctrl = self.nv if self.inverse_dynamics else self.n_act
            if nu_list[k] > 0:
                tau_act_k = None if self.inverse_dynamics else u_k[:n_ctrl]
                a_k = u_k[:n_ctrl] if self.inverse_dynamics else None
                f_k = u_k[n_ctrl:]
            else:
                tau_act_k = None
                a_k = None
                f_k = None

            # -- dynamics transition, added *before* this stage's path
            # constraints (no transition out of the terminal node) --
            if k < self.N - 1:
                if self.inverse_dynamics:
                    dyn_fn = self._get_invdyn_dynamics_fn()
                    x_next = dyn_fn(q_k, v_k, a_k)
                else:
                    dyn_fn = self._get_dynamics_fn(frames_k)
                    x_next = dyn_fn(q_k, v_k, tau_act_k, f_k)
                opti.subject_to(x[k + 1] == x_next)
                violation_terms[k].append((x_next, x[k + 1], x_next))

            # -- path constraints for stage k --
            ng.append(0)

            def _add(lb, expr, ub):
                opti.subject_to((lb <= expr) <= ub)
                ng[-1] += expr.nnz()
                violation_terms[k].append((lb, expr, ub))

            if k == 0:
                _add(ca.DM(x0), x[k], ca.DM(x0))

            _add(lb_q, q_k[self._base_nq :], ub_q)
            _add(lb_dq, v_k[self._base_nv :], ub_dq)

            if tau_act_k is not None:
                _add(lb_tau, tau_act_k, ub_tau)
            if self.inverse_dynamics and a_k is not None:
                effort_fn = self._get_invdyn_effort_fn(frames_k)
                tau_eff_k = effort_fn(q_k, v_k, a_k, f_k)
                _add(lb_tau_full, tau_eff_k, ub_tau_full)

            for frame, lb, ub in self._vel_bounds.get(k, []):
                kin_fn = self._get_frame_kin_fn(frame)
                v_frame, _ = kin_fn(q_k, v_k)
                _add(ca.DM(lb), v_frame, ca.DM(ub))

            for frame, lb, ub in self._pos_bounds.get(k, []):
                pos_fn = self._get_frame_pos_fn(frame)
                p_frame = pos_fn(q_k)
                _add(ca.DM(lb), p_frame, ca.DM(ub))

            for i, frame in enumerate(frames_k):
                kin_fn = self._get_frame_kin_fn(frame)
                _, p_z = kin_fn(q_k, v_k)
                max_nf, eq_tol = next(
                    (mf, tol)
                    for (fr, mf, tol) in self._contacts[k]
                    if fr == frame
                )
                h_z = contact_heights[frame]
                _add(h_z - eq_tol, p_z, h_z + eq_tol)
                f_i = f_k[3 * i : 3 * i + 3]
                # Contact forces are decision variables in their local frame,
                # while SSQP evaluates its cone in the world-aligned terrain
                # frame, so rotate before applying the pyramid
                placement_fn = self._get_frame_placement_fn(frame)
                R_frame, _ = placement_fn(q_k)
                f_global = R_frame @ f_i
                for lb_c, expr_c, ub_c in self._friction_cone(
                    f_global, max_nf
                ):
                    _add(lb_c, expr_c, ub_c)

        # -- 3. objective (order doesn't affect Fatrop's structure) --
        # Every term uses the 0.5 * weight * ||e||^2 convention shared with
        # the other solver formulations (see AligatorProblem/MimSolversProblem)
        J = 0
        for k in range(self.N):
            q_k = x[k][: self.nq]
            v_k = x[k][self.nq :]
            n_ctrl = self.nv if self.inverse_dynamics else self.n_act
            if nu_list[k] > 0:
                tau_act_k = None if self.inverse_dynamics else u[k][:n_ctrl]
                a_k = u[k][:n_ctrl] if self.inverse_dynamics else None
                f_k = u[k][n_ctrl:]
            else:
                tau_act_k = None
                a_k = None
                f_k = None

            if k in self._vel_cost_wgt:
                v_cost = v_k[self._base_nv :]
                J += 0.5 * self._vel_cost_wgt[k] * ca.sumsqr(v_cost)
            if tau_act_k is not None and k in self._u_cost_wgt:
                J += 0.5 * self._u_cost_wgt[k] * ca.sumsqr(tau_act_k)
            # Under inverse dynamics, "control" is the raw acceleration
            # decision variable a_k (matching SSQP's SquaredControlCost,
            # which under InverseDynamics squares u.head(nv), see
            # Model.cpp), not the RNEA-derived torque, and not the trailing
            # per-contact force entries (those get addSquaredForceCost)
            if a_k is not None and k in self._u_cost_wgt:
                a_cost = a_k[self._base_nv :] if self.free_flyer else a_k
                J += 0.5 * self._u_cost_wgt[k] * ca.sumsqr(a_cost)
            if (
                f_k is not None
                and f_k.numel() > 0
                and k in self._force_cost_wgt
            ):
                J += 0.5 * self._force_cost_wgt[k] * ca.sumsqr(f_k)
            for desired_config, weight in self._config_cost.get(k, []):
                J += (
                    0.5
                    * weight
                    * ca.sumsqr(self._config_error(q_k, desired_config))
                )
            for frame, p_des, R_des, t_wgt, r_wgt in self._pose_cost.get(
                k, []
            ):
                placement_fn = self._get_frame_placement_fn(frame)
                R_frame, p_frame = placement_fn(q_k)
                log6_fn = self._get_log6_fn()
                err = log6_fn(R_frame, p_frame, ca.DM(R_des), ca.DM(p_des))
                J += 0.5 * t_wgt * ca.sumsqr(err[:3])
                J += 0.5 * r_wgt * ca.sumsqr(err[3:])
            if k in self._q_cost_wgt:
                J += (
                    0.5
                    * self._q_cost_wgt[k]
                    * ca.sumsqr(
                        self._config_error(
                            q_k, np.zeros(self.nq - self._base_nq)
                        )
                    )
                )

        opti.minimize(J)

        # -- initial guess --
        for k in range(self.N):
            opti.set_initial(x[k], ca.DM(x0))
            if nu_list[k] > 0:
                opti.set_initial(u[k], ca.DM.zeros(nu_list[k]))

        if plugin == "fatrop":
            solve_params.setdefault("structure_detection", "manual")
            solve_params.setdefault("nx", nx_list)
            solve_params.setdefault("nu", nu_list)
            solve_params.setdefault("ng", ng)
            solve_params.setdefault("N", H)

        opti.solver(plugin, solve_params)

        try:
            start = time.perf_counter()
            sol = opti.solve()
            self.solve_time = time.perf_counter() - start
        except RuntimeError:
            print("CasADi solve failed; last iterate for the final state:")
            print(opti.debug.value(x[-1]))
            raise

        self._opti = opti
        self._x = x
        self._u = u
        self._sol = sol
        self._nu = nu_list
        self._contact_frames_at = contact_frames_at
        stats = sol.stats()
        self.converged = bool(stats.get("success", True))
        self.iterations = stats.get("iter_count")
        self.objective = float(sol.value(J))
        self.constraint_violation = self._evaluate_violation(
            sol, violation_terms
        )
        return sol

    def _evaluate_violation(self, sol, violation_terms):
        """Sum, over stages, of the worst single bound violation at that
        stage -- matching SSQP's trajectory_constraint_violation (see
        ssqp::_cf in src/solvers/ssqp.cpp): per stage, take the infinity
        norm of how far outside [lb, ub] the (possibly vector-valued)
        constraint sits, then sum that per-stage worst case across stages.
        """
        def _num(term):
            # A bound is either a plain constant (the usual case) or, for
            # the dynamics-transition term, itself a symbolic expression
            # (the predicted next state) that needs the same sol.value()
            # evaluation as expr
            if isinstance(term, (ca.MX, ca.SX)):
                return np.asarray(sol.value(term), dtype=float).reshape(-1)
            return np.asarray(ca.DM(term), dtype=float).reshape(-1)

        total = 0.0
        for terms in violation_terms.values():
            stage_worst = 0.0
            for lb, expr, ub in terms:
                lb_arr = _num(lb)
                ub_arr = _num(ub)
                value = _num(expr)
                violation = np.maximum(
                    0.0, np.maximum(lb_arr - value, value - ub_arr)
                )
                if violation.size:
                    stage_worst = max(stage_worst, float(violation.max()))
            total += stage_worst
        return total

    # ------------------------------------------------------------------
    # ProblemWrapper interface: saving results
    # ------------------------------------------------------------------
    def save(self, output_json):
        if self._sol is None:
            raise RuntimeError("save() called before solve()")

        sol = self._sol
        H = self.H

        joint_names = [
            self.model.names[i] for i in range(1, self.model.njoints)
        ]
        base_offset = 0
        if self.free_flyer:
            # names[1] is the free-flyer joint itself; report the joints
            # after it as the "state joints", matching the other backends'
            # save() output
            joint_names = [
                self.model.names[i] for i in range(2, self.model.njoints)
            ]
            base_offset = 1

        stage_data = []
        for k in range(self.N):
            xk = np.asarray(sol.value(self._x[k])).reshape(-1)
            if self.free_flyer:
                stage = {
                    "position": [float(v) for v in xk[0:3]],
                    "orientation": [float(v) for v in xk[3:7]],
                    "linear_velocity": [
                        float(v) for v in xk[self.nq : self.nq + 3]
                    ],
                    "angular_velocity": [
                        float(v) for v in xk[self.nq + 3 : self.nq + 6]
                    ],
                    "joint_configuration": [float(v) for v in xk[7 : self.nq]],
                    "joint_velocity": [
                        float(v) for v in xk[self.nq + 6 : self.nx]
                    ],
                }
            else:
                stage = {"state": [float(v) for v in xk]}

            if self._nu[k] > 0:
                uk = np.asarray(sol.value(self._u[k])).reshape(-1)
                if self.inverse_dynamics:
                    a_k = uk[: self.nv]
                    f_k = uk[self.nv :]
                    frames_k = self._contact_frames_at.get(k, [])
                    effort_fn = self._get_invdyn_effort_fn(frames_k)
                    # Recovered actuator torque, physically meaningful and
                    # comparable to the "control" field the other dynamics
                    # modes report, unlike the raw acceleration decision
                    # variable a_k itself
                    tau_eff = np.asarray(
                        effort_fn(xk[: self.nq], xk[self.nq :], a_k, f_k)
                    ).reshape(-1)
                    stage["control"] = [
                        float(v) for v in tau_eff[self._act_idx]
                    ]
                    # Already in each contact frame's own LOCAL axes (see
                    # _get_invdyn_effort_fn), matching the JSON convention
                    # shared with the other solver backends directly
                    if f_k.size > 0:
                        stage["forces"] = [float(v) for v in f_k]
                else:
                    stage["control"] = [float(v) for v in uk[: self.n_act]]
                    if uk.shape[0] > self.n_act:
                        stage["forces"] = [
                            float(v) for v in uk[self.n_act :]
                        ]

            stage_data.append(stage)

        output = {
            "benchmark": {
                "format_version": 1,
                "solver": self.casadi_plugin,
                "success": self.converged,
                "iterations": self.iterations,
                "runtime_ms": self.solve_time * 1000.0,
                "objective": self.objective,
                "constraint_violation": self.constraint_violation,
            },
            "properties": {
                "horizon": H,
                "dt": self.dt,
                "solver_mode": (
                    "InverseDynamics"
                    if self.inverse_dynamics
                    else "ForwardDynamics"
                ),
                "xml": True,
                "nq": self.nq,
                "nv": self.nv,
                "nu": self.n_act,
                "njoints": self.model.njoints - 1,
                "total_joints": self.model.njoints,
                "floating_base": self.free_flyer,
                "state_joints": joint_names,
                "joint_types": [
                    "revolute"
                    for _ in range(self.model.njoints - 1 - base_offset)
                ],
                "stages": [],
            },
            "stage_data": stage_data,
        }
        if self._contact_frame_order:
            output["properties"]["contact_frames"] = self._contact_frame_order

        for k in range(self.N):
            has_u = self._nu[k] > 0
            frames_k = self._contact_frames_at.get(k, []) if has_u else []
            stage_info = {
                "size": self.nx,
                "control": self.n_act if has_u else 0,
                "nforces": len(frames_k),
                "ordered_force_frames": frames_k if frames_k else None,
            }
            output["properties"]["stages"].append(stage_info)

        with open(output_json, "w") as f:
            json.dump(output, f, indent=4)

        print(f"Saved solution to {output_json}")
