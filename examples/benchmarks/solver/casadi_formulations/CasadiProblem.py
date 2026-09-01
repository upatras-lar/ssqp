"""
CasadiProblem: a ProblemWrapper implementation built on top of CasADi's
``Opti`` stack, using Pinocchio's CasADi bindings (``pinocchio.casadi``)
for the rigid-body dynamics.

This is a *generalization* of the two hand-written scripts in
``problems.py`` (``run_pendubot_problem`` / ``run_go2_standing_problem``).
Those scripts hard-code the OCP for one specific robot each; this class
instead builds the OCP incrementally, one stage at a time, through the
same ``addXXX`` calls that ``run_benchmark.py`` already uses to build the
SSQP / mim_solvers formulations. That means the *same* problem-construction
code in ``run_benchmark.py`` (``make_pendubot_problem`` /
``make_go2_standing_problem``) can build a CasadiProblem without any
changes, and the two example problems reproduce the physics in
``problems.py`` exactly.

Design notes / mapping from problems.py -> the ProblemWrapper API
-------------------------------------------------------------------
* Horizon convention: ``horizon`` (``self.H``) is the number of *control
  intervals*. There are ``horizon + 1`` nodes, indexed ``k = 0 .. horizon``
  inclusive, matching ``run_benchmark.py`` (which calls the terminal-cost
  ``addSquaredConfigurationErrorCost`` at ``k = problem_wrapper.H``).
* Joint position / velocity / torque limits are *not* part of the
  ``ProblemWrapper`` interface (there's no explicit "addJointBound" call in
  run_benchmark.py). In problems.py these were pulled straight from the
  URDF/model instead of being requested explicitly, so this class does the
  same: it always enforces the model's position/velocity/effort limits at
  every stage, for every actuated joint.
* ``addContactConstraint(k, frame, max_normal_force, eq_tol)`` bundles what
  problems.py's go2 script did "by hand": a linearized friction cone
  (bounded by ``max_normal_force`` and a fixed friction coefficient) plus
  an equality-with-tolerance constraint holding the foot's *height* at the
  value it had in the initial state (``+/- eq_tol``). It does **not** pin
  the foot's horizontal position or its velocity -- velocity is handled
  separately via ``addFrameLinearVelocityBound`` (used for the feet in
  stance *and* for the terminal base-velocity constraint), matching how
  run_benchmark.py calls both methods side by side for every foot.
* Damping (``model.damping``) is always subtracted from the actuated
  torques before integrating the dynamics. For go2 this is a no-op
  (the URDF declares no joint damping), for the pendubot it reproduces the
  passive-joint-friction term problems.py applies explicitly.
"""

from __future__ import annotations

import json

import casadi as ca
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin

from ProblemWrapper import ProblemWrapper


class CasadiProblem(ProblemWrapper):
    def __init__(self, horizon, dt, urdf_path, constructor_parameters=None):
        super().__init__(horizon, dt, urdf_path)

        cp = dict(constructor_parameters or {})

        # ``solver_mode`` is an ssqpy-only concept (forward vs. constrained
        # dynamics); CasADi/Pinocchio-CasADi always uses forward dynamics
        # (cpin.aba), so it's accepted for interface-compatibility but
        # otherwise ignored.
        self.free_flyer = bool(cp.get("free_flyer", False))
        self.actuated_joints = cp.get("actuated_joints", None)
        self.friction_coeff = float(cp.get("friction_coeff", 0.6))
        # Which CasADi NLP plugin to hand to ``opti.solver`` -- "ipopt" or
        # "fatrop" for this benchmark, but any casadi nlpsol plugin works.
        self.casadi_plugin = cp.get("casadi_plugin", "ipopt")

        # Number of OCP nodes: k = 0 .. horizon inclusive.
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
        # clobber whatever the ABA pass left behind in ``self.cdata``.
        self.cdata_kin = self.cmodel.createData()

        # Which nv-space indices are actuated (i.e. which rows of the
        # torque vector are free NLP decision variables).
        if self.actuated_joints is not None:
            self._act_idx = list(self.actuated_joints)
        else:
            self._act_idx = list(range(self._base_nv, self.nv))
        self.n_act = len(self._act_idx)

        # Selection matrix mapping the actuated-torque vector into the
        # full nv-length torque vector used by ABA.
        S = np.zeros((self.nv, self.n_act))
        for col, row in enumerate(self._act_idx):
            S[row, col] = 1.0
        self._act_selection = ca.DM(S)

        # Model limits (only meaningful for the non-base dofs).
        self._q_lb = np.asarray(self.model.lowerPositionLimit).reshape(-1)
        self._q_ub = np.asarray(self.model.upperPositionLimit).reshape(-1)
        self._dq_max = np.asarray(self.model.velocityLimit).reshape(-1)
        self._tau_max = np.asarray(self.model.effortLimit).reshape(-1)
        self._damping = np.asarray(self.model.damping).reshape(-1)

        # --- Incremental problem description, filled in by addXXX calls ---
        self._vel_cost_wgt = {}  # k -> accumulated weight
        self._u_cost_wgt = {}  # k -> accumulated weight
        self._force_cost_wgt = {}  # k -> accumulated weight
        self._q_cost_wgt = {}  # k -> accumulated weight (joint-angle cost)
        self._config_cost = {}  # k -> [(desired_config, weight), ...]
        self._vel_bounds = {}  # k -> [(frame, lb, ub), ...]
        self._pos_bounds = {}  # k -> [(frame, lb, ub), ...]
        self._contacts = {}  # k -> [(frame, max_normal_force, eq_tol), ...]

        self._contact_frame_order = []  # stable global frame ordering

        # Caches of pre-built SX ca.Function objects (see the note at the
        # top of the "Helpers" section below for why these exist).
        self._dyn_fn_cache = {}
        self._frame_kin_fn_cache = {}
        self._frame_pos_fn_cache = {}
        self._cfg_err_fn_cache = {}

        self.init_states = None
        self.parameter_dict = {}

        # Populated by solve().
        self._opti = None
        self._x = None
        self._u = None
        self._sol = None
        self._nu = None  # per-stage control dimension, k = 0..H-1
        self._contact_frames_at = None  # k -> [frame, ...] (active feet)

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
    # over SX (casadi's ``SXElem`` scalar). Pinocchio-CasADi functions like
    # ``cpin.aba``/``cpin.forwardKinematics``/``cpin.integrate`` therefore
    # only accept SX arguments -- passing the MX opti variables straight
    # into them raises a Boost.Python ArgumentError. The fix (matching
    # problems.py's own pattern) is to build each piece of SX-only graph
    # exactly once with fresh ``ca.SX.sym`` symbols, wrap it in a
    # ``ca.Function``, and then *call* that Function with the MX opti
    # variables -- casadi embeds the compiled SX graph as a black-box node
    # inside the MX graph, which is exactly what Opti needs.
    # ------------------------------------------------------------------
    def _config_error(self, q_mx, desired_config):
        """Config-error residual (MX expr), matching problems.py's two
        conventions: a full-nq desired config uses the proper manifold
        difference (needed for e.g. the pendubot's revolute joints, and
        would be required for quaternion joints too); a leg/joint-only
        desired config (length nq - base_nq) is subtracted directly,
        matching the go2 script (which never references the base pose in
        this cost).
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
        fn = ca.Function(f"f_cfg_err_{mode}", [q_s, d_s], [err])
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
        fn = ca.Function(f"f_kin_{frame}", [q_s, v_s], [v_frame, p_z])
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
        fn = ca.Function(f"f_pos_{frame}", [q_s], [p_frame])
        self._frame_pos_fn_cache[frame] = fn
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
        tau_eff = tau_full - ca.DM(self._damping) * v_s

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
        fn = ca.Function(name, [q_s, v_s, tau_act_s, f_feet_s], [x_next])
        self._dyn_fn_cache[frames] = fn
        return fn

    def _frame_id(self, frame):
        if not self.model.existFrame(frame):
            raise ValueError(f"Unknown frame '{frame}' in URDF model.")
        return self.model.getFrameId(frame)

    def _friction_cone(self, f_i, max_normal_force):
        mu = self.friction_coeff
        fx, fy, fz = f_i[0], f_i[1], f_i[2]

        # All bounds are constants.
        # Symbolic dependence is entirely in the residual expression.
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

        # Reference contact heights: the z-height each contact frame had at
        # the initial state, computed once with a plain (non-symbolic)
        # Pinocchio pass -- mirrors problems.py's "_num_data" block.
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
        # variable declaration and (for Fatrop) the manual structure spec.
        # No control variables at all on the terminal node (k = H),
        # matching problems.py -- but note we still allocate a (size-0)
        # opti variable for u[H] to keep the x0,u0,x1,u1,...,x_{N-1},u_{N-1}
        # interleaving pattern Fatrop expects intact.
        contact_frames_at = {}
        nx_list = [self.nx] * self.N
        nu_list = [0] * self.N
        for k in range(H):
            frames_k = sorted(
                {c[0] for c in self._contacts.get(k, [])},
                key=self._contact_frame_order.index,
            )
            contact_frames_at[k] = frames_k
            nu_list[k] = self.n_act + 3 * len(frames_k)

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

        # -- 2. dynamics_0, path_0, dynamics_1, path_1, ..., path_{N-1} --
        ng = []
        for k in range(self.N):
            q_k = x[k][: self.nq]
            v_k = x[k][self.nq :]
            u_k = u[k]
            frames_k = contact_frames_at.get(k, [])
            # nu_list[k] == 0 at the terminal node (no control/forces
            # there), so u_k is a size-0 opti variable -- guard against
            # slicing into it, rather than always taking u_k[:n_act].
            if nu_list[k] > 0:
                tau_act_k = u_k[: self.n_act]
                f_k = u_k[self.n_act :]
            else:
                tau_act_k = None
                f_k = None

            # -- dynamics transition, added *before* this stage's path
            # constraints (no transition out of the terminal node) --
            if k < self.N - 1:
                dyn_fn = self._get_dynamics_fn(frames_k)
                x_next = dyn_fn(q_k, v_k, tau_act_k, f_k)
                opti.subject_to(x[k + 1] == x_next)

            # -- path constraints for stage k --
            ng.append(0)

            def _add(lb, expr, ub):
                opti.subject_to((lb <= expr) <= ub)
                ng[-1] += expr.nnz()

            if k == 0:
                _add(ca.DM(x0), x[k], ca.DM(x0))

            _add(lb_q, q_k[self._base_nq :], ub_q)
            _add(lb_dq, v_k[self._base_nv :], ub_dq)

            if tau_act_k is not None:
                _add(lb_tau, tau_act_k, ub_tau)

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
                for lb_c, expr_c, ub_c in self._friction_cone(f_i, max_nf):
                    _add(lb_c, expr_c, ub_c)

        # -- 3. objective (order doesn't affect Fatrop's structure) --
        J = 0
        for k in range(self.N):
            q_k = x[k][: self.nq]
            v_k = x[k][self.nq :]
            if nu_list[k] > 0:
                tau_act_k = u[k][: self.n_act]
                f_k = u[k][self.n_act :]
            else:
                tau_act_k = None
                f_k = None

            if k in self._vel_cost_wgt:
                J += self._vel_cost_wgt[k] * ca.sumsqr(v_k)
            if tau_act_k is not None and k in self._u_cost_wgt:
                J += self._u_cost_wgt[k] * ca.sumsqr(tau_act_k)
            if (
                f_k is not None
                and f_k.numel() > 0
                and k in self._force_cost_wgt
            ):
                J += self._force_cost_wgt[k] * ca.sumsqr(f_k)
            for desired_config, weight in self._config_cost.get(k, []):
                J += weight * ca.sumsqr(
                    self._config_error(q_k, desired_config)
                )
            if k in self._q_cost_wgt:
                J += self._q_cost_wgt[k] * ca.sumsqr(
                    self._config_error(q_k, x0[self._base_nq : self.nq])
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
            sol = opti.solve()
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
        return sol

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
            # after it as the "state joints", matching problems.py's go2
            # output (which starts at model.names[2:]).
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
                stage["control"] = [float(v) for v in uk[: self.n_act]]
                if uk.shape[0] > self.n_act:
                    stage["forces"] = [float(v) for v in uk[self.n_act :]]

            stage_data.append(stage)

        output = {
            "properties": {
                "horizon": H,
                "dt": self.dt,
                "solver_mode": "ForwardDynamics",
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
