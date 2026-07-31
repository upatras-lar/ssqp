import casadi as ca
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin

import json


def run_pendubot_problem(solver, solver_args, output_json):
    """
    Pendubot swing-up trajectory optimization (Direct Angle Optimization).

    Dynamics come from Pinocchio's CasADi interface (cpin.aba).
    joint1/joint2 are standard revolute joints, meaning:
        nq = 2  ->  q = [q1, q2]             (direct angles in radians)
        nv = 2  ->  v = [dq1, dq2]           (angular velocities)
    The full OCP state is x = [q(2), v(2)] (nx = 4).
    """
    # ---------------------------------------------------------------------------
    # Build a Pinocchio model from the URDF
    # ---------------------------------------------------------------------------
    URDF_PATH = "urdf/double_pendulum/pendubot.urdf"

    model = pin.buildModelFromUrdf(
        URDF_PATH
    )  # base_link is fixed to the world

    nq, nv = model.nq, model.nv  # Now 2, 2
    assert nq == 2 and nv == 2

    tau1_max = float(model.effortLimit[0])
    dq_max = np.asarray(model.velocityLimit).reshape(-1)
    damping = np.asarray(model.damping).reshape(-1)

    cmodel = cpin.Model(model)  # CasADi (symbolic) mirror of the model
    cdata = cmodel.createData()

    # We add 1 to the problem's regular horizon, since all solvers
    # formulate the problem for stages [0, H] (inclusive), 2501 total stages
    # with horizon 2500
    H = 2501
    dt = 0.001

    nx = [nq + nv for _ in range(H)]  # 4 = [q1, q2, dq1, dq2]
    nu = [1 for _ in range(H - 1)] + [
        0
    ]  # tau1 only; no control on the last node

    # ---------------------------------------------------------------------------
    # Symbolic dynamics
    # ---------------------------------------------------------------------------
    q = ca.SX.sym("q", nq)
    dq = ca.SX.sym("dq", nv)
    state = ca.vertcat(q, dq)

    tau1 = ca.SX.sym("tau1")
    tau_full = ca.vertcat(tau1, 0)  # joint2 is unactuated
    tau_eff = tau_full - damping * dq  # subtract viscous joint damping

    ddq = cpin.aba(cmodel, cdata, q, dq, tau_eff)

    dq_next = dq + dt * ddq
    q_next = cpin.integrate(
        cmodel, q, dq_next * dt
    )  # Standard vector addition for R^2
    state_next = ca.vertcat(q_next, dq_next)

    f_intg = ca.Function("f_dyn_discr", [state, tau1], [state_next])

    # ---------------------------------------------------------------------------
    # Costs & Target States
    # ---------------------------------------------------------------------------
    V_WGT = 1e-4
    U_WGT = 1e-4
    C_WGT = 1.0

    # x0: Hang straight down [q1=0, q2=0, dq1=0, dq2=0]
    x0 = ca.DM([0.0, 0.0, 0.0, 0.0])
    # xf: Balance straight up [q1=pi, q2=0, dq1=0, dq2=0]
    xf = ca.DM([np.pi, 0.0, 0.0, 0.0])

    xf_q = ca.SX(xf[0:2])

    f_vel_cost = ca.Function(
        "f_vel_cost",
        [state],
        [V_WGT * ca.sumsqr(state[2:4])],  # Velocity starts at index 2 now
    )

    f_u_cost = ca.Function("f_u_cost", [tau1], [U_WGT * tau1**2])

    f_config_cost = ca.Function(
        "f_config_cost",
        [state],
        [
            C_WGT * ca.sumsqr(cpin.difference(cmodel, state[0:2], xf_q))
        ],  # Config is index 0:2
    )

    # ---------------------------------------------------------------------------
    # Bounds and boundary conditions
    # ---------------------------------------------------------------------------
    lb_u, ub_u = -tau1_max, tau1_max
    lb_dq, ub_dq = ca.DM(-dq_max), ca.DM(dq_max)

    # ---------------------------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------------------------
    def discrete_dynamics(u, x, k):
        return f_intg(x, u)

    def path_constraints(u, x, k):
        constrs = [
            (lb_dq, x[2:4], ub_dq),  # Velocity limits start at index 2
        ]
        if k == 0:
            constrs.append((x0, x, x0))  # fix initial state
        if nu[k] > 0:
            constrs.append((lb_u, u, ub_u))  # joint1 torque limit
        return constrs

    def cost(u, x, k):
        Jk = f_vel_cost(x)
        if nu[k] > 0:
            Jk += f_u_cost(u)
        if k > H - 500:
            Jk += f_config_cost(x)

        return Jk

    def initial_guess(k):
        x_g = x0
        u_g = ca.DM.zeros(1)
        return u_g, x_g

    # ---------------------------------------------------------------------------
    # Build and solve the OCP
    # ---------------------------------------------------------------------------
    opti = ca.Opti()
    x = []
    u = []
    for k in range(H):
        x.append(opti.variable(nx[k]))
        u.append(opti.variable(nu[k]))

    ng = []
    for k in range(H):
        if k < H - 1:
            opti.subject_to(x[k + 1] == discrete_dynamics(u[k], x[k], k))
        path_constr = path_constraints(u[k], x[k], k)
        ng.append(0)
        for constr in path_constr:
            ng[-1] += constr[1].nnz()
            opti.subject_to((constr[0] <= constr[1]) <= constr[2])

    J = 0
    for k in range(H):
        J += cost(u[k], x[k], k)

    opti.minimize(J)

    for k in range(H):
        u_init, x_init = initial_guess(k)
        opti.set_initial(u[k], u_init)
        opti.set_initial(x[k], x_init)

    opti.solver(solver, solver_args(nx, nu, ng, H))

    try:
        sol = opti.solve()
    except RuntimeError:
        print(opti.debug.value(x[-1]))
        exit(0)

    # ---------------------------------------------------------------------------
    # Extract solution values
    # ---------------------------------------------------------------------------
    tau1_traj = np.array([sol.value(u[k]) for k in range(H - 1)])

    print(f"Final state (q1, q2, dq1, dq2): {sol.value(x[H - 1])}")
    print(
        f"Max |tau1| used: {np.max(np.abs(tau1_traj)):.4f} Nm (limit {tau1_max})"
    )

    # ---------------------------------------------------------------------------
    # Save solution in the JSON format required by the visualizer
    # ---------------------------------------------------------------------------
    stage_data = []

    for k in range(H):
        stage = {
            "state": [
                float(sol.value(x[k][0])),  # q1
                float(sol.value(x[k][1])),  # q2
                float(sol.value(x[k][2])),  # dq1
                float(sol.value(x[k][3])),  # dq2
            ]
        }

        # No control on the final node
        if k < H - 1:
            stage["control"] = [float(sol.value(u[k]))]

        stage_data.append(stage)

    output = {
        "properties": {
            "horizon": H - 1,
            "dt": dt,
            "solver_mode": "ForwardDynamics",
            "xml": True,
            "nq": nq,
            "nv": nv,
            "nu": 1,
            "njoints": model.njoints - 1,
            "total_joints": model.njoints,
            "floating_base": False,
            "state_joints": [model.names[i] for i in range(1, model.njoints)],
            "joint_types": ["revolute", "revolute"],
            "stages": [],
        },
        "stage_data": stage_data,
    }

    # Fill stage metadata
    for k in range(H):
        stage_info = {
            "size": nx[k],
            "control": nu[k],
            "nforces": 0,
            "ordered_force_frames": [],
        }

        if nu[k] == 0:
            stage_info["ordered_force_frames"] = None

        output["properties"]["stages"].append(stage_info)

    with open(output_json, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Saved solution to {output_json}")


def run_go2_standing_problem(solver, solver_args, output_json):
    """
    Go2 quadruped standing trajectory optimization (Direct Transcription),
    ported from ssqpy to CasADi + pinocchio-casadi, following the same
    stage-loop structure as run_pendubot_problem above.

    Free-flyer base, so:
        nq = 19  ->  q = [pos(3), quat_xyzw(4), leg_joints(12)]
        nv = 18  ->  v = [base_twist(6), leg_joint_vel(12)]
    State x = [q(19), v(18)]  (nx = 37).

    Contact model: unlike aligator's constrained-ABA formulation (which
    solves for each foot's contact force implicitly via a KKT system inside
    the dynamics evaluation), this direct-transcription version makes the
    four feet's contact forces EXPLICIT NLP decision variables at every
    stage. Per-stage control is therefore the stacked vector
        u = [tau(12), f_FL(3), f_FR(3), f_RL(3), f_RR(3)]   (nu = 24)
    and the forces are injected into cpin.aba as external spatial forces.
    "Foot stays planted" is then a single equality path constraint (foot
    linear velocity == 0) -- this is what ssqpy's separate
    addContactConstraint + FrameLinearVelocityBound collapse to for an
    always-in-stance standing task.

    VERIFY: the fext-construction block below (using
    frame.placement.act(cpin.Force(...))) is the standard Pinocchio pattern
    for injecting a contact force expressed at a frame into ABA's per-joint
    external-force list, but the exact cpin.Force constructor argument
    order (linear, angular) and whether .act() accepts a plain SX 3-vector
    or requires wrapping can vary by pinocchio-casadi version -- check
    `help(cpin.Force)` before trusting it.
    """
    # ---------------------------------------------------------------------------
    # Build a Pinocchio model from the URDF (free-flyer base)
    # ---------------------------------------------------------------------------
    URDF_PATH = "urdf/go2/go2.urdf"

    model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())

    nq, nv = model.nq, model.nv  # 19, 18
    nu_act = 12  # actuated leg torques

    FOOT_FRAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    foot_frame_ids = [model.getFrameId(f) for f in FOOT_FRAMES]
    n_feet = len(FOOT_FRAMES)
    nf = 3 * n_feet  # 12 stacked contact-force components

    # Joint limits are only meaningful for the 12 leg joints -- the first 7
    # nq entries (base pos + quat) and first 6 nv entries (base twist) are
    # placeholders in most URDFs.
    q_lb = np.asarray(model.lowerPositionLimit[7:nq]).reshape(-1)
    q_ub = np.asarray(model.upperPositionLimit[7:nq]).reshape(-1)
    dq_max = np.asarray(model.velocityLimit[6:nv]).reshape(-1)
    tau_max = np.asarray(model.effortLimit[6:nv]).reshape(-1)

    cmodel = cpin.Model(model)
    cdata = cmodel.createData()  # used for the ABA / dynamics pass
    cdata_kin = cmodel.createData()  # separate data for kinematics-only
    # residuals (foot velocity, base velocity) so their forwardKinematics
    # calls don't clobber whatever ABA left in `cdata`.

    HORIZON = 100
    dt = 0.01
    H = HORIZON + 1  # 101 stages [0, H-1] inclusive, matching the
    # pendubot convention above

    nx = [nq + nv for _ in range(H)]
    nu = [nu_act + nf for _ in range(H - 1)] + [0]  # no control/forces on
    # the terminal node

    # ---------------------------------------------------------------------------
    # Weights (identical to the ssqpy / aligator versions)
    # ---------------------------------------------------------------------------
    A_WGT = 10.0
    V_WGT = 1e-2
    U_WGT = 1e-3
    F_WGT = 1e-5
    MAX_NORMAL_FORCE = 50.0
    MU = 0.6

    base_pos = np.array([0.0, 0.0, 0.3879793820122])
    base_config = np.array([0.0, np.pi / 6, -np.pi / 3] * 4)  # 12 leg angles

    q0 = np.zeros(nq)
    q0[:3] = base_pos
    q0[6] = 1.0  # quaternion w -> identity orientation
    q0[7:19] = base_config
    x0 = ca.DM(np.concatenate([q0, np.zeros(nv)]))

    # Contact-constraint target height h_z per foot: the ssqpy contact
    # constraint pins each foot to p_z = h_z, which is the foot's height at
    # the standing configuration, not necessarily 0 (depends on the URDF's
    # foot-frame origin vs. ground plane). Computed once numerically (not
    # symbolically) via a plain, non-CasADi Pinocchio FK pass at q0.
    _num_data = model.createData()
    pin.forwardKinematics(model, _num_data, q0)
    pin.updateFramePlacements(model, _num_data)
    foot_h_z = np.array(
        [_num_data.oMf[fid].translation[2] for fid in foot_frame_ids]
    )

    xf_q_joints = ca.SX(base_config)  # config cost only targets the legs,
    # matching ssqpy's SquaredConfigurationErrorCost(model, base_config, ...)
    # which never references the base pose

    # ---------------------------------------------------------------------------
    # Symbolic dynamics: constrained ABA via explicit contact-force
    # decision variables applied as external spatial forces
    # ---------------------------------------------------------------------------
    q = ca.SX.sym("q", nq)
    v = ca.SX.sym("v", nv)
    state = ca.vertcat(q, v)

    tau_act = ca.SX.sym("tau", nu_act)
    f_feet = ca.SX.sym(
        "f_feet", nf
    )  # stacked [f_FL(3), f_FR(3), f_RL(3), f_RR(3)], world-aligned linear force

    tau_full = ca.vertcat(ca.SX.zeros(6), tau_act)  # base is unactuated

    fext = [cpin.Force.Zero() for _ in range(model.njoints)]
    for i, frame_id in enumerate(foot_frame_ids):
        f_i = f_feet[3 * i : 3 * i + 3]
        frame = cmodel.frames[frame_id]
        joint_id = frame.parentJoint
        # frame.placement is the FIXED offset from the parent joint to the
        # foot frame (from the URDF), so this doesn't need forwardKinematics
        # -- it transforms the contact force into the joint's local frame
        # for ABA's fext argument.
        f_local = frame.placement.act(cpin.Force(f_i, ca.SX.zeros(3)))
        fext[joint_id] = fext[joint_id] + f_local

    ddq = cpin.aba(cmodel, cdata, q, v, tau_full, fext)

    v_next = v + dt * ddq
    q_next = cpin.integrate(cmodel, q, v_next * dt)  # handles the quaternion
    # part of the free-flyer correctly, same call ssqpy relies on implicitly
    state_next = ca.vertcat(q_next, v_next)

    f_intg = ca.Function(
        "f_dyn_discr_go2", [state, tau_act, f_feet], [state_next]
    )

    # ---------------------------------------------------------------------------
    # Kinematic residuals: foot velocity (contact constraint) and base
    # velocity (terminal constraint), evaluated on the separate cdata_kin
    # ---------------------------------------------------------------------------
    cpin.forwardKinematics(cmodel, cdata_kin, q, v)
    cpin.updateFramePlacements(cmodel, cdata_kin)

    foot_vel_fns = []
    foot_pz_fns = []
    for frame_id in foot_frame_ids:
        v_foot = cpin.getFrameVelocity(
            cmodel, cdata_kin, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        ).linear
        foot_vel_fns.append(ca.Function("f_foot_vel", [q, v], [v_foot]))
        # Position only depends on q algebraically (not v), even though it
        # was populated by the same forwardKinematics(q, v) call above, so
        # a Function with just q as input is valid.
        p_z_foot = cdata_kin.oMf[frame_id].translation[2]
        foot_pz_fns.append(ca.Function("f_foot_pz", [q], [p_z_foot]))

    base_frame_id = model.getFrameId("base")  # VERIFY: may be "trunk" or
    # "base_link" depending on your Go2 URDF's frame naming
    base_lin_vel = cpin.getFrameVelocity(
        cmodel,
        cdata_kin,
        base_frame_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    ).linear
    f_base_vel = ca.Function("f_base_vel", [q, v], [base_lin_vel])

    # ---------------------------------------------------------------------------
    # Costs
    # ---------------------------------------------------------------------------
    f_vel_cost = ca.Function(
        "f_vel_cost_go2",
        [state],
        # Full nv-length velocity, base twist included -- ssqpy calls this
        # "SquaredJointVelocityCost" but nothing else in that script damps
        # the base twist separately, same assumption used in the aligator
        # port. Swap to state[nq + 6 :] if you intended leg-joints-only.
        [V_WGT * ca.sumsqr(state[nq:])],
    )

    f_u_cost = ca.Function(
        "f_u_cost_go2", [tau_act], [U_WGT * ca.sumsqr(tau_act)]
    )

    f_config_cost = ca.Function(
        "f_config_cost_go2",
        [state],
        # Leg joints (indices 7:19 of q) are plain revolute coordinates, so
        # direct subtraction is exact here -- no need for cpin.difference
        # the way the pendubot cost uses it for its (trivial) manifold.
        [A_WGT * ca.sumsqr(state[7:19] - xf_q_joints)],
    )

    f_force_cost = ca.Function(
        "f_force_cost_go2", [f_feet], [F_WGT * ca.sumsqr(f_feet)]
    )

    # ---------------------------------------------------------------------------
    # Bounds
    # ---------------------------------------------------------------------------
    lb_q, ub_q = ca.DM(q_lb), ca.DM(q_ub)
    lb_dq, ub_dq = ca.DM(-dq_max), ca.DM(dq_max)
    lb_tau, ub_tau = ca.DM(-tau_max), ca.DM(tau_max)
    zero3 = ca.DM.zeros(3)

    def friction_cone_constraints(f_i):
        """
        Linearized (4-facet pyramidal) friction cone for one foot's force
        vector f_i = [f1, f2, f_n] (f_n assumed to be the world-z / ground-
        normal component, matching the [x, y, z] stacking used for f_feet
        elsewhere in this function):
            f_n in [0, MAX_NORMAL_FORCE]
            |f1| <= mu * f_n   ->   -mu*f_n <= f1 <= mu*f_n
            |f2| <= mu * f_n   ->   -mu*f_n <= f2 <= mu*f_n
        The cone bounds are themselves expressions in f_n (not constants),
        which the (lb, expr, ub) triples below support directly since
        opti.subject_to((lb <= expr) <= ub) accepts symbolic lb/ub.
        """
        f1, f2, fn = f_i[0], f_i[1], f_i[2]
        return [
            (0.0, fn, MAX_NORMAL_FORCE),
            (-MU * fn, f1, MU * fn),
            (-MU * fn, f2, MU * fn),
        ]

    # ---------------------------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------------------------
    def discrete_dynamics(u, x, k):
        tau_k = u[0:nu_act]
        f_k = u[nu_act : nu_act + nf]
        return f_intg(x, tau_k, f_k)

    def path_constraints(u, x, k):
        q_k, v_k = x[0:nq], x[nq:]
        constrs = [
            (lb_q, q_k[7:19], ub_q),
            (lb_dq, v_k[6:18], ub_dq),
        ]
        if k == 0:
            constrs.append((x0, x, x0))  # fix initial state
        if nu[k] > 0:
            tau_k = u[0:nu_act]
            f_k = u[nu_act : nu_act + nf]
            constrs.append((lb_tau, tau_k, ub_tau))
            for i in range(n_feet):
                f_i = f_k[3 * i : 3 * i + 3]
                constrs.extend(friction_cone_constraints(f_i))
            for i, foot_vel_fn in enumerate(foot_vel_fns):
                constrs.append((zero3, foot_vel_fn(q_k, v_k), zero3))
            for i, foot_pz_fn in enumerate(foot_pz_fns):
                h_z = float(foot_h_z[i])
                constrs.append((h_z, foot_pz_fn(q_k), h_z))
        if k == H - 1:
            constrs.append((zero3, f_base_vel(q_k, v_k), zero3))
        return constrs

    def cost(u, x, k):
        Jk = f_vel_cost(x) + f_config_cost(x)
        if nu[k] > 0:
            tau_k = u[0:nu_act]
            f_k = u[nu_act : nu_act + nf]
            Jk += f_u_cost(tau_k)
            Jk += f_force_cost(f_k)
        return Jk

    def initial_guess(k):
        x_g = x0
        u_g = ca.DM.zeros(nu[k])
        return u_g, x_g

    # ---------------------------------------------------------------------------
    # Build and solve the OCP
    # ---------------------------------------------------------------------------
    opti = ca.Opti()
    x = []
    u = []
    for k in range(H):
        x.append(opti.variable(nx[k]))
        u.append(opti.variable(nu[k]))

    ng = []
    for k in range(H):
        if k < H - 1:
            opti.subject_to(x[k + 1] == discrete_dynamics(u[k], x[k], k))
        path_constr = path_constraints(u[k], x[k], k)
        ng.append(0)
        for constr in path_constr:
            ng[-1] += constr[1].nnz()
            opti.subject_to((constr[0] <= constr[1]) <= constr[2])

    J = 0
    for k in range(H):
        J += cost(u[k], x[k], k)

    opti.minimize(J)

    for k in range(H):
        u_init, x_init = initial_guess(k)
        opti.set_initial(u[k], u_init)
        opti.set_initial(x[k], x_init)

    opti.solver(solver, solver_args(nx, nu, ng, H))

    try:
        sol = opti.solve()
    except RuntimeError:
        print(opti.debug.value(x[-1]))
        exit(0)

    # ---------------------------------------------------------------------------
    # Extract solution values
    # ---------------------------------------------------------------------------
    tau_traj = np.array([sol.value(u[k][0:nu_act]) for k in range(H - 1)])

    print(f"Final base position: {sol.value(x[H - 1][0:3])}")
    print(
        f"Max |tau| used: {np.max(np.abs(tau_traj)):.4f} Nm (limit {tau_max.max():.1f})"
    )

    # ---------------------------------------------------------------------------
    # Save solution in the JSON format required by the visualizer
    # ---------------------------------------------------------------------------
    stage_data = []

    for k in range(H):
        x_k = sol.value(x[k])
        stage = {
            "position": [float(v) for v in x_k[0:3]],
            "orientation": [float(v) for v in x_k[3:7]],
            "linear_velocity": [float(v) for v in x_k[nq : nq + 3]],
            "angular_velocity": [float(v) for v in x_k[nq + 3 : nq + 6]],
            "joint_configuration": [float(v) for v in x_k[7:nq]],
            "joint_velocity": [float(v) for v in x_k[nq + 6 : nq + nv]],
        }

        if k < H - 1:
            u_k = sol.value(u[k])
            stage["control"] = [float(v) for v in u_k[0:nu_act]]
            stage["forces"] = [float(v) for v in u_k[nu_act:]]

        stage_data.append(stage)

    output = {
        "properties": {
            "horizon": H - 1,
            "dt": dt,
            "solver_mode": "ForwardDynamics",
            "xml": True,
            "nq": nq,
            "nv": nv,
            "nu": nu_act,
            "njoints": model.njoints - 1,
            "total_joints": model.njoints,
            "floating_base": True,
            "state_joints": [model.names[i] for i in range(2, model.njoints)],
            "joint_types": ["revolute" for _ in range(model.njoints - 2)],
            "contact_frames": FOOT_FRAMES,
            "stages": [],
        },
        "stage_data": stage_data,
    }

    for k in range(H):
        stage_info = {
            "size": nx[k],
            "control": nu_act if nu[k] > 0 else 0,
            "nforces": n_feet if nu[k] > 0 else 0,
            "ordered_force_frames": FOOT_FRAMES if nu[k] > 0 else None,
        }
        output["properties"]["stages"].append(stage_info)

    with open(output_json, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Saved solution to {output_json}")
