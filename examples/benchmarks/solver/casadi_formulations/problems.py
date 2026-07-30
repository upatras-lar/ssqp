import casadi as ca
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin

import json
from time import time


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
