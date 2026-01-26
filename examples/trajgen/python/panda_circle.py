import ssqpy
import numpy as np
from math import pi, sin, cos

PLOT = True

if PLOT:
    import matplotlib.pyplot as plt

nq = 9
nv = 9
H = 400
dt = 0.01
P_WGT = 10
R_WGT = 1
V_WGT = 1e-5
U_WGT = 1e-5

CONSTRAIN = True
constraint_hlines = [0.1, 0.41]
constraint_vlines = [-0.15, 0.15]

cx = 0.3
cy = 0
cz = 0.26
cr = 0.22

DISTANCE = 2 * pi
ω = DISTANCE / (H * dt)
φ0 = 0

traj = []

for k in range(H + 1):
    φk = φ0 + ω * k * dt
    traj.append(np.array([cx, cy + cr * sin(φk), cz + cr * cos(φk)]))

Rd = np.eye(3)
Rd[1, 1] = -1
Rd[2, 2] = -1

model = ssqpy.model.Model(H, dt, urdf_path="../../urdf/panda/panda.urdf")

init_state = np.zeros(2 * nq)
init_state[:nq] = [
    0.0013,
    -0.9115,
    -0.0010,
    -2.6316,
    -0.0008,
    1.7431,
    0.7862,
    0.0,
    0.0,
]

"""
class CustomCost(ssqpy.model.costs.StageCost):
    def evaluate(self, x, u):
        return U_WGT * 0.5 * u.T @ u

    def gradient(self, x, u):
        grad_x = np.zeros_like(x)
        grad_u = U_WGT * u
        return grad_x, grad_u

    def hessian_xx(self, x, u):
        return np.zeros((x.shape[0], x.shape[0]))

    def hessian_xu(self, x, u):
        return np.zeros((x.shape[0], u.shape[0]))

    def hessian_uu(self, x, u):
        return U_WGT * np.eye(u.shape[0])
"""

lb = np.array((-np.inf, constraint_vlines[0], constraint_hlines[0]))
ub = np.array((np.inf, constraint_vlines[1], constraint_hlines[1]))
box_constraint = ssqpy.model.constraints.FramePositionBound(
    model, "panda_ee", lb, ub, nq, nv
)

for k in range(H):
    desired_pose = np.zeros((4, 4))
    desired_pose[:3, :3] = Rd
    desired_pose[:3, 3] = traj[k]
    desired_pose[3, 3] = 1

    model.addCost(
        k,
        ssqpy.model.costs.FrameSquaredPoseErrorCost(
            model, desired_pose, "panda_ee", nq, nq, P_WGT, R_WGT
        ),
    )
    model.addCost(k, ssqpy.model.costs.SquaredControlCost(nq, U_WGT))
    model.addCost(k, ssqpy.model.costs.SquaredJointVelocityCost(nq, V_WGT))

    if CONSTRAIN:
        model.addConstraints(k, box_constraint)

desired_pose = np.zeros((4, 4))
desired_pose[:3, :3] = Rd
desired_pose[:3, 3] = traj[k]
desired_pose[3, 3] = 1

model.addCost(
    H,
    ssqpy.model.costs.FrameSquaredPoseErrorCost(
        model, desired_pose, "panda_ee", nq, nq, P_WGT, R_WGT
    ),
)
model.addCost(H, ssqpy.model.costs.SquaredJointVelocityCost(nq, V_WGT))

if CONSTRAIN:
    model.addConstraints(H, box_constraint)

model.finalize(init_states=[init_state])

solver = ssqpy.solvers.ssqp(
    model,
    loose_success=False,
    tolerance=1e-1,
    admm_abs_tolerance=1e-3,
    admm_rel_tolerance=1e-4,
    admm_adaptive_rho_tolerance=5,
    admm_rho_update_iter_freq=25,
    admm_max_iter=1000,
)

state, x, u = solver.solve()

solver.write("../data/panda_circle.json")

print("x:")
x.print()
print("u:")
u.print()

if PLOT:
    state_data = np.array(
        [model.forwardKinematics(q[:9], "panda_ee")[0] for q in x.toarray()]
    )
    traj = np.array(traj)

    fig, axs = plt.subplots(1, 2)
    axs[0].set_title("Total Error")
    axs[0].plot(
        np.arange(H + 1),
        np.linalg.norm(state_data - traj, axis=1),
    )
    axs[0].set(xlabel="Step", ylabel="End effector position error")
    axs[0].grid(True)

    axs[1].set_title("Trajectory")

    traj = np.delete(traj, 0, axis=1)

    state_data = np.delete(state_data, 0, axis=1)

    axs[1].scatter(*zip(*traj), label="Cost trajectory")

    if CONSTRAIN:
        axs[1].hlines(
            y=constraint_hlines,
            xmin=constraint_vlines[0],
            xmax=constraint_vlines[1],
        )
        axs[1].vlines(
            x=constraint_vlines,
            ymin=constraint_hlines[0],
            ymax=constraint_hlines[1],
        )

    axs[1].scatter(*zip(*state_data), label="Actual trajectory")
    axs[1].legend()

    plt.show()
