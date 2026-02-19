import ssqpy
import numpy as np
from math import pi, sin, cos

"""
Panda Circle example
Perform a circular motion with the Panda's end effector on the yz-plane
Optionally, constrain the movement within a rectangle inside the circle
"""

# @script-parameter
# Whether to plot error and trajectory
PLOT = False
# @script-parameter
# Whether to add a box constraint within the circle for the end effector
CONSTRAIN = True

if PLOT:
    import matplotlib.pyplot as plt

nq = 9
nv = 9
H = 400
dt = 0.01

# Weights
T_WGT = 3.0
R_WGT = 1.0
V_WGT = 3e-3
U_WGT = 3e-3

# Horizontal and vertical line coordinates for constraints
constraint_hlines = [0.1, 0.41]
constraint_vlines = [-0.15, 0.15]

# Circle properties
cx = 0.3
cy = 0
cz = 0.26
cr = 0.22

DISTANCE = 2 * pi
ω = DISTANCE / (H * dt)
φ0 = 0

# Trajectory
traj = []

# Keep x-coordinate constant
# Move on a circular path on the yz-plane
for k in range(H + 1):
    φk = φ0 + ω * k * dt
    traj.append(np.array([cx, cy + cr * sin(φk), cz + cr * cos(φk)]))

Rd = np.eye(3)
Rd[1, 1] = -1
Rd[2, 2] = -1

# Create an SSQP model with H and dt, from the URDF at the below path
model = ssqpy.model.Model(
    H,
    dt,
    urdf_path="../../assets/urdf/panda/panda.urdf",
    dynamics_mode=ssqpy.model.DynamicsMode.Inverse,
)

# Use the following initial state
# This will act as:
#  1) An initial state for the system at the first time step (x0); this cannot
#     be changed by the solver
#  2) An initial guess for every time step xk; this will be changed by the
#     solver to minimize the cost and satisfy the constraints
# If we wanted to help the solver converge faster, we could also choose to
# provide a more accurate initial guess
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

# Create a box constraint for the position of the end effector
# This will constrain the position within `vlines[0]` <= p[1] <= `vlines[1]` in the
# y-axis and `hlines[0]` <= p[2] <= `hlines[1]` in the z-axis
# Note: here, we only create the constraint; for it to be applied to some stage
# of the horizon, we actually have to add it to the model
lb = np.array((-np.inf, constraint_vlines[0], constraint_hlines[0]))
ub = np.array((np.inf, constraint_vlines[1], constraint_hlines[1]))
box_constraint = ssqpy.model.constraints.FramePositionBound(
    model, "panda_ee", lb, ub, nq, nv
)

for k in range(H):
    # Add a squared pose error cost for the end effector (`"panda_ee"` frame) to
    # follow the generated trajectory, while keeping its orientation at `Rd`
    # We use weights `P_WGT` and `R_WGT` for the translation and rotation
    # respectively
    model.addCost(
        k,
        ssqpy.model.costs.FrameSquaredPoseErrorCost(
            model, traj[k], Rd, "panda_ee", nq, nq, T_WGT, R_WGT
        ),
    )

    # Add regularization costs for joint velocity (v) and control (u)
    model.addCost(k, ssqpy.model.costs.SquaredJointVelocityCost(nq, V_WGT))
    model.addCost(k, ssqpy.model.costs.SquaredControlCost(nq, U_WGT))

    if CONSTRAIN:
        # Add the previously created box constraint to stage k
        model.addConstraints(k, box_constraint)

# Add the same pose error for the terminal stage
model.addCost(
    H,
    ssqpy.model.costs.FrameSquaredPoseErrorCost(
        model, traj[H], Rd, "panda_ee", nq, nq, T_WGT, R_WGT
    ),
)

# There are no controls (u) at the terminal stage, so we cannot add a control
# regularization cost; thus, we only add the joint velocity regularization cost
model.addCost(H, ssqpy.model.costs.SquaredJointVelocityCost(nq, V_WGT))

if CONSTRAIN:
    # Add box constraint to terminal stage as well
    model.addConstraints(H, box_constraint)

# Finalize the model
# This will generate the final model, ready for use by SSQP for optimization
# By passing a single configuration as an init_state, this configuration will
# be applied to all stages (including as an initial stage)
model.finalize(init_states=init_state)

# Initialize an SSQP solver object with the created model
# Parameters are passed as keyword arguments
# ADMM-specific parameters start with "admm_"
solver = ssqpy.solvers.ssqp(
    model,
    loose_success=True,
    tolerance=1e-1,
    use_filter=False,
    accept_worse_update=True,
    # admm_abs_tolerance=1e-3,
    # admm_rel_tolerance=1e-3,
    # admm_adaptive_rho_tolerance=10,
    # admm_rho_update_iter_freq=25,
    # admm_max_iter=1000,
    qp_solver_type=ssqpy.solvers.QPSolverType.HPIPM,
    hpipm_max_iter=100,
    hpipm_tol_stat=1e-3,
    hpipm_tol_comp=1e-3,
    hpipm_tol_eq=1e-3,
    hpipm_tol_ineq=1e-3,
)

# Solve using the SSQP solver and the model we created
state, x, u = solver.solve()

# Write the solution to an output file
# The generated trajectory can be visualized using the `visualize.py` script
solver.write("../data/panda_circle.json")

print("x:")
x.print()
print("u:")
u.print()

# Plot error and end effector position on 2D plane, compared to expected
# unconstrained and constrained position
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
