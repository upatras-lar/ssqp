import ssqpy
import numpy as np

H = 200
dt = 0.01

V_WGT = 1e-2
U_WGT = 1e-2
C_WGT = 5.0

DESIRED_CONFIG = np.array((-1.0, 0.0, 1.0, 0.0))

model = ssqpy.model.Model(
    H,
    dt,
    urdf_path="../../../assets/urdf/double_pendulum/pendubot_cont.urdf",
    # actuated_joints=[0],
    solver_mode=ssqpy.model.SolverMode.ForwardDynamics,
)

nq = model.getnq()
nv = model.getnv()
nu = model.getnu()

vel_cost = ssqpy.model.costs.SquaredJointVelocityCost(model, V_WGT)
u_cost = ssqpy.model.costs.SquaredControlCost(model, U_WGT)
config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
    model, DESIRED_CONFIG, C_WGT
)

for k in range(H):
    model.addCost(k, vel_cost)
    model.addCost(k, u_cost)

    if k > 150:
        model.addCost(k, config_cost)

model.finalize(terminal_cost=config_cost)

# ssqpy.setLogLevel(ssqpy.LogLevel.Debug)
# print(model.getConstraints(0).output_size())

solver = ssqpy.solvers.ssqp(
    model,
    tolerance=1e-2,
    qp_solver_type=ssqpy.solvers.QPSolverType.HPIPM,
    admm_warm_start=False,
    # admm_reset_rho=False,
    # admm_reset_y=False,
)

state, x, u = solver.solve()

print("xs:")
x.print()
print("us:")
u.print()
