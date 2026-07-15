import ssqpy
import numpy as np

H = 2500
dt = 0.001

V_WGT = 1e-4
U_WGT = 1e-5
C_WGT = 5.0

DESIRED_CONFIG = np.array((-1.0, 0.0, 1.0, 0.0))

model = ssqpy.model.Model(
    H,
    dt,
    urdf_path="../../../assets/urdf/double_pendulum/pendubot_cont.urdf",
    actuated_joints=[0],
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

    if k > H - 50:
        model.addCost(k, config_cost)

model.finalize(terminal_cost=config_cost)

# ssqpy.setLogLevel(ssqpy.LogLevel.Debug)
# print(model.getConstraints(0).output_size())

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.tolerance = 1e-2
ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM

hpipm_params = ssqpy.solvers.hpipmParams()
hpipm_params.tol_comp = 1e-2
hpipm_params.tol_stat = 1e-2
hpipm_params.tol_eq = 1e-2
hpipm_params.tol_ineq = 1e-2

ssqp_params.hpipmParams = hpipm_params

solver = ssqpy.solvers.ssqp(model, ssqp_params)

state, x, u = solver.solve()

print("xs:")
x.print()
print("us:")
u.print()

solver.write("trajectory.json")
