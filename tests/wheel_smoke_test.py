"""
Smoke test for built wheels (see [tool.cibuildwheel] in pyproject.toml)
Solves a short Panda reaching problem from the embedded preset with both QP
solvers, which exercises Pinocchio, HPIPM/BLASFEO and the ADMM solver
"""

import numpy as np
import ssqpy

H, dt = 20, 0.05

init_state = np.zeros(18)
init_state[:7] = [0.0, -0.9, 0.0, -2.6, 0.0, 1.7, 0.8]

target = np.array([0.4, 0.1, 0.3])
Rd = np.diag([1.0, -1.0, -1.0])

for qp_solver in (ssqpy.solvers.QPSolverType.ADMM, ssqpy.solvers.QPSolverType.HPIPM):
    model = ssqpy.model.Model(
        H,
        dt,
        from_preset=ssqpy.model.Preset.Panda,
        solver_mode=ssqpy.model.SolverMode.InverseDynamics,
    )

    for k in range(H + 1):
        model.addCost(
            k,
            ssqpy.model.costs.FrameSquaredPoseErrorCost(
                model, "panda_ee", target, Rd, 1.0, 1.0
            ),
        )
        model.addCost(k, ssqpy.model.costs.SquaredJointVelocityCost(model, 1e-3))
        if k < H:
            model.addCost(k, ssqpy.model.costs.SquaredControlCost(model, 1e-3))

    model.finalize(init_states=init_state)

    solver = ssqpy.solvers.ssqp(model, qp_solver_type=qp_solver)
    state, x, u = solver.solve()

    print(f"{qp_solver}: {state}")
    assert state == ssqpy.solvers.SolverState.Success, state
