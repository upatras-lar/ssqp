"""
Centroidal Voronoi Tessellation (CVT) sampler for n-DOF robot state space.
State: (q1, ..., qn, dq1, ..., dqn)
Domain: [q_lo, q_hi] x [v_lo, v_hi] per dimension (not necessarily symmetric).
Lloyd relaxation via Monte Carlo centroid estimation.
"""

import ssqpy
import numpy as np


def cvt(
    n_generators: int,
    q_lo: np.ndarray = [-np.pi, -np.pi],
    q_hi: np.ndarray = [np.pi, np.pi],
    v_lo: np.ndarray = [-25.0, -25.0],
    v_hi: np.ndarray = [25.0, 25.0],
    n_iterations: int = 200,
    n_samples_per_iter: int = 10_000,
    seed: int = 42,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Run CVT Lloyd relaxation in a 2*nq dimensional state space.

    Parameters
    ----------
    n_generators      : number of CVT generators (output configurations)
    q_lo, q_hi        : (nq,) position lower / upper bounds
    v_lo, v_hi        : (nv,) velocity lower / upper bounds
    n_iterations      : max Lloyd iterations
    n_samples_per_iter: Monte Carlo samples per iteration
    seed              : RNG seed
    tol               : convergence threshold on max generator displacement

    Returns
    -------
    generators : (n_generators, 2*nq) array — columns [q1..qn, dq1..dqn]
    """
    q_lo = np.asarray(q_lo, dtype=float)
    q_hi = np.asarray(q_hi, dtype=float)
    v_lo = np.asarray(v_lo, dtype=float)
    v_hi = np.asarray(v_hi, dtype=float)
    assert q_lo.shape == q_hi.shape == v_lo.shape == v_hi.shape

    lo = np.concatenate([q_lo, v_lo])
    hi = np.concatenate([q_hi, v_hi])
    ndim = len(lo)
    span = hi - lo

    rng = np.random.default_rng(seed)
    generators = rng.uniform(0.0, 1.0, size=(n_generators, ndim))

    for iteration in range(n_iterations):
        samples = rng.uniform(0.0, 1.0, size=(n_samples_per_iter, ndim))

        diff = samples[:, np.newaxis, :] - generators[np.newaxis, :, :]
        assignments = np.argmin(np.sum(diff**2, axis=-1), axis=-1)

        new_generators = np.zeros_like(generators)
        counts = np.zeros(n_generators, dtype=np.int64)
        np.add.at(new_generators, assignments, samples)
        np.add.at(counts, assignments, 1)

        nonempty = counts > 0
        new_generators[nonempty] /= counts[nonempty, np.newaxis]
        new_generators[~nonempty] = generators[~nonempty]

        displacement = np.max(np.linalg.norm(new_generators - generators, axis=-1))
        generators = new_generators

        if displacement < tol:
            print(f"Converged at iteration {iteration + 1} (disp={displacement:.2e})")
            break
    else:
        print(f"Max iterations ({n_iterations}), disp={displacement:.2e}")

    return generators * span + lo


def generate_pendubot_problems(n, horizon, dynamics_mode):
    V_WGT = 1e-3
    U_WGT = 1e-3 if dynamics_mode == ssqpy.model.DynamicsMode.Inverse else 1e-4
    TIP_WGT = 5.0
    C_WGT = 4.0

    nq = 2
    nv = 2
    dt = 0.01

    problems = []

    init_states = cvt(n, v_lo=[-1.0, -1.0], v_hi=[1.0, 1.0])

    for i in range(n):
        model = ssqpy.model.Model(
            horizon,
            dt,
            urdf_path="../../assets/urdf/double_pendulum/pendubot.urdf",
            actuated_joints=[0],
            dynamics_mode=dynamics_mode,
        )

        vel_cost = ssqpy.model.costs.SquaredJointVelocityCost(nq, V_WGT)
        u_cost = ssqpy.model.costs.SquaredControlCost(nv, U_WGT)
        tip_cost = ssqpy.model.costs.FrameSquaredTranslationErrorCost(
            model, np.array((0.0, 0.0, 0.1)), "tip", nq, nv, TIP_WGT
        )
        config_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(
            np.array((np.pi, 0.0)), nq, nv, C_WGT, False
        )

        for k in range(horizon):
            model.addCost(k, vel_cost)
            model.addCost(k, u_cost)
            # model.addCost(k, tip_cost)

            if k > 100:
                model.addCost(k, config_cost)

        model.addCost(horizon, vel_cost)
        # model.addCost(horizon, tip_cost)
        model.addCost(horizon, config_cost)

        model.finalize(init_states=init_states[i])

        problems.append(model)

    return problems
