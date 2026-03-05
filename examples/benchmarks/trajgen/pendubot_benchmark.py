import ssqpy
import multiprocessing
import benchmark
import generate_pendubot_problems


# Total number of problems to generate
N_problems = 50

H = 200


def run(horizon, dynamics_mode):
    dynamics_mode_str = (
        "forward"
        if dynamics_mode == ssqpy.model.DynamicsMode.Forward
        else "inverse"
    )

    problems = generate_pendubot_problems.generate_pendubot_problems(
        N_problems, horizon, dynamics_mode
    )

    # ====
    # ADMM
    # ====
    params_admm = benchmark.get_params(
        f"data/parameters/pendubot_admm_params_{dynamics_mode_str}.json",
        max_iter=2000,
    )
    benchmark.benchmark(
        problems,
        params_admm,
        f"data/results/pendubot_admm_{dynamics_mode_str}.json",
    )

    """
    # =====
    # HPIPM
    # =====
    params_hpipm = benchmark.get_params(
        f"data/parameters/pendubot_hpipm_params_{dynamics_mode_str}.json"
    )
    benchmark.benchmark(
        problems,
        params_hpipm,
        f"data/results/pendubot_hpipm_{dynamics_mode_str}.json",
    )
    """


H_inv_p = multiprocessing.Process(
    target=run,
    args=(
        H,
        ssqpy.model.DynamicsMode.Inverse,
    ),
)

H_fwd_p = multiprocessing.Process(
    target=run,
    args=(
        H,
        ssqpy.model.DynamicsMode.Forward,
    ),
)


H_inv_p.start()
H_fwd_p.start()

H_inv_p.join()
H_fwd_p.join()
