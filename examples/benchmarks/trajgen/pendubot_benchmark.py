import ssqpy
import multiprocessing
import sys

sys.path.insert(0, "..")

import benchmark
import generate_pendubot_problems


# Total number of problems to generate
N_problems = 1000

# Small and large horizon values
# Equivalent parameter files must exist for these values
H_small = 25
H_large = 200


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
        f"../data/parameters/pendubot_admm_params_H{horizon}_{dynamics_mode_str}.json"
    )
    benchmark.benchmark(
        problems,
        params_admm,
        f"../data/results/pendubot_admm_H{horizon}_{dynamics_mode_str}.json",
    )

    # =====
    # HPIPM
    # =====
    params_hpipm = benchmark.get_params(
        f"../data/parameters/pendubot_hpipm_params_H{horizon}_{dynamics_mode_str}.json"
    )
    benchmark.benchmark(
        problems,
        params_hpipm,
        f"../data/results/pendubot_hpipm_H{horizon}_{dynamics_mode_str}.json",
    )


H_small_inv_p = multiprocessing.Process(
    target=run,
    args=(
        H_small,
        ssqpy.model.DynamicsMode.Inverse,
    ),
)

H_large_inv_p = multiprocessing.Process(
    target=run,
    args=(
        H_large,
        ssqpy.model.DynamicsMode.Inverse,
    ),
)

H_small_fwd_p = multiprocessing.Process(
    target=run,
    args=(
        H_small,
        ssqpy.model.DynamicsMode.Forward,
    ),
)

H_large_fwd_p = multiprocessing.Process(
    target=run,
    args=(
        H_large,
        ssqpy.model.DynamicsMode.Forward,
    ),
)


H_small_inv_p.start()
H_large_inv_p.start()
H_small_fwd_p.start()
H_large_fwd_p.start()

H_small_inv_p.join()
H_large_inv_p.join()
H_small_fwd_p.join()
H_large_fwd_p.join()
