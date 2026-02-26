import ssqpy
import multiprocessing
import sys

sys.path.insert(0, "..")

import benchmark
import get_panda_problems


def run(dynamics_mode):
    dynamics_mode_str = (
        "forward"
        if dynamics_mode == ssqpy.model.DynamicsMode.Forward
        else "inverse"
    )

    problems = get_panda_problems.get_panda_problems(dynamics_mode)

    params_admm = benchmark.get_params(
        f"../data/parameters/panda_admm_params_{dynamics_mode_str}.json"
    )
    benchmark.benchmark(
        problems,
        params_admm,
        f"../data/results/panda_admm_{dynamics_mode_str}.json",
    )

    params_hpipm = benchmark.get_params(
        f"../data/parameters/panda_hpipm_params_{dynamics_mode_str}.json"
    )
    benchmark.benchmark(
        problems,
        params_hpipm,
        f"../data/results/panda_hpipm_{dynamics_mode_str}.json",
    )


inv = multiprocessing.Process(
    target=run, args=(ssqpy.model.DynamicsMode.Inverse,)
)

fwd = multiprocessing.Process(
    target=run, args=(ssqpy.model.DynamicsMode.Forward,)
)

inv.start()
fwd.start()

inv.join()
fwd.join()
