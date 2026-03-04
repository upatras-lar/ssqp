import ssqpy
import multiprocessing
import benchmark
import generate_go2_problems

N_problems = 50


def run(dynamics_mode):
    dynamics_mode_str = (
        "forward"
        if dynamics_mode == ssqpy.model.DynamicsMode.Forward
        else "inverse"
    )

    problems = generate_go2_problems.generate_go2_walk_problems(
        N_problems, dynamics_mode
    )

    params = ssqpy.solvers.ssqpParams()
    params.tolerance = 1e-2
    params.max_iter = 100
    params.accept_worse_update = True

    params.admmParams.abs_tolerance = 1e-3
    params.admmParams.rel_tolerance = 1e-3
    params.admmParams.max_iter = 500

    benchmark.benchmark(
        problems,
        params,
        f"data/results/go2_admm_{dynamics_mode_str}.json",
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
