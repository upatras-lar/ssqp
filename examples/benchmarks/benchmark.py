import ssqpy
import multiprocessing
import json
from os import remove


def get_params(filepath, max_iter=1000):
    """
    Load parameters from provided filepath, altering tolerance and max iterations
    to get a full run
    """
    sqp_params = ssqpy.solvers.ssqpParams(filepath)
    sqp_params.tolerance = 1e-3
    sqp_params.max_iter = max_iter

    sqp_params.admmParams.abs_tolerance = 1e-4
    sqp_params.admmParams.rel_tolerance = 1e-4

    return sqp_params


def benchmark(problems, sqp_params, save_to):
    """
    Benchmark given problems using the provided parameters and save all
    results to provided filename
    """
    results = []

    pid = multiprocessing.current_process().pid

    tmp_filepath = f"../data/tmp{pid}.json"

    for i, problem in enumerate(problems):
        solver = ssqpy.solvers.ssqp(problem, sqp_params)
        solver.solve()
        solver.write(tmp_filepath)

        with open(tmp_filepath, "r") as f:
            data = json.load(f)

        results.append(data)

        print(f"Process {pid} progress: {(i + 1) / len(problems) * 100}%")

    remove(tmp_filepath)

    with open(save_to, "w") as f:
        f.write(json.dumps(results, indent=4))
