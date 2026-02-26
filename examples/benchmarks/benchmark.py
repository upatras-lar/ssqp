import ssqpy
import multiprocessing
import json
from os import remove


def get_params(filepath):
    """
    Load parameters from provided filepath, altering tolerance and max iterations
    to get a full run
    """
    sqp_params = ssqpy.solvers.ssqpParams(filepath)
    sqp_params.tolerance = 1e-6
    sqp_params.max_iter = 500

    return sqp_params


def benchmark(problems, sqp_params, save_to):
    """
    Benchmark given problems using the provided parameters and save all
    results to provided filename
    """
    results = []

    tmp_filepath = f"../data/tmp{multiprocessing.current_process().pid}.json"

    for problem in problems:
        solver = ssqpy.solvers.ssqp(problem, sqp_params)
        solver.solve()
        solver.write(tmp_filepath)

        with open(tmp_filepath, "r") as f:
            data = json.load(f)

        results.append(data)

    remove(tmp_filepath)

    with open(save_to, "w") as f:
        f.write(json.dumps(results, indent=4))
