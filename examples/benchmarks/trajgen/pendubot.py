import ssqpy
import json
import sys

sys.path.insert(0, "..")

import generate_pendubot_problems


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

    for problem in problems:
        solver = ssqpy.solvers.ssqp(problem, sqp_params)
        solver.solve()
        solver.write("../data/tmp.json")

        with open("../data/tmp.json", "r") as f:
            data = json.load(f)

        results.append(data)

    with open(save_to, "w") as f:
        f.write(json.dumps(results, indent=4))


# Total number of problems to generate
N_problems = 1000

# Small and large horizon values
# Equivalent parameter files must exist for these values
H_small = 25
H_large = 200

# ===============================
# Small Horizon, Inverse Dynamics
# ===============================
inverse_problems_small = generate_pendubot_problems.generate_pendubot_problems(
    N_problems, H_small, ssqpy.model.DynamicsMode.Inverse
)

# ====
# ADMM
# ====
inverse_params_admm_H_small = get_params(
    f"../data/parameters/pendubot_admm_params_H{H_small}_inverse.json"
)
benchmark(
    inverse_problems_small,
    inverse_params_admm_H_small,
    f"../data/results/pendubot_admm_H{H_small}_inverse.json",
)

# =====
# HPIPM
# =====
inverse_params_hpipm_H_small = get_params(
    f"../data/parameters/pendubot_hpipm_params_H{H_small}_inverse.json"
)
benchmark(
    inverse_problems_small,
    inverse_params_hpipm_H_small,
    f"../data/results/pendubot_hpipm_H{H_small}_inverse.json",
)

# ===============================
# Large Horizon, Inverse Dynamics
# ===============================
inverse_problems_large = generate_pendubot_problems.generate_pendubot_problems(
    N_problems, H_large, ssqpy.model.DynamicsMode.Inverse
)

# ====
# ADMM
# ====
inverse_params_admm_H_large = get_params(
    f"../data/parameters/pendubot_admm_params_H{H_large}_inverse.json"
)
benchmark(
    inverse_problems_large,
    inverse_params_admm_H_large,
    f"../data/results/pendubot_admm_H{H_large}_inverse.json",
)

# =====
# HPIPM
# =====
inverse_params_hpipm_H_large = get_params(
    f"../data/parameters/pendubot_hpipm_params_H{H_large}_inverse.json"
)
benchmark(
    inverse_problems_large,
    inverse_params_hpipm_H_large,
    f"../data/results/pendubot_hpipm_H{H_large}_inverse.json",
)

# ===============================
# Small Horizon, Forward Dynamics
# ===============================
forward_problems_small = generate_pendubot_problems.generate_pendubot_problems(
    N_problems, H_small, ssqpy.model.DynamicsMode.Forward
)

# ===
# ADMM
# ===
forward_params_admm_H_small = get_params(
    f"../data/parameters/pendubot_admm_params_H{H_small}_forward.json"
)
benchmark(
    forward_problems_small,
    forward_params_admm_H_small,
    f"../data/results/pendubot_admm_H{H_small}_forward.json",
)

# =====
# HPIPM
# =====
forward_params_hpipm_H_small = get_params(
    f"../data/parameters/pendubot_hpipm_params_H{H_small}_forward.json"
)
benchmark(
    forward_problems_small,
    forward_params_hpipm_H_small,
    f"../data/results/pendubot_hpipm_H{H_small}_forward.json",
)

# ===============================
# Large Horizon, Forward Dynamics
# ===============================
forward_problems_large = generate_pendubot_problems.generate_pendubot_problems(
    N_problems, H_large, ssqpy.model.DynamicsMode.Forward
)

# ====
# ADMM
# ====
forward_params_admm_H_large = get_params(
    f"../data/parameters/pendubot_admm_params_H{H_large}_forward.json"
)
benchmark(
    forward_problems_large,
    forward_params_admm_H_large,
    f"../data/results/pendubot_admm_H{H_large}_forward.json",
)

# =====
# HPIPM
# =====
forward_params_hpipm_H_large = get_params(
    f"../data/parameters/pendubot_hpipm_params_H{H_large}_forward.json"
)
benchmark(
    forward_problems_large,
    forward_params_hpipm_H_large,
    f"../data/results/pendubot_hpipm_H{H_large}_forward.json",
)
