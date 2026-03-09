import ssqpy
from generate_pendubot_problems import generate_pendubot_problems
from generate_panda_problems import generate_panda_feasible_region_problems
from generate_go2_problems import generate_go2_walk_problems

# problem = generate_pendubot_problems(
#     20,
#     dynamics_mode=ssqpy.model.DynamicsMode.Forward,
#     horizon=200,
# )[10]
# problem = generate_panda_feasible_region_problems(
#     40, ssqpy.model.DynamicsMode.Inverse
# )[39]
problem = generate_go2_walk_problems(10, ssqpy.model.DynamicsMode.Inverse)[5]


# Pendubot params
# params = ssqpy.solvers.ssqpParams(
#     "data/parameters/pendubot_admm_params_forward.json"
# )
# params.tolerance = 1e-3
# params.max_iter = 2000
# params.admmParams.abs_tolerance = 1e-4
# params.admmParams.rel_tolerance = 1e-4
# params.admmParams.adaptive_rho_tolerance = 10.0
# params.admmParams.rho_update_iter_freq = 5

# Panda params
# params = ssqpy.solvers.ssqpParams(
#     "data/parameters/panda_admm_params_inverse.json"
# )
# params.max_iter = 1000

# GO2 params
params = ssqpy.solvers.ssqpParams()
params.max_iter = 100
params.tolerance = 1e-2

params.admmParams.abs_tolerance = 1e-3
params.admmParams.rel_tolerance = 1e-3
params.admmParams.adaptive_rho_tolerance = 10.0
params.admmParams.max_iter = 500

solver = ssqpy.solvers.ssqp(problem, params)

state, _, _ = solver.solve()

print("Solver State:", state)

solver.write("test.json")
