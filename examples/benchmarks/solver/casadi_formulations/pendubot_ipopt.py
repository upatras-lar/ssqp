from problems import run_pendubot_problem

run_pendubot_problem(
    "ipopt",
    lambda nx, nu, ng, H: {
        "ipopt.max_iter": 1000,
        "ipopt.tol": 1e-2,
        "ipopt.acceptable_tol": 1e-3,
        "ipopt.mu_strategy": "adaptive",
        "print_time": True,
    },
    "pendubot_ipopt.json",
)
