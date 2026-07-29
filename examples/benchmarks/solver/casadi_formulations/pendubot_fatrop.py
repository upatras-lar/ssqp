from problems import run_pendubot_problem

run_pendubot_problem(
    "fatrop",
    lambda nx, nu, ng, H: {
        "structure_detection": "manual",
        "nx": nx,
        "nu": nu,
        "ng": ng,
        "N": H - 1,
        "expand": True,
    },
    "pendubot_fatrop.json",
)
