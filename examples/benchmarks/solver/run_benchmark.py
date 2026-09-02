import argparse
import os

from problems import *


def parse_args():
    parser = argparse.ArgumentParser(description="Solver benchmark script")

    parser.add_argument(
        "problem",
        choices=[
            "pendubot",
            "go2_standing",
            "go2_jump",
            "go2_frog",
            "panda_circle",
        ],
        help="The benchmarking problem to run",
    )

    parser.add_argument(
        "-s",
        "--solver",
        nargs="?",
        choices=[
            "ssqp-admm",
            "ssqp-hpipm",
            "fatrop",
            "ipopt",
            "aligator",
            "mim_solvers",
        ],
        default="ssqp-admm",
        help="The solver to benchmark",
    )

    return parser.parse_args()


args = parse_args()

# Outer-iteration caps, sized per problem rather than one flat number for
# everything. Values are set from what's actually been observed to be
# needed for a converging solve on each problem (with headroom), not
# guessed: pendubot's slowest converging case (Aligator) needs ~371
# iterations; go2_standing's slowest (Fatrop, post mu_init fix) needs
# ~36; go2_jump's ssqp-admm needs as many as ~900 (right up against the
# old flat 1000 cap) to escape a long plateau, so it -- and go2_frog,
# which is untested at this end and has 2x go2_jump's horizon (bigger,
# more expensive NLP per iteration on top of needing possibly more of
# them) -- keep the full 1000 rather than risk cutting off a slow-but-
# real convergence. This mainly saves wall-clock time on the small
# problems, where grinding through a full 1000-iteration budget after
# converging in under 40 is pure waste. panda_circle is likewise
# untested at this end -- fixed-base and contact-free like pendubot, but
# a much bigger NLP (400-step horizon vs. pendubot's underactuated
# 2-dof arm) -- so it also keeps the full 1000 rather than guessing at a
# tighter cap.
MAX_ITER_BY_PROBLEM = {
    "pendubot": 500,
    "go2_standing": 200,
    "go2_jump": 1000,
    "go2_frog": 1000,
    "panda_circle": 1000,
}
max_iter = MAX_ITER_BY_PROBLEM[args.problem]

problem_wrapper_t = None
constructor_parameters = {}
solve_parameters = {}
if args.solver == "ssqp-admm" or args.solver == "ssqp-hpipm":
    import ssqpy
    from ssqp_formulations.SSQPProblem import SSQPProblem

    problem_wrapper_t = SSQPProblem

    constructor_parameters["solver_mode"] = (
        ssqpy.model.SolverMode.ForwardDynamics
    )
    if args.problem == "pendubot":
        constructor_parameters["actuated_joints"] = [0]
    elif args.problem in ("go2_standing", "go2_jump", "go2_frog"):
        constructor_parameters["free_flyer"] = True

    solve_parameters["tolerance"] = 1e-2
    solve_parameters["max_iter"] = max_iter

    if args.solver == "ssqp-admm":
        solve_parameters["qp_solver"] = "admm"
        # panda_circle's per-stage QP is harder for ADMM to fully converge
        # on than pendubot/go2's -- at the shared default of 100 it pegs
        # every single QP solve at the cap (never detects convergence, see
        # the rho_update_iter_freq note below) and the outer SQP loop
        # stalls with a persistent ~0.01 constraint violation at one stage
        # that never shrinks, even after 1000 outer iterations. Verified
        # directly against the container: 100 never converges, 200
        # doesn't either, but 400 converges in 3 outer iterations.
        solve_parameters["admm_max_iter"] = (
            400 if args.problem == "panda_circle" else 100
        )
        solve_parameters["admm_tol"] = 1e-3
        # rho_update_iter_freq also gates ADMM's own termination check (see
        # ssqp's admm_qp.hpp), so setting it equal to admm_max_iter (100)
        # forces every ADMM solve to burn its full iteration budget even
        # after it has actually converged -- it can never exit early. Using
        # ssqp's own default (25) lets ADMM detect convergence well before
        # the cap, several times per QP solve.
        solve_parameters["admm_adaptive_rho_tolerance"] = 5
        solve_parameters["admm_rho_update_iter_freq"] = 25
        solve_parameters["admm_warm_start"] = True
    else:
        solve_parameters["qp_solver"] = "hpipm"
        solve_parameters["hpipm_tol"] = 1e-3

elif args.solver == "mim_solvers":
    from mim_solvers_formulations.MimSolversProblem import (
        MimSolversProblem,
        PendubotActuationModel,
    )
    import pinocchio as pin
    import crocoddyl

    problem_wrapper_t = MimSolversProblem

    if args.problem == "pendubot":
        pin_model = pin.buildModelFromUrdf(
            "urdf/double_pendulum/pendubot.urdf"
        )
        state = crocoddyl.StateMultibody(pin_model)
        custom_actuation = PendubotActuationModel(state, pin_model)

        constructor_parameters["floating_base"] = False
        constructor_parameters["actuation_model"] = custom_actuation
    elif args.problem in ("go2_standing", "go2_jump", "go2_frog"):
        constructor_parameters["floating_base"] = True
    elif args.problem == "panda_circle":
        # Explicit, rather than relying on MimSolversProblem's
        # buildModelFromUrdf/JointModelFreeFlyer-fails try/except fallback:
        # adding a free-flyer joint on top of a fixed-base URDF like
        # panda.urdf doesn't raise, it just silently adds 6 spurious DOF at
        # the root, so that fallback would never trigger here.
        constructor_parameters["floating_base"] = False

    solve_parameters["tolerance"] = 1e-2
    solve_parameters["max_iter"] = max_iter
    # CSQP's own convergence break only happens inside
    # `qp_iters_ % rho_update_interval_ == 0` (csqp.cpp), so with
    # rho_update_interval == max_qp_iters the QP loop can never detect
    # convergence and exit early (confirmed: every pendubot/go2 run pegs
    # QP iters at the cap). Lowering rho_update_interval alone (e.g. to 25)
    # made the floating-base problems (go2_standing/go2_jump) diverge --
    # more frequent rho re-estimation is unstable on these contact-rich
    # QPs. Instead, keep rho_update_interval at ssqp/crocoddyl's stable
    # cadence and just give the QP subsolver more raw budget, so it gets
    # more convergence-check points (300/600/900...) without the unstable
    # rho-update frequency.
    solve_parameters["max_qp_iters"] = 300
    solve_parameters["eps_abs"] = 1e-3
    solve_parameters["eps_rel"] = 1e-3
    solve_parameters["adaptive_rho_tolerance"] = 1
    solve_parameters["rho_update_interval"] = 100

elif args.solver == "aligator":
    from aligator_formulations.AligatorProblem import AligatorProblem

    problem_wrapper_t = AligatorProblem

    if args.problem == "pendubot":
        constructor_parameters["actuated_joints"] = [0]
    elif args.problem in ("go2_standing", "go2_jump", "go2_frog"):
        constructor_parameters["free_flyer"] = True

    solve_parameters["tol"] = 1e-2
    solve_parameters["mu_init"] = 1e-2
    # NOTE: go2_jump fails to converge here ("Convergence failure": preg
    # balloons past 1e5 while stuck at prim_err ~0.7-0.9, alpha collapsing
    # to 1e-6) -- the solver can't find a feasible direction from the naive
    # (static, zero-velocity/control) initial guess against the
    # contact-switching + exact terminal foot-placement constraints. Tried
    # mu_init in {1e-4, 1e-2, 10.0}; all three get stuck at a similarly
    # infeasible plateau (different cost/prim_err per value, no clear
    # winner, none actually converge), so this is left at the shared
    # default rather than picking one unresolved trade-off over another.
    # Likely needs a warm-started/kinematically-sensible initial guess
    # rather than further mu_init tuning.
    solve_parameters["max_iters"] = max_iter

elif args.solver == "fatrop" or args.solver == "ipopt":
    from casadi_formulations.CasadiProblem import CasadiProblem

    problem_wrapper_t = CasadiProblem

    constructor_parameters["casadi_plugin"] = args.solver
    if args.problem == "pendubot":
        constructor_parameters["actuated_joints"] = [0]
    elif args.problem in ("go2_standing", "go2_jump", "go2_frog"):
        constructor_parameters["free_flyer"] = True

    if args.solver == "fatrop":
        # nx/nu/ng/N + structure_detection="manual" are filled in
        # automatically by CasadiProblem.solve() (it's the only thing that
        # knows the per-stage dimensions once the problem is built), so we
        # only need to pass Fatrop's own numerical tolerances/iteration cap
        # here.
        solve_parameters["fatrop"] = {"tol": 1e-2, "max_iter": max_iter}
        if args.problem in ("go2_standing", "go2_jump"):
            # Both fall into an unrecoverable restoration phase from the
            # zero-velocity/static initial guess (CasADi's Fatrop plugin
            # only exposes a subset of Fatrop's own options -- e.g.
            # bound_relax_factor isn't in that subset, hence mu_init here
            # instead). A larger initial barrier parameter is more
            # conservative about how hard it pushes toward the boundary of
            # go2's many contact/velocity constraints in early iterations.
            solve_parameters["fatrop"]["mu_init"] = 1.0
    else:
        solve_parameters["ipopt"] = {
            "max_iter": max_iter,
            "tol": 1e-2,
            "acceptable_tol": 1e-3,
            "mu_strategy": "adaptive",
        }
        if args.problem == "go2_jump":
            # Same restoration-phase failure as Fatrop above (stuck past
            # iteration 450 without escaping); a larger initial barrier
            # parameter gives the same relief here.
            solve_parameters["ipopt"]["mu_init"] = 1.0
        solve_parameters["print_time"] = True

problems = {
    "pendubot": make_pendubot_problem,
    "go2_standing": make_go2_standing_problem,
    "go2_jump": make_go2_jump_problem,
    "go2_frog": make_go2_frog_problem,
    "panda_circle": make_panda_circle_problem,
}


problem_wrapper = problems[args.problem](
    problem_wrapper_t, constructor_parameters
)

problem_wrapper.solve(solve_parameters)

save_path = f"data/{args.solver}/"
save_path = save_path.replace("-", "_")

if not os.path.exists(save_path):
    os.makedirs(save_path)

problem_wrapper.save(save_path + args.problem + ".json")
