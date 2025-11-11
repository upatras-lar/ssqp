#include <iostream>
#include <cmath>
#include <vector>

#include "solvers/ssqp.hpp"
#include "solvers/admm_qp.hpp"
#include "model/Model.hpp"
#include "model/stagewise/predefined/Costs.hpp"
#include "pinocchio_wrapper/PinocchioWrapper.hpp"

#define STATE_SIZE 19
#define VEL_SIZE   18
#define HORIZON    400
#define dt         0.01
#define P_WGT      10
#define R_WGT      1
#define V_WGT      1e-5
#define U_WGT      1e-5
#define TERM_WGT   1

using namespace ssqp::model::stagewise::predefined;

int main()
{
    Eigen::Vector3d base_pos;
    base_pos << 0, 0, 0.4;

    Eigen::VectorXd init = Eigen::VectorXd::Zero(37);
    init[3]              = 1.0;

    init.segment(7, 12) << 0.0, 0.8, -1.6, 0, 0.8, -1.6, 0, 0.8, -1.6, 0, 0.8,
      -1.6;

    ssqp::model::Model model("examples/urdf/a1.urdf", HORIZON, dt, {}, true);

    /*
    Eigen::VectorXd pos;
    Eigen::MatrixXd rot;
    std::tie(pos, rot) =
      model.forwardKinematics(Eigen::VectorXd::Zero(STATE_SIZE), "trunk");
    LOGLN(pos.transpose());
    */

    for (uint k = 0; k < HORIZON; ++k)
    {
        model.addCost<costs::FrameSquaredPositionErrorCost>(k,
                                                            model,
                                                            base_pos,
                                                            "trunk",
                                                            STATE_SIZE,
                                                            VEL_SIZE,
                                                            P_WGT);

        model.addCost<costs::SquaredControlCost>(k, VEL_SIZE, U_WGT);
    }

    model.addCost<costs::FrameSquaredPositionErrorCost>(HORIZON,
                                                        model,
                                                        base_pos,
                                                        "trunk",
                                                        STATE_SIZE,
                                                        VEL_SIZE,
                                                        P_WGT);

    model.finalize({}, { init });

    ssqp::solvers::ssqp::Params params;
    // params.loose_success = false;
    params.tolerance = 3e-2;

    ssqp::solvers::admm_qp::Params admm_params;
    admm_params.abs_tolerance = 6e-3;
    admm_params.rel_tolerance = 1e-3;

    // params.loose_success_tolerance = 1e-3;
    ssqp::solvers::ssqp        solver(model, params, admm_params);
    ssqp::solvers::ssqp::Guess solution(model.dims(), model.linear_dims());
    ssqp::solvers::SolverState state = solver.solve(solution);

    std::cout << "Solver state: " << ssqp::solvers::to_string(state)
              << std::endl;

    if (state == ssqp::solvers::SolverState::SUCCESS ||
        state == ssqp::solvers::SolverState::LOOSE_SUCCESS)
    {
        std::cout << "x: " << std::endl << solution.x << std::endl;
        std::cout << "u: " << std::endl << solution.u << std::endl;
    }

    return 0;
}