#include <iostream>
#include <cmath>

#include "solvers/ssqp.hpp"
#include "model/Model.hpp"

#define HORIZON 100
#define dt      0.05

int main()
{
    ssqp::model::Model model("examples/urdf/iiwa.urdf", HORIZON, dt);
    std::cout << 1 << std::endl;

    ssqp::solvers::ssqp        solver(model);
    ssqp::solvers::ssqp::Guess solution(model.dims(), model.linear_dims());
    ssqp::solvers::SolverState state = solver.solve(solution);

    std::cout << "Solver state: " << ssqp::solvers::to_string(state)
              << std::endl;

    if (state == ssqp::solvers::SolverState::SUCCESS ||
        state == ssqp::solvers::SolverState::LOOSE_SUCCESS)
    {
        std::cout << "x: " << solution.x << std::endl;
        std::cout << "u: " << solution.u << std::endl;
    }

    return 0;
}