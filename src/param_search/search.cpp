#include "param_search/search.hpp"

namespace ssqp
{
namespace param_search
{
ParamSearch::ParamSearch(const std::vector<solvers::ssqp::Params> &hyperparams,
                         const std::vector<model::Model>          &problems)
  : _hyperparams(hyperparams)
  , _problems(problems)
{
}

void ParamSearch::addHyperparams(const solvers::ssqp::Params &sqp_params)
{
    _hyperparams.push_back(sqp_params);
}

void ParamSearch::addProblem(const model::Model &model)
{
    _problems.push_back(model);
}

const solvers::ssqp::Params ParamSearch::search() const
{
    double                       score = -1.0;
    const solvers::ssqp::Params *best_params;

    for (const solvers::ssqp::Params &sqp_params : _hyperparams)
    {
        // TODO: decide metric
        double metric = 0.0;

        for (const model::Model &problem : _problems)
        {
            solvers::ssqp        solver(problem, sqp_params);
            solvers::ssqp::Guess solution(problem.dims(),
                                          problem.linear_dims());

            try
            {
                solvers::SolverState state = solver.solve(solution);

                const solvers::ssqp::ConvergenceData icd =
                  solver.convergenceData();

                metric += 10.0 / icd.iterations + 1.0 / icd.total_duration;
            }
            catch (std::runtime_error &e)
            {
            }
        }

        metric /= _problems.size();

        if (metric > score)
        {
            best_params = &sqp_params;
            score       = metric;
        }
    }

    return *best_params;
}
}
}