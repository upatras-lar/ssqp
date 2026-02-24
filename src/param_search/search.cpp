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

const std::pair<solvers::ssqp::Params, model::Model> ParamSearch::search() const
{
    double                       score = -1.0;
    const solvers::ssqp::Params *best_sqp_params;
    const model::Model          *best_problem;

    for (const solvers::ssqp::Params &sqp_params : _hyperparams)
    {
        for (const model::Model &problem : _problems)
        {
            solvers::ssqp        solver(problem, sqp_params);
            solvers::ssqp::Guess solution(problem.dims(),
                                          problem.linear_dims());

            double metric = -2.0;
            try
            {
                solvers::SolverState state = solver.solve(solution);

                const solvers::ssqp::ConvergenceData icd =
                  solver.convergenceData();

                // TODO: decide metric
                metric = 10.0 / icd.iterations + 1.0 / icd.total_duration;
            }
            catch (std::runtime_error &e)
            {
            }

            if (metric > score)
            {
                best_sqp_params = &sqp_params;
                best_problem    = &problem;
                score           = metric;
            }
        }
    }

    return { *best_sqp_params, *best_problem };
}
}
}