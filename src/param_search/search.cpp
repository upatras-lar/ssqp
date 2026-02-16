#include "param_search/search.hpp"

namespace ssqp
{
namespace param_search
{
ParamSearch::ParamSearch(
  const std::vector<std::pair<solvers::ssqp::Params, solvers::admm_qp::Params>>
                                  &hyperparams,
  const std::vector<model::Model> &problems)
  : _hyperparams(hyperparams)
  , _problems(problems)
{
}

void ParamSearch::addHyperparams(const solvers::ssqp::Params    &sqp_params,
                                 const solvers::admm_qp::Params &admm_params)
{
    _hyperparams.push_back({ sqp_params, admm_params });
}

void ParamSearch::addProblem(const model::Model &model)
{
    _problems.push_back(model);
}

const std::tuple<solvers::ssqp::Params, solvers::admm_qp::Params, model::Model>
ParamSearch::search() const
{
    double                          score = -1.0;
    const solvers::ssqp::Params    *best_sqp_params;
    const solvers::admm_qp::Params *best_admm_params;
    const model::Model             *best_problem;

    for (const std::pair<solvers::ssqp::Params, solvers::admm_qp::Params>
           &param_pair : _hyperparams)
    {
        for (const model::Model &problem : _problems)
        {
            solvers::ssqp solver(problem, param_pair.first, param_pair.second);
            solvers::ssqp::Guess solution(problem.dims(),
                                          problem.linear_dims());
            solvers::SolverState state = solver.solve(solution);

            const solvers::ssqp::ConvergenceData icd = solver.convergenceData();

            // TODO: decide metric
            double metric = 1.0 / icd.iterations;
            if (metric > score)
            {
                best_sqp_params  = &param_pair.first;
                best_admm_params = &param_pair.second;
                best_problem     = &problem;
                score            = metric;
            }
        }
    }

    return { *best_sqp_params, *best_admm_params, *best_problem };
}
}
}