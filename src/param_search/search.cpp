#include "param_search/search.hpp"

namespace ssqp
{
namespace param_search
{
DynamicsParamGaussNewton::DataPoint::DataPoint(const VecX &x_prev,
                                               const VecX &u,
                                               const VecX &x_pred,
                                               const VecX &x_actual)
  : x_prev(x_prev)
  , u(u)
  , x_pred(x_pred)
  , x_actual(x_actual)
{
}

DynamicsParamGaussNewton::DynamicsParamGaussNewton(
  const std::shared_ptr<model::stagewise::StageDynamics> &dynamics,
  const uint                                              window_size)
  : _dynamics(dynamics)
  , _state_size(dynamics->state_diff_size())
  , _param_size(dynamics->parameter_size())
  , _window_size(window_size)
{
}

void DynamicsParamGaussNewton::addData(const VecX &x, const VecX &u,
                                       const VecX &x_next)
{
    _window.push_back(DataPoint(x, u, _dynamics->evaluate(x, u), x_next));

    if (_window.size() > _window_size)
    {
        _window.pop_front();
    }

    if (_window.size() == _window_size)
    {
        // 1. Initialize stacked residuals (r) and Jacobian (J)
        Eigen::VectorXd r(_window_size * _state_size);
        Eigen::MatrixXd J(_window_size * _state_size, _param_size);

        Eigen::VectorXd theta = _dynamics->parameters();

        // 2. Compute residuals and Jacobian
        for (uint i = 0; i < _window.size(); ++i)
        {
            const DataPoint &p       = _window[i];
            uint             row_idx = i * _state_size;

            r.segment(row_idx, _state_size) = p.x_actual - p.x_pred;

            J.block(i * _state_size, 0, _state_size, _param_size) =
              _dynamics->parameter_gradient(p.x_prev, p.u);
        }

        // 3. Solve equations
        double          lambda  = 1e-4;
        Eigen::MatrixXd H       = J.transpose() * J;
        H.diagonal()           += lambda * H.diagonal();

        Eigen::VectorXd dTheta = H.ldlt().solve(-J.transpose() * r);

        // 4. Update parameters
        // TODO: line search
        theta += dTheta;

        for (uint i = 0; i < 6; ++i)
        {
            if (theta[i] < 1e-3)
            {
                theta[i] = 1e-3;
            }
        }

        _dynamics->update_parameters(theta);
    }
}

HyperParamSearch::HyperParamSearch(
  const std::vector<solvers::ssqp::Params> &hyperparams,
  const std::vector<model::Model>          &problems)
  : _hyperparams(hyperparams)
  , _problems(problems)
{
}

void HyperParamSearch::addHyperparams(const solvers::ssqp::Params &sqp_params)
{
    _hyperparams.push_back(sqp_params);
}

void HyperParamSearch::addProblem(const model::Model &model)
{
    _problems.push_back(model);
}

const solvers::ssqp::Params HyperParamSearch::search() const
{
    double score = std::numeric_limits<double>::infinity();
    const solvers::ssqp::Params *best_params;

    for (const solvers::ssqp::Params &sqp_params : _hyperparams)
    {
        double metric = 0.0;

        for (const model::Model &problem : _problems)
        {
            solvers::ssqp        solver(problem, sqp_params);
            solvers::ssqp::Guess solution(problem.dims(),
                                          problem.linear_dims());

            try
            {
                solvers::SolverState state = solver.solve(solution);

                const solvers::ssqp::ConvergenceData cd =
                  solver.convergenceData();

                metric += cd.iterations + 2.0 * cd.total_duration +
                          20.0 * cd.trajectory_cost +
                          8.0 * cd.trajectory_constraint_violation +
                          cd.trajectory_stage_gap;
            }
            catch (std::runtime_error &e)
            {
            }
        }

        metric /= _problems.size();

        if (metric < score)
        {
            best_params = &sqp_params;
            score       = metric;
        }
    }

    return *best_params;
}
}
}