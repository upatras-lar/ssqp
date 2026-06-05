#include "param_search/search.hpp"

namespace ssqp
{
namespace param_search
{
DynamicsParamGaussNewton::DataPoint::DataPoint(const VecX &x_prev,
                                               const VecX &u,
                                               const VecX &x_actual)
  : x_prev(x_prev)
  , u(u)
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
    _r.resize(_window_size * _state_size);
    _J.resize(_window_size * _state_size, _param_size);
}

void DynamicsParamGaussNewton::addData(const VecX &x, const VecX &u,
                                       const VecX &x_next)
{
    _window.push_back(DataPoint(x, u, x_next));

    if (_window.size() > _window_size)
    {
        // Pre-calculate Jacobians for each data point
        std::vector<MatX> jacobians(_window.size());

        for (uint i = 0; i < _window.size(); ++i)
        {
            const DataPoint &p = _window[i];
            jacobians[i]       = _dynamics->parameter_gradient(p.x_prev, p.u);
        }

        // New point is the last
        const MatX &J_new = jacobians.back();

        // Find most similar data point using similarity in Jacobians
        uint   closest_idx  = 0;
        double closest_dist = _jacobianSimilarity(jacobians[0], J_new);

        for (uint i = 1; i < _window_size; ++i)
        {
            const double dist = _jacobianSimilarity(jacobians[i], J_new);

            if (dist < closest_dist)
            {
                closest_idx  = i;
                closest_dist = dist;
            }
        }

        // Replace most similar data point with new
        _window[closest_idx] = _window.back();
        _window.pop_back();

        // Replace its Jacobian with the new one
        jacobians[closest_idx] = jacobians.back();
        jacobians.pop_back();

        VecX theta = _dynamics->parameters();

        // 2. Compute residuals and Jacobian
        for (uint i = 0; i < _window_size; ++i)
        {
            const DataPoint &p       = _window[i];
            uint             row_idx = i * _state_size;

            _r.segment(row_idx, _state_size) =
              _dynamics->state_right_minus(p.x_actual,
                                           _dynamics->evaluate(p.x_prev, p.u));

            // Use precalculated jacobians
            _J.block(i * _state_size, 0, _state_size, _param_size) =
              -jacobians[i];
        }

        // Map Jacobian to log-space
        for (uint j = 0; j < _param_size; ++j)
        {
            _J.col(j) *= theta[j];
        }

        // 3. Solve equations
        double lambda  = 1e-4;
        MatX   H       = _J.transpose() * _J;
        H.diagonal()  += lambda * VecX::Ones(_param_size);

        Eigen::LDLT ldlt = H.ldlt();

        if (ldlt.info() != Eigen::Success)
        {
            logln_at(LogLevel::Warning, "Warning: (J^T)J LDLT failed");
        }

        VecX dTheta_log = ldlt.solve(-_J.transpose() * _r);

        // 4. Update parameters
        // TODO: line search
        for (uint j = 0; j < _param_size; ++j)
        {
            theta[j] *= std::exp(dTheta_log[j]);
        }

        _dynamics->update_parameters(theta);
    }
}

double DynamicsParamGaussNewton::_jacobianSimilarity(const MatX &J1,
                                                     const MatX &J2)
{
    return (J1 - J2).squaredNorm();
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