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
  const uint window_size, const double armijo_beta, const double armijo_c1)
  : _dynamics(dynamics)
  , _state_size(dynamics->state_diff_size())
  , _param_size(dynamics->parameter_size())
  , _window_size(window_size)
  , _armijo_beta(armijo_beta)
  , _armijo_c1(armijo_c1)
{
    _r.resize(_window_size * _state_size);
    _J.resize(_window_size * _state_size, _param_size);
}

bool DynamicsParamGaussNewton::addData(const VecX &x, const VecX &u,
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

        // 4. Armijo Line Search Parameters
        double       alpha = 1.0;
        const VecX   g = _J.transpose() * _r; // Gradient of the cost function
        const double initial_cost = 0.5 * _r.squaredNorm();
        const double descent_slope =
          g.dot(dTheta_log); // Directional derivative

        VecX theta_orig = _dynamics->parameters();
        VecX theta_new  = theta_orig;

        // Line search loop
        while (alpha > 1e-6) // Minimum step size threshold
        {
            // New theta using log-space update
            for (uint j = 0; j < _param_size; ++j)
            {
                theta_new[j] = theta_orig[j] * std::exp(alpha * dTheta_log[j]);
            }

            // Evaluate cost at theta_new
            double new_cost = 0.0;
            for (const auto &p : _window)
            {
                // Temporarily set parameters to evaluate
                _dynamics->update_parameters(theta_new);
                VecX r_i = _dynamics->state_right_minus(
                  p.x_actual,
                  _dynamics->evaluate(p.x_prev, p.u));
                new_cost += 0.5 * r_i.squaredNorm();
            }

            // Check Armijo Condition
            if (new_cost <= initial_cost + _armijo_c1 * alpha * descent_slope)
            {
                break; // Sufficient decrease achieved
            }

            alpha *= _armijo_beta; // Shrink step size
        }

        // Final update
        _dynamics->update_parameters(theta_new);

        return true;
    }

    return false;
}

void DynamicsParamGaussNewton::clearData()
{
    _window.clear();
}

void DynamicsParamGaussNewton::setWindowSize(const uint window_size)
{
    _window_size = window_size;

    if (_window.size() > _window_size)
    {
        // If shrinking window, try to keep the data points that are furthest
        // away from each other, according to jacobian similarity
        std::vector<MatX> jacobians(_window.size());

        for (uint i = 0; i < _window.size(); ++i)
        {
            const DataPoint &p = _window[i];
            jacobians[i]       = _dynamics->parameter_gradient(p.x_prev, p.u);
        }

        std::vector<DataPoint>   selected;
        std::unordered_set<uint> selected_indices;

        selected.push_back(_window[0]);
        selected_indices.emplace(0);

        while (selected.size() < _window_size)
        {
            double best_distance = -1.0;
            uint   best_index    = 0;

            for (uint i = 0; i < _window.size(); ++i)
            {
                if (selected_indices.find(i) != selected_indices.end())
                {
                    continue;
                }

                double nearest = std::numeric_limits<double>::max();

                for (uint j = 0; j < selected.size(); ++j)
                {
                    nearest =
                      std::min(nearest,
                               _jacobianSimilarity(jacobians[i], jacobians[j]));
                }

                if (nearest > best_distance)
                {
                    best_distance = nearest;
                    best_index    = i;
                }
            }

            selected.push_back(_window[best_index]);
            selected_indices.emplace(best_index);
        }

        _window = selected;
    }

    _r.resize(_window_size * _state_size);
    _J.resize(_window_size * _state_size, _param_size);
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