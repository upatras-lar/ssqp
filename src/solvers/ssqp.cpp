#include "solvers/ssqp.hpp"
#include "sqp_step/filter.hpp"

#include "tracy/Tracy.hpp"

namespace ssqp
{
namespace solvers
{
ssqp::Guess::Guess(const model::Dimensions &dims,
                   const model::Dimensions &lin_dims)
  : x(dims.state, dims.stage_state)
  , u(dims.control, dims.stage_control)
  , lambda(lin_dims.state, lin_dims.stage_state)
  , mu(dims.constraint, dims.stage_constraint)
{
}

ssqp::ssqp(model::Model &model, const Params &params)
  : ssqp(static_cast<const model::Model &>(model), params)
{
    if (!model.finalized())
    {
        model.finalize();
    }
}

ssqp::ssqp(const model::Model &model, const Params &params)
  : _params(params)
  , _model(model)
  , _guess(model.dims(), model.linear_dims())
  , _qp(model.linear_dims())
  , _prev_cost_dx(model.linear_dims().state, model.linear_dims().stage_state)
  , _cost_dx(_prev_cost_dx, false)
  , _prev_cost_du(_guess.u, false)
  , _cost_du(_guess.u, false)
  , _tolerance_init(_params.tolerance)
{
    model::Dimensions              dims     = model.dims();
    const model::DynamicsContainer dynamics = model.dynamics();

    for (uint k = 0; k <= dims.horizon; ++k)
    {
        assert(_guess.x.stage(k).size() == dynamics[k]->init_state().size() &&
               "Initial state of incorrect size passed");
        _guess.x.stage(k) = dynamics[k]->init_state();
    }

    for (uint k = 0; k < dims.horizon; ++k)
    {
        assert(_guess.u.stage(k).size() == dynamics[k]->init_control().size() &&
               "Initial controls of incorrect size passed");
        _guess.u.stage(k) = dynamics[k]->init_control();
    }

    if (_params.use_filter)
    {
        _filter_memory.push_front({ _lf(_guess.x, _guess.u),
                                    _gammaf(_guess.x, _guess.u),
                                    _cf(_guess.x, _guess.u) });
    }

    DEBUG_LOGLN("SSQP initial guess:");
    DEBUG_LOGLN("x:");
    DEBUG_LOGLN(_guess.x);
    DEBUG_LOGLN("u:");
    DEBUG_LOGLN(_guess.u);
}

const model::Model &ssqp::model() const
{
    return _model;
}

SolverState ssqp::solve(Guess &guess)
{
    ZoneScopedN("SSQP: solve");

    _stale_iters = 0;

    SolverState ssqp_state;

    DEBUG_LOGLN("Starting SSQP solver loop");
    auto start = std::chrono::high_resolution_clock::now();

    _setup_qp<true>();
    std::unique_ptr<QPSolver> qp_solver;
    QPSolver::Solution        sol(_qp.dims);

    switch (_params.qp_solver_type)
    {
    case ADMM:
        qp_solver = std::make_unique<admm_qp>(_qp, _params.admm_params);
        break;
    case HPIPM:
        qp_solver = std::make_unique<hpipm_qp>(_qp, _params.hpipm_params);
        break;
    }

    _params.tolerance = _tolerance_init;

    for (_iter = 0; _iter < _params.max_iter; ++_iter)
    {
        _convergence_data.iteration_data.push_back(IterationConvergenceData());

        auto        start_qp = std::chrono::high_resolution_clock::now();
        SolverState state    = qp_solver->solve(sol);
        _convergence_data.iteration_data.back().qp_duration =
          std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start_qp)
            .count();

        switch (state)
        {
        case PRIMAL_INFEASIBLE:
        case DUAL_INFEASIBLE:
            return state;
        default:
            break;
        }

        FlatVec x_n(_guess.x, false), u_n(_guess.u, false);

        _convergence_data.iteration_data.back().qp_iters =
          qp_solver->iterations();

        // Guess (x, u) update
        if (_params.use_filter)
        {
            std::tie(x_n,
                     u_n,
                     _convergence_data.iteration_data.back().step_size) =
              step::step(
                [this](const step::Point &a,
                       const step::Point &b,
                       const step::Point &offset,
                       const double alpha)
                {
                    return step::filter(
                      a,
                      b,
                      _filter_memory,
                      [this](const FlatVec &x, const FlatVec &u)
                      { return _lf(x, u); },
                      [this](const FlatVec &x, const FlatVec &u)
                      { return _gammaf(x, u); },
                      [this](const FlatVec &x, const FlatVec &u)
                      { return _cf(x, u); });
                },
                _guess.x,
                _guess.u,
                sol.dx,
                sol.du,
                _model.dynamics(),
                _params.inv_alpha_scale_factor,
                _params.min_alpha);
        }
        else
        {
            std::tie(x_n,
                     u_n,
                     _convergence_data.iteration_data.back().step_size) =
              step::step(
                [this](const step::Point &a,
                       const step::Point &b,
                       const step::Point &offset,
                       const double alpha)
                {
                    return step::merit(
                      a,
                      b,
                      offset,
                      alpha,
                      [this](const FlatVec &x, const FlatVec &u)
                      {
                          return _lf(x, u) +
                                 _params.merit_mu_gamma * _gammaf(x, u) +
                                 _params.merit_mu_c * _cf(x, u);
                      },
                      [this](const FlatVec &x,
                             const FlatVec &u,
                             FlatVec       &dx,
                             FlatVec       &du)
                      {
                          _lf_grad(x, u, dx, du);

                          FlatVec tmp_x(dx, false), tmp_u(du, false);

                          if (_params.merit_mu_gamma > 0.0)
                          {
                              _gammaf_grad(x, u, tmp_x, tmp_u);
                              dx.vec += _params.merit_mu_gamma * tmp_x.vec;
                              du.vec += _params.merit_mu_gamma * tmp_u.vec;

                              tmp_x.vec.setZero();
                              tmp_u.vec.setZero();
                          }

                          if (_params.merit_mu_c > 0.0)
                          {
                              _cf_grad(x, u, tmp_x, tmp_u);
                              dx.vec += _params.merit_mu_c * tmp_x.vec;
                              du.vec += _params.merit_mu_c * tmp_u.vec;
                          }
                      },
                      _params.merit_armijo_beta);
                },
                _guess.x,
                _guess.u,
                sol.dx,
                sol.du,
                _model.dynamics(),
                _params.inv_alpha_scale_factor,
                _params.min_alpha);
        }

        if (!_params.accept_worse_update &&
            _convergence_data.iteration_data.back().step_size <
              _params.min_alpha)
        {
            if (_params.skip_regularization)
            {
                throw std::runtime_error("LQR: Regularization failed");
            }

            DEBUG_LOGLN("Line search failed, increasing regularization");
            qp_solver->increaseRegularization();
            --_iter;
            continue;
        }

        _guess.x = x_n;
        _guess.u = u_n;

        if (_params.use_filter)
        {
            _filter_memory.push_front({ _lf(_guess.x, _guess.u),
                                        _gammaf(_guess.x, _guess.u),
                                        _cf(_guess.x, _guess.u) });

            if (_filter_memory.size() > _params.filter_size)
            {
                _filter_memory.pop_back();
            }
        }

        DEBUG_LOGLN("Current x: ");
        DEBUG_LOGLN(_guess.x);
        DEBUG_LOGLN("Current u: ");
        DEBUG_LOGLN(_guess.u);

        _guess.lambda = sol.lambda;
        _guess.mu     = sol.mu;

        // Termination
        if ((ssqp_state =
               _check_termination<Params::calculate_full_violations>()) !=
            ONGOING)
        {
            _convergence_data.total_duration =
              std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start)
                .count();

            _printIterationInfo();
            LOGLN("SSQP: Total runtime: " << _convergence_data.total_duration
                                          << "ms");

            guess = _guess;
            return ssqp_state;
        }

        if (_params.loose_success)
        {
            _prev_cost_dx = _cost_dx;
            _prev_cost_du = _cost_du;
        }

        if (_params.tolerance_final > 0.0 &&
            _params.tolerance > _params.tolerance_final)
        {
            _params.tolerance += (_params.tolerance_final - _tolerance_init) /
                                 (_params.tolerance_update_steps == 0
                                    ? _params.max_iter
                                    : _params.tolerance_update_steps);
        }

        if (_iter % 40 == 0)
        {
            _printIterationHeader();
        }
        _printIterationInfo();

        // Skip gradient calculation in _setup_qp, since they have been
        // calculated in _check_termination
        _setup_qp<false>();
        qp_solver->reset();

        FrameMarkNamed("SQP Iteration");
    }

    _convergence_data.total_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start)
        .count();

    LOGLN("SSQP: Total runtime: " << _convergence_data.total_duration << "ms");

    guess = _guess;
    return MAX_ITER_REACHED;
}

bool ssqp::write(const std::string &filename)
{
    using namespace nlohmann;
    static const model::Dimensions &dims = _model.dims();
    static const double            &dt   = _model.dt();

    std::ofstream out;
    out.open(filename);
    if (!out)
    {
        return false;
    }

    ordered_json j;
    j["solver"]              = ordered_json({});
    j["solver"]["tolerance"] = _params.tolerance;

    j["convergence"]                         = ordered_json({});
    j["convergence"]["runtime_ms"]           = _convergence_data.total_duration;
    j["convergence"]["iterations"]           = _iter + 1;
    j["convergence"]["trajectory_cost"]      = _lf(_guess.x, _guess.u);
    j["convergence"]["trajectory_stage_gap"] = _gammaf(_guess.x, _guess.u);
    j["convergence"]["trajectory_constraint_violation"] =
      _cf(_guess.x, _guess.u);
    j["convergence"]["iteration_data"] = ordered_json::array();

    for (uint i = 0; i < _convergence_data.iteration_data.size(); ++i)
    {
        ordered_json iteration_data({
          {         "qp_iterations",_convergence_data.iteration_data[i].qp_iters                                    },
          {            "runtime_us", _convergence_data.iteration_data[i].qp_duration },
          {             "step_size",   _convergence_data.iteration_data[i].step_size },
          {               "norm_dx",     _convergence_data.iteration_data[i].norm_dx },
          {               "norm_du",     _convergence_data.iteration_data[i].norm_du },
          { "first_violation_stage",
           _convergence_data.iteration_data[i].first_violation_stage                },
          {             "violation",   _convergence_data.iteration_data[i].violation }
        });

        if (_convergence_data.iteration_data[i].gap_violations.size() ==
            dims.horizon)
        {
            iteration_data["stage_gap_violations"] = ordered_json::array();
            iteration_data["stage_constraint_violations"] =
              ordered_json::array();
            for (uint k = 0; k < dims.horizon; ++k)
            {
                iteration_data["stage_gap_violations"].push_back(
                  _convergence_data.iteration_data[i].gap_violations[k]);
                iteration_data["stage_constraint_violations"].push_back(
                  _convergence_data.iteration_data[i].constraint_violations[k]);
            }

            iteration_data["stage_constraint_violations"].push_back(
              _convergence_data.iteration_data[i]
                .constraint_violations[dims.horizon]);
        }

        j["convergence"]["iteration_data"].push_back(iteration_data);
    }

    j["properties"]            = ordered_json({});
    j["properties"]["horizon"] = dims.horizon;
    j["properties"]["dt"]      = dt;
    j["properties"]["stages"]  = ordered_json::array();
    j["stage_data"]            = ordered_json::array();

    // TODO: Assuming floatingBase is the same every stage
    for (uint k = 0; k < dims.horizon; ++k)
    {
        uint                     nq      = _model.nq(k);
        uint                     nv      = _model.nv(k);
        uint                     njoints = _model.njoints(k);
        uint                     nforces = _model.nforces(k);
        std::vector<std::string> ordered_force_frames =
          _model.orderedForceFrames(k);
        bool                     floating_base = _model.floatingBase();
        std::vector<std::string> actuated_joints =
          _model.orderedActuatedJoints();

        j["properties"]["stages"].push_back(ordered_json({
          {                 "size",   dims.stage_state[k] },
          {              "control", dims.stage_control[k] },
          {                   "nq",                    nq },
          {                   "nv",                    nv },
          {              "njoints",               njoints },
          {              "nforces",               nforces },
          { "ordered_force_frames",  ordered_force_frames },
          {        "floating_base",         floating_base },
          {         "state_joints",       actuated_joints }
        }));

        ordered_json stage_data;

        VecX u;

        if (floating_base)
        {
            VecX x = _guess.x.stage(k);
            u      = _guess.u.stage(k);

            stage_data["position"]    = x.head(3);
            stage_data["orientation"] = x.segment(3, 4);
            stage_data["state"]       = x.segment(7, njoints);
            stage_data["control"]     = u.segment(6, njoints);
        }
        else
        {
            u = _guess.u.stage(k);

            stage_data["state"]   = _guess.x.stage(k);
            stage_data["control"] = u.head(nv);
        }

        if (nforces > 0)
        {
            stage_data["forces"] = u.tail(3 * nforces);
        }

        j["stage_data"].push_back(stage_data);
    }

    out << std::setw(4) << j;
    out.close();

    return true;
}

const ssqp::ConvergenceData ssqp::convergenceData() const
{
    return _convergence_data;
}

void ssqp::_printIterationHeader() const
{
    LOG(std::setw(18));
    LOG("SQP Iteration |");
    LOG(std::setw(18));
    LOG("Norm/dx |");
    LOG(std::setw(18));
    LOG("Norm/du |");
    LOG(std::setw(18));
    LOG("Violation Stage |");
    LOG(std::setw(18));
    LOG("Violation |");
    LOG(std::setw(18));
    LOG("QP Iterations |");
    LOG(std::setw(18));
    LOG("QP Runtime (us) |");
    LOG(std::setw(18));
    LOGLN("Step Size");
}

void ssqp::_printIterationInfo() const
{
    const IterationConvergenceData &icd =
      _convergence_data.iteration_data.back();

    LOG(std::setw(16));
    LOG(_iter << " |");
    LOG(std::setw(16) << std::setprecision(5));
    LOG(icd.norm_dx << " |");
    LOG(std::setw(16) << std::setprecision(5));
    LOG(icd.norm_du << " |");
    LOG(std::setw(16));
    if (icd.first_violation_stage < 0.0)
    {
        LOG("-" << " |");
    }
    else
    {
        LOG(icd.first_violation_stage << " |");
    }
    LOG(std::setw(16) << std::setprecision(5));
    LOG(icd.violation << " |");
    LOG(std::setw(16));
    LOG(icd.qp_iters << " |");
    LOG(std::setw(16));
    LOG(icd.qp_duration << " |");
    LOG(std::setw(16));
    LOGLN(icd.step_size);
}

double ssqp::_lf(const FlatVec &x, const FlatVec &u) const
{
    ZoneScopedN("SSQP: _lf");

    static const uint           horizon = _model.horizon();
    double                      sum     = 0.0;
    const model::CostsContainer costs   = _model.costs();

    for (uint k = 0; k < horizon; ++k)
    {
        sum += costs[k]->evaluate(x.stage(k), u.stage(k))(0);
    }

    return sum + costs[horizon]->evaluate(x.stage(horizon), VecX::Zero(1))(0);
}

void ssqp::_lf_grad(const FlatVec &x, const FlatVec &u, FlatVec &dx,
                    FlatVec &du) const
{
    ZoneScopedN("SSQP: _lf_grad");

    static const uint           horizon = _model.horizon();
    const model::CostsContainer costs   = _model.costs();

    VecX dxk, duk;
    for (uint k = 0; k < horizon; ++k)
    {
        costs[k]->gradient(x.stage(k), u.stage(k), dxk, duk);
        dx.stage(k) = dxk;
        du.stage(k) = duk;
    }

    costs[horizon]->gradient(x.stage(horizon), VecX::Zero(1), dxk, duk);
    dx.stage(horizon) = dxk;
}

double ssqp::_gammaf(const FlatVec &x, const FlatVec &u) const
{
    ZoneScopedN("SSQP: _gammaf");

    static const uint              horizon  = _model.horizon();
    const model::DynamicsContainer dynamics = _model.dynamics();
    double                         sum      = 0.0;

    for (uint k = 0; k < horizon; ++k)
    {
        sum += (dynamics[k]->state_right_minus(
                  x.stage(k + 1),
                  dynamics[k]->evaluate(x.stage(k), u.stage(k))))
                 .lpNorm<Eigen::Infinity>();
    }

    return sum;
}

double ssqp::_gammaf_softmax(const FlatVec &x, const FlatVec &u) const
{
    static const uint              horizon  = _model.horizon();
    const model::DynamicsContainer dynamics = _model.dynamics();
    double                         sum      = 0.0;

    for (uint k = 0; k < horizon; ++k)
    {
        Eigen::VectorXd r = dynamics[k]->state_right_minus(
          x.stage(k + 1),
          dynamics[k]->evaluate(x.stage(k), u.stage(k)));

        double soft_inf =
          (_params.merit_grad_softmax_beta * r.array().abs()).exp().sum();
        soft_inf = std::log(soft_inf) / _params.merit_grad_softmax_beta;

        sum += soft_inf;
    }

    return sum;
}

void ssqp::_gammaf_grad(const FlatVec &x, const FlatVec &u, FlatVec &dx,
                        FlatVec &du) const
{
    ZoneScopedN("SSQP: _gammaf_grad");

    static const uint              horizon  = _model.horizon();
    const model::DynamicsContainer dynamics = _model.dynamics();

    MatX dXk, dUk;
    for (uint k = 0; k < horizon; ++k)
    {
        VecX f   = dynamics[k]->evaluate(x.stage(k), u.stage(k));
        VecX r_k = dynamics[k]->state_right_minus(x.stage(k + 1), f);

        VecX a = (_params.merit_grad_softmax_beta * r_k.array().abs()).exp();
        r_k    = (r_k.array().sign() * a.array()) / a.sum();

        const uint nq = _model.nq(k);
        const uint nv = _model.nv(k);

        MatX Jr_inv_wrt_x = MatX::Zero(nv, nv);
        _model.dDifference(x.stage(k + 1).head(nq),
                           f.head(nq),
                           Jr_inv_wrt_x,
                           0);

        MatX Jr_inv_wrt_f = MatX::Zero(nv, nv);
        _model.dDifference(x.stage(k + 1).head(nq),
                           f.head(nq),
                           Jr_inv_wrt_f,
                           1);

        dynamics[k]->gradient(x.stage(k), u.stage(k), dXk, dUk);

        // J_r = [ Jr_inv_wrt_f, 0; 0, -I ];
        MatX J_r                  = MatX::Zero(2 * nv, 2 * nv);
        J_r.topLeftCorner(nv, nv) = Jr_inv_wrt_f;
        J_r.bottomRightCorner(nv, nv).diagonal().setConstant(-1.0);

        dx.stage(k) += (J_r * dXk).transpose() * r_k;
        du.stage(k) += (J_r * dUk).transpose() * r_k;

        dx.stage(k + 1).head(nv) += Jr_inv_wrt_x.transpose() * r_k.head(nv);
        dx.stage(k + 1).tail(nv) += r_k.tail(nv);
    }
}

double ssqp::_cf(const FlatVec &x, const FlatVec &u) const
{
    ZoneScopedN("SSQP: _cf");

    static const uint                 horizon     = _model.horizon();
    const model::ConstraintsContainer constraints = _model.constraints();
    double                            sum         = 0.0;

    VecX c;
    for (uint k = 0; k < horizon; ++k)
    {
        if (constraints[k] != nullptr)
        {
            c = constraints[k]->evaluate(x.stage(k), u.stage(k));
            sum +=
              constraints[k]
                ->right_minus(c, constraints[k]->lower_bound())
                .cwiseMin(0)
                .cwiseAbs()
                .cwiseMax(
                  constraints[k]->right_minus(c, constraints[k]->upper_bound()))
                .lpNorm<Eigen::Infinity>();
        }
    }

    if (constraints[horizon] != nullptr)
    {
        c    = constraints[horizon]->evaluate(x.stage(horizon), VecX::Zero(1));
        sum += constraints[horizon]
                 ->right_minus(c, constraints[horizon]->lower_bound())
                 .cwiseMin(0)
                 .cwiseAbs()
                 .cwiseMax(constraints[horizon]->right_minus(
                   c,
                   constraints[horizon]->upper_bound()))
                 .lpNorm<Eigen::Infinity>();
    }

    return sum;
}

double ssqp::_cf_softmax(const FlatVec &x, const FlatVec &u) const
{
    static const uint                 horizon     = _model.horizon();
    const model::ConstraintsContainer constraints = _model.constraints();

    double sum = 0.0;

    VecX c;

    for (uint k = 0; k < horizon; ++k)
    {
        if (!constraints[k])
        {
            continue;
        }

        c = constraints[k]->evaluate(x.stage(k), u.stage(k));

        VecX v =
          constraints[k]
            ->right_minus(c, constraints[k]->lower_bound())
            .cwiseMin(0)
            .cwiseAbs()
            .cwiseMax(
              constraints[k]->right_minus(c, constraints[k]->upper_bound()));

        sum += softmax_max(v, _params.merit_grad_softmax_beta);
    }

    if (constraints[horizon])
    {
        c = constraints[horizon]->evaluate(x.stage(horizon), VecX::Zero(1));

        VecX v = constraints[horizon]
                   ->right_minus(c, constraints[horizon]->lower_bound())
                   .cwiseMin(0)
                   .cwiseAbs()
                   .cwiseMax(constraints[horizon]->right_minus(
                     c,
                     constraints[horizon]->upper_bound()));

        sum += softmax_max(v, _params.merit_grad_softmax_beta);
    }

    return sum;
}

void ssqp::_cf_grad(const FlatVec &x, const FlatVec &u, FlatVec &dx,
                    FlatVec &du) const
{
    ZoneScopedN("SSQP: _cf_grad");

    static const uint                 horizon     = _model.horizon();
    const model::ConstraintsContainer constraints = _model.constraints();

    MatX dXk, dUk;
    for (uint k = 0; k < horizon; ++k)
    {
        VecX c_k = constraints[k]->evaluate(x.stage(k), u.stage(k));
        VecX lb  = constraints[k]->lower_bound();
        VecX ub  = constraints[k]->upper_bound();

        VecX v =
          constraints[k]->right_minus(c_k, lb).cwiseMin(0).cwiseAbs().cwiseMax(
            constraints[k]->right_minus(c_k, ub));

        VecX w =
          (_params.merit_grad_softmax_beta * (v.array() - v.maxCoeff())).exp();
        w /= w.sum();

        VecX dv_dc = VecX::Zero(v.size());
        for (int i = 0; i < v.size(); ++i)
        {
            if (c_k[i] < lb[i])
            {
                dv_dc[i] = -1.0;
            }
            else if (c_k[i] > ub[i])
            {
                dv_dc[i] = +1.0;
            }
        }

        c_k = w.cwiseProduct(dv_dc);

        constraints[k]->gradient(x.stage(k), u.stage(k), dXk, dUk);

        dx.stage(k).noalias() += dXk.transpose() * c_k;
        du.stage(k).noalias() += dUk.transpose() * c_k;
    }

    VecX c_H = constraints[horizon]->evaluate(x.stage(horizon), VecX::Zero(1));
    VecX lb  = constraints[horizon]->lower_bound();
    VecX ub  = constraints[horizon]->upper_bound();

    VecX v = constraints[horizon]
               ->right_minus(c_H, lb)
               .cwiseMin(0)
               .cwiseAbs()
               .cwiseMax(constraints[horizon]->right_minus(c_H, ub));

    VecX w =
      (_params.merit_grad_softmax_beta * (v.array() - v.maxCoeff())).exp();
    w /= w.sum();

    VecX dv_dc = VecX::Zero(v.size());
    for (int i = 0; i < v.size(); ++i)
    {
        if (c_H[i] < lb[i])
        {
            dv_dc[i] = -1.0;
        }
        else if (c_H[i] > ub[i])
        {
            dv_dc[i] = +1.0;
        }
    }

    c_H = w.cwiseProduct(dv_dc);

    constraints[horizon]->gradient(x.stage(horizon), VecX::Zero(1), dXk, dUk);

    dx.stage(horizon).noalias() += dXk.transpose() * c_H;
}
}
}