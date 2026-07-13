#include "solvers/ssqp.hpp"
#include "sqp_step/filter.hpp"

#include "tracy/Tracy.hpp"

namespace ssqp
{
namespace solvers
{
auto to_compile_time(bool b)
{
    using T = std::integral_constant<bool, true>;
    using F = std::integral_constant<bool, false>;
    using V = std::variant<T, F>;
    return b ? V { T {} } : V { F {} };
}

std::pair<FlatVec, FlatVec> loadTrajectory(const std::string &filepath)
{
    using namespace nlohmann;

    std::ifstream in;
    in.open(filepath);
    if (!in)
    {
        throw std::runtime_error("SSQP trajectory file could not be opened");
    }

    ordered_json j = json::parse(in);

    const uint &horizon = j["properties"]["horizon"];

    uint x_size = 0;
    uint u_size = 0;

    std::vector<uint> x_stage_sizes(horizon + 1);
    std::vector<uint> u_stage_sizes(horizon);

    std::vector<uint> stage_nforces(horizon);

    for (uint k = 0; k < horizon; ++k)
    {
        x_stage_sizes[k] = j["properties"]["stages"][k]["size"];
        u_stage_sizes[k] = j["properties"]["stages"][k]["control"];

        x_size += x_stage_sizes[k];
        u_size += u_stage_sizes[k];

        stage_nforces[k] = j["properties"]["stages"][k]["nforces"];
    }

    x_stage_sizes[horizon]  = j["properties"]["stages"][horizon]["size"];
    x_size                 += x_stage_sizes[horizon];

    FlatVec x(x_size, x_stage_sizes);
    FlatVec u(u_size, u_stage_sizes);

    const model::SolverMode solver_mode =
      model::str_to_DM.at(j["properties"]["solver_mode"]);
    const bool &loaded_from_xml = j["properties"]["xml"];

    for (uint k = 0; k < horizon; ++k)
    {
        if (loaded_from_xml)
        {
            x.stage(k).head(3) = Eigen::Map<VecX>(
              static_cast<std::vector<double>>(j["stage_data"][k]["position"])
                .data(),
              3);
            x.stage(k).segment(3, 4) =
              Eigen::Map<VecX>(static_cast<std::vector<double>>(
                                 j["stage_data"][k]["orientation"])
                                 .data(),
                               4);
            x.stage(k).tail(x_stage_sizes[k] - 7) = Eigen::Map<VecX>(
              static_cast<std::vector<double>>(j["stage_data"][k]["state"])
                .data(),
              x_stage_sizes[k] - 7);
        }
        else
        {
            x.stage(k) = Eigen::Map<VecX>(
              static_cast<std::vector<double>>(j["stage_data"][k]["state"])
                .data(),
              x_stage_sizes[k]);
        }

        uint offset = 0;

        if (solver_mode == model::SolverMode::ForwardDynamics)
        {
            offset = 6;
            u.stage(k).head(6).setZero();
        }

        u.stage(k).segment(offset, u_stage_sizes[k] - 3 * stage_nforces[k]) =
          Eigen::Map<VecX>(
            static_cast<std::vector<double>>(j["stage_data"][k]["control"])
              .data(),
            u_stage_sizes[k] - 3 * stage_nforces[k]);
        u.stage(k).tail(3 * stage_nforces[k]) = Eigen::Map<VecX>(
          static_cast<std::vector<double>>(j["stage_data"][k]["forces"]).data(),
          3 * stage_nforces[k]);
    }

    if (loaded_from_xml)
    {
        x.stage(horizon).head(3) = Eigen::Map<VecX>(
          static_cast<std::vector<double>>(j["stage_data"][horizon]["position"])
            .data(),
          3);
        x.stage(horizon).segment(3, 4) =
          Eigen::Map<VecX>(static_cast<std::vector<double>>(
                             j["stage_data"][horizon]["orientation"])
                             .data(),
                           4);
        x.stage(horizon).tail(x_stage_sizes[horizon] - 7) = Eigen::Map<VecX>(
          static_cast<std::vector<double>>(j["stage_data"][horizon]["state"])
            .data(),
          x_stage_sizes[horizon] - 7);
    }
    else
    {
        x.stage(horizon) = Eigen::Map<VecX>(
          static_cast<std::vector<double>>(j["stage_data"][horizon]["state"])
            .data(),
          x_stage_sizes[horizon]);
    }

    return { x, u };
}

ssqp::Params::Params(const std::string &filepath)
{
    using namespace nlohmann;

    std::ifstream in;
    in.open(filepath);
    if (!in)
    {
        throw std::runtime_error("SSQP parameter file could not be opened");
    }

    ordered_json j = json::parse(in);

    max_iter               = j["max_iter"];
    tolerance              = j["tolerance"];
    tolerance_final        = j["tolerance_final"];
    tolerance_update_steps = j["tolerance_update_steps"];

    loose_success             = j["loose_success"];
    loose_success_tolerance   = j["loose_success_tolerance"];
    loose_success_stale_iters = j["loose_success_stale_iters"];

    if (j["line_search_method"] == "merit")
    {
        line_search_method = LineSearchMethod::Merit;
    }
    else
    {
        line_search_method = LineSearchMethod::Filter;
    }

    filter_size = j["filter_size"];

    merit_mu_gamma    = j["merit_mu_gamma"];
    merit_mu_c        = j["merit_mu_c"];
    merit_armijo_beta = j["merit_armijo_beta"];

    inv_alpha_scale_factor = j["inv_alpha_scale_factor"];
    min_alpha              = j["min_alpha"];

    accept_worse_update = j["accept_worse_update"];
    skip_regularization = j["skip_regularization"];

    if (j["qp_solver_type"] == "HPIPM")
    {
#ifdef WITH_HPIPM
        qp_solver_type = QPSolverType::HPIPM;
        hpipm_params   = hpipm_qp::Params(j["hpipm"]);
#else
        throw std::runtime_error("SSQP was not compiled with HPIPM support!");
#endif
    }
    else
    {
        qp_solver_type = QPSolverType::ADMM;
        admm_params    = admm_qp::Params(j["admm"]);
    }
}

void ssqp::Params::print() const
{
    std::cout << "SSQP Parameters" << std::endl;
    std::cout << "===============" << std::endl;
    std::cout << "Maximum iterations: " << max_iter << std::endl;
    std::cout << "Tolerance: " << tolerance << std::endl;
    if (tolerance_final > 0.0)
    {
        std::cout << "Final tolerance: " << tolerance_final << std::endl;
        std::cout << "Update steps: " << tolerance_update_steps << std::endl;
    }

    std::cout << "---------------" << std::endl;
    std::cout << "Loose success enabled: " << loose_success << std::endl;
    if (loose_success)
    {
        std::cout << "Loose success tolerance: " << loose_success_tolerance
                  << std::endl;
        std::cout << "Loose success stale iterations: "
                  << loose_success_stale_iters << std::endl;
    }

    std::cout << "---------------" << std::endl;
    std::cout << "Line search method: "
              << (line_search_method == LineSearchMethod::Filter ? "filter"
                                                                 : "merit")
              << std::endl;
    if (line_search_method == LineSearchMethod::Filter)
    {
        std::cout << "Filter size: " << filter_size << std::endl;
    }
    else if (line_search_method == LineSearchMethod::Merit)
    {
        std::cout << "Merit mu (gamma): " << merit_mu_gamma << std::endl;
        std::cout << "Merit mu (c): " << merit_mu_c << std::endl;
        std::cout << "Merit armijo beta: " << merit_armijo_beta << std::endl;
    }

    std::cout << "---------------" << std::endl;
    std::cout << "Inverse alpha scaling factor: " << inv_alpha_scale_factor
              << std::endl;
    std::cout << "Minimum alpha: " << min_alpha << std::endl;

    std::cout << "---------------" << std::endl;
    std::cout << "Accept worse update enabled: " << accept_worse_update
              << std::endl;
    std::cout << "Skip regularization: " << skip_regularization << std::endl;

    std::cout << "---------------" << std::endl;
    std::cout << "QP Solver: "
              << (qp_solver_type == QPSolverType::ADMM ? "ADMM" : "HPIPM")
              << std::endl;
    std::cout << "---------------" << std::endl;

    if (qp_solver_type == QPSolverType::ADMM)
    {
        admm_params.print();
    }
    else if (qp_solver_type == QPSolverType::HPIPM)
    {
#ifdef WITH_HPIPM
        hpipm_params.print();
#else
        throw std::runtime_error("SSQP was not compiled with HPIPM support!");
#endif
    }
}

void ssqp::Params::write(nlohmann::ordered_json &j) const
{
    j["max_iter"]               = max_iter;
    j["tolerance"]              = tolerance;
    j["tolerance_final"]        = tolerance_final;
    j["tolerance_update_steps"] = tolerance_update_steps;

    j["loose_success"]             = loose_success;
    j["loose_success_tolerance"]   = loose_success_tolerance;
    j["loose_success_stale_iters"] = loose_success_stale_iters;

    j["line_search_method"] =
      line_search_method == LineSearchMethod::Filter ? "filter" : "merit";

    j["filter_size"] = filter_size;

    j["merit_mu_gamma"]    = merit_mu_gamma;
    j["merit_mu_c"]        = merit_mu_c;
    j["merit_armijo_beta"] = merit_armijo_beta;

    j["inv_alpha_scale_factor"] = inv_alpha_scale_factor;
    j["min_alpha"]              = min_alpha;

    j["accept_worse_update"] = accept_worse_update;
    j["skip_regularization"] = skip_regularization;

    if (qp_solver_type == QPSolverType::ADMM)
    {
        j["qp_solver_type"] = "ADMM";
        admm_params.write(j);
    }
    else if (qp_solver_type == QPSolverType::HPIPM)
    {
#ifdef WITH_HPIPM
        j["qp_solver_type"] = "HPIPM";
        hpipm_params.write(j);
#else
        throw std::runtime_error("SSQP was not compiled with HPIPM support!");
#endif
    }
}

bool ssqp::Params::write(const std::string &filepath) const
{
    using namespace nlohmann;

    std::ofstream out;
    out.open(filepath);
    if (!out)
    {
        return false;
    }

    ordered_json j;

    write(j);

    out << std::setw(4) << j;
    out.close();

    return true;
}

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

    const model::Dimensions lin_dims = model.linear_dims();

    uint max_x = *std::max_element(lin_dims.stage_state.begin(),
                                   lin_dims.stage_state.end());
    uint max_u = *std::max_element(lin_dims.stage_control.begin(),
                                   lin_dims.stage_control.end());

    _tmp.x1.resize(max_x);
    _tmp.u1.resize(max_u);
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
  , _reduced_output(false)
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

    if (_params.line_search_method == LineSearchMethod::Filter)
    {
        _filter_memory.push_front({ _lf(_guess.x, _guess.u),
                                    _gammaf(_guess.x, _guess.u),
                                    _cf(_guess.x, _guess.u) });
    }

    if (_params.constraint_tolerance < 0.0)
    {
        _params.constraint_tolerance = _params.tolerance;
    }

    logln_at(LogLevel::Debug, "SSQP initial guess:");
    logln_at(LogLevel::Debug, "x:");
    logln_at(LogLevel::Debug, _guess.x);
    logln_at(LogLevel::Debug, "u:");
    logln_at(LogLevel::Debug, _guess.u);
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

    logln_at(LogLevel::Debug, "Starting SSQP solver loop");
    auto start_time = std::chrono::high_resolution_clock::now();

    auto xml_model = to_compile_time(_model.xml());

    std::visit([this](auto xml) { _setup_qp<true, xml.value>(); }, xml_model);
    std::unique_ptr<QPSolver> qp_solver;
    QPSolver::Solution        sol(_qp.dims);

    switch (_params.qp_solver_type)
    {
    case ADMM:
        qp_solver = std::make_unique<admm_qp>(_qp, _params.admm_params);
        break;
    case HPIPM:
#ifdef WITH_HPIPM
        qp_solver = std::make_unique<hpipm_qp>(_qp, _params.hpipm_params);
#else
        throw std::runtime_error("SSQP was not compiled with HPIPM support!");
#endif
        break;
    }

    _params.tolerance = _tolerance_init;

    for (_iter = 0; _iter < _params.max_iter; ++_iter)
    {
        _convergence_data.iteration_data.push_back(IterationConvergenceData());

        auto        start_iter_time = std::chrono::high_resolution_clock::now();
        SolverState state           = qp_solver->solve(sol);

        sol.dx.stage(0).setZero();

        if (GlobalLogMask & LogLevel::Warning &&
            (sol.dx.vec.hasNaN() || sol.du.vec.hasNaN() ||
             sol.lambda.vec.hasNaN() || sol.mu.vec.hasNaN()))
        {
            logln_at(LogLevel::Warning,
                     "Warning: NaN detected in QP solver solution");
        }

        _convergence_data.iteration_data.back().qp_duration =
          std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start_iter_time)
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
        if (_params.line_search_method == LineSearchMethod::Filter)
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
                logln_at(
                  LogLevel::Error,
                  "SSQP: Line search failed (regularization not enabled)");
                throw std::runtime_error("SSQP: Line search failed");
            }

            logln_at(LogLevel::Debug,
                     "Line search failed, increasing regularization");
            qp_solver->increaseRegularization();
            --_iter;
            continue;
        }

        _guess.x = x_n;
        _guess.u = u_n;

        if (_params.line_search_method == LineSearchMethod::Filter)
        {
            _filter_memory.push_front({ _lf(_guess.x, _guess.u),
                                        _gammaf(_guess.x, _guess.u),
                                        _cf(_guess.x, _guess.u) });

            if (_filter_memory.size() > _params.filter_size)
            {
                _filter_memory.pop_back();
            }
        }

        logln_at(LogLevel::Debug, "Current x: ");
        logln_at(LogLevel::Debug, _guess.x);
        logln_at(LogLevel::Debug, "Current u: ");
        logln_at(LogLevel::Debug, _guess.u);

        _guess.lambda = sol.lambda;
        _guess.mu     = sol.mu;

        // Termination
        if ((ssqp_state =
               _check_termination<Params::calculate_stagewise_convergence>()) !=
            ONGOING)
        {
            _convergence_data.iteration_data.back().iter_duration =
              std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - start_iter_time)
                .count();
            _convergence_data.iteration_data.back().trajectory_cost =
              _lf(_guess.x, _guess.u);
            _convergence_data.iteration_data.back().trajectory_stage_gap =
              _gammaf(_guess.x, _guess.u);
            _convergence_data.iteration_data.back()
              .trajectory_constraint_violation = _cf(_guess.x, _guess.u);

            _convergence_data.total_duration =
              std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time)
                .count();

            _convergence_data.trajectory_cost =
              _convergence_data.iteration_data.back().trajectory_cost;
            _convergence_data.trajectory_stage_gap =
              _convergence_data.iteration_data.back().trajectory_stage_gap;
            _convergence_data.trajectory_constraint_violation =
              _convergence_data.iteration_data.back()
                .trajectory_constraint_violation;

            if (_iter % 40 == 0)
            {
                _printIterationHeader();
            }
            _printIterationInfo();
            logln_at(LogLevel::Info,
                     "SSQP: Total runtime: " << _convergence_data.total_duration
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
        std::visit([this](auto xml) { _setup_qp<false, xml.value>(); },
                   xml_model);
        qp_solver->reset();

        _convergence_data.iteration_data.back().iter_duration =
          std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start_iter_time)
            .count();
        _convergence_data.iteration_data.back().trajectory_cost =
          _lf(_guess.x, _guess.u);
        _convergence_data.iteration_data.back().trajectory_stage_gap =
          _gammaf(_guess.x, _guess.u);
        _convergence_data.iteration_data.back()
          .trajectory_constraint_violation = _cf(_guess.x, _guess.u);

        FrameMarkNamed("SQP Iteration");
    }

    _convergence_data.total_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_time)
        .count();

    _convergence_data.max_iter_reached     = true;
    _convergence_data.trajectory_cost      = _lf(_guess.x, _guess.u);
    _convergence_data.trajectory_stage_gap = _gammaf(_guess.x, _guess.u);
    _convergence_data.trajectory_constraint_violation = _cf(_guess.x, _guess.u);

    logln_at(LogLevel::Info,
             "SSQP: Total runtime: " << _convergence_data.total_duration
                                     << "ms");

    guess = _guess;
    return MAX_ITER_REACHED;
}

bool ssqp::write(const std::string &filepath)
{
    using namespace nlohmann;
    static const model::Dimensions &dims = _model.dims();
    static const double            &dt   = _model.dt();

    std::ofstream out;
    out.open(filepath);
    if (!out)
    {
        return false;
    }

    ordered_json j;
    j["solver"] = ordered_json({});
    _params.write(j["solver"]);

    j["convergence"]                    = ordered_json({});
    j["convergence"]["runtime_ms"]      = _convergence_data.total_duration;
    j["convergence"]["iterations"]      = _iter;
    j["convergence"]["trajectory_cost"] = _convergence_data.trajectory_cost;
    j["convergence"]["trajectory_stage_gap"] =
      _convergence_data.trajectory_stage_gap;
    j["convergence"]["trajectory_constraint_violation"] =
      _convergence_data.trajectory_constraint_violation;
    j["convergence"]["iteration_data"] = ordered_json::array();

    for (uint i = 0; i < _convergence_data.iteration_data.size(); ++i)
    {
        ordered_json iteration_data({
          {                   "qp_iterations",_convergence_data.iteration_data[i].qp_iters                                              },
          {                   "qp_runtime_us", _convergence_data.iteration_data[i].qp_duration },
          {                "iter_duration_us",
           _convergence_data.iteration_data[i].iter_duration                                  },
          {                       "step_size",   _convergence_data.iteration_data[i].step_size },
          {                         "norm_dx",     _convergence_data.iteration_data[i].norm_dx },
          {                         "norm_du",     _convergence_data.iteration_data[i].norm_du },
          {                 "trajectory_cost",
           _convergence_data.iteration_data[i].trajectory_cost                                },
          {            "trajectory_stage_gap",
           _convergence_data.iteration_data[i].trajectory_stage_gap                           },
          { "trajectory_constraint_violation",
           _convergence_data.iteration_data[i]
           .trajectory_constraint_violation                                                   },
          {           "first_violation_stage",
           _convergence_data.iteration_data[i].first_violation_stage                          },
          {                       "violation",   _convergence_data.iteration_data[i].violation }
        });

        if (_convergence_data.iteration_data[i].gap_violations.size() ==
            dims.horizon)
        {
            iteration_data["stage_costs"]          = ordered_json::array();
            iteration_data["stage_gap_violations"] = ordered_json::array();
            iteration_data["stage_constraint_violations"] =
              ordered_json::array();
            for (uint k = 0; k < dims.horizon; ++k)
            {
                iteration_data["stage_costs"].push_back(
                  _convergence_data.iteration_data[i].costs[k]);
                iteration_data["stage_gap_violations"].push_back(
                  _convergence_data.iteration_data[i].gap_violations[k]);
                iteration_data["stage_constraint_violations"].push_back(
                  _convergence_data.iteration_data[i].constraint_violations[k]);
            }

            iteration_data["stage_costs"].push_back(
              _convergence_data.iteration_data[i].costs[dims.horizon]);
            iteration_data["stage_constraint_violations"].push_back(
              _convergence_data.iteration_data[i]
                .constraint_violations[dims.horizon]);
        }

        j["convergence"]["iteration_data"].push_back(iteration_data);
    }

    j["properties"]            = ordered_json({});
    j["properties"]["horizon"] = dims.horizon;
    j["properties"]["dt"]      = dt;

    model::SolverMode solver_mode     = _model.solverMode();
    bool              loaded_from_xml = _model.xml();
    if (loaded_from_xml)
    {
        j["properties"]["solver_mode"] = model::DM_to_str.at(solver_mode);
    }
    else
    {
        j["properties"]["solver_mode"] = "Custom";
    }
    j["properties"]["xml"] = loaded_from_xml;
    bool floating_base;
    uint nq, nv, nu, njoints;
    if (loaded_from_xml || _model.hasCustomProperties())
    {
        floating_base = _model.floatingBase();
        nq            = _model.nq();
        nv            = _model.nv();
        nu            = _model.nu();
        njoints       = _model.njoints();

        j["properties"]["nq"]            = nq;
        j["properties"]["nv"]            = nv;
        j["properties"]["nu"]            = nu;
        j["properties"]["njoints"]       = njoints;
        j["properties"]["total_joints"]  = _model.total_joints();
        j["properties"]["floating_base"] = floating_base;
        j["properties"]["state_joints"]  = _model.orderedJoints();
        j["properties"]["joint_types"]   = _model.jointTypes();
    }

    j["properties"]["stages"] = ordered_json::array();
    j["stage_data"]           = ordered_json::array();

    for (uint k = 0; k < dims.horizon; ++k)
    {
        uint                     nforces = _model.nforces(k);
        std::vector<std::string> ordered_force_frames =
          _model.orderedForceFrames(k);

        j["properties"]["stages"].push_back(ordered_json({
          {                 "size",   dims.stage_state[k] },
          {              "control", dims.stage_control[k] },
          {              "nforces",               nforces },
          { "ordered_force_frames",  ordered_force_frames }
        }));

        ordered_json stage_data;

        VecX u;

        if (loaded_from_xml && floating_base)
        {
            VecX x = _guess.x.stage(k);
            u      = _guess.u.stage(k);

            stage_data["position"]            = x.head(3);
            stage_data["orientation"]         = x.segment(3, 4);
            stage_data["linear_velocity"]     = x.segment(nq, 3);
            stage_data["angular_velocity"]    = x.segment(nq + 3, 3);
            stage_data["joint_configuration"] = x.segment(7, njoints);
            stage_data["joint_velocity"]      = x.segment(nq + 6, njoints);
            if (solver_mode == model::SolverMode::ReducedInverseDynamics)
            {
                stage_data["control"] = u.head(nu);
            }
            else
            {
                stage_data["control"] = u.segment(6, njoints);
            }
        }
        else
        {
            u = _guess.u.stage(k);

            stage_data["state"]   = _guess.x.stage(k);
            stage_data["control"] = u.head(u.size() - nforces);
        }

        if (nforces > 0)
        {
            stage_data["forces"] = u.tail(3 * nforces);
        }

        j["stage_data"].push_back(stage_data);
    }

    j["properties"]["stages"].push_back(ordered_json({
      {                 "size", dims.stage_state[dims.horizon] },
      {              "control",                              0 },
      {              "nforces",                              0 },
      { "ordered_force_frames",                             {} }
    }));

    ordered_json stage_data;

    if (floating_base)
    {
        VecX x = _guess.x.stage(dims.horizon);

        stage_data["position"]            = x.head(3);
        stage_data["orientation"]         = x.segment(3, 4);
        stage_data["linear_velocity"]     = x.segment(nq, 3);
        stage_data["angular_velocity"]    = x.segment(nq + 3, 3);
        stage_data["joint_configuration"] = x.segment(7, njoints);
        stage_data["joint_velocity"]      = x.segment(nq + 6, njoints);
    }
    else
    {
        stage_data["state"] = _guess.x.stage(dims.horizon);
    }

    j["stage_data"].push_back(stage_data);

    out << std::setw(4) << j;
    out.close();

    return true;
}

const ssqp::ConvergenceData ssqp::convergenceData() const
{
    return _convergence_data;
}

void ssqp::setReducedOutput(bool reduced)
{
    _reduced_output = reduced;
}

void ssqp::_printIterationHeader() const
{
    log_at(LogLevel::Info, std::setw(18));
    log_at(LogLevel::Info, "SQP Iteration |");
    log_at(LogLevel::Info, std::setw(18));
    log_at(LogLevel::Info, "Norm/dx |");
    log_at(LogLevel::Info, std::setw(18));
    log_at(LogLevel::Info, "Norm/du |");
    log_at(LogLevel::Info, std::setw(18));
    log_at(LogLevel::Info, "Violation |");
    if (!_reduced_output)
    {
        log_at(LogLevel::Info, std::setw(18));
        log_at(LogLevel::Info, "Violation Stage |");
        log_at(LogLevel::Info, std::setw(18));
        log_at(LogLevel::Info, "QP Iterations |");
        log_at(LogLevel::Info, std::setw(18));
        log_at(LogLevel::Info, "QP Runtime (us) |");
        log_at(LogLevel::Info, std::setw(18));
        logln_at(LogLevel::Info, "Step Size");
    }
    else
    {
        log_at(LogLevel::Info, std::endl);
    }
}

void ssqp::_printIterationInfo() const
{
    const IterationConvergenceData &icd =
      _convergence_data.iteration_data.back();

    log_at(LogLevel::Info, std::setw(16));
    log_at(LogLevel::Info, _iter << " |");
    log_at(LogLevel::Info, std::setw(16) << std::setprecision(5));
    log_at(LogLevel::Info, icd.norm_dx << " |");
    log_at(LogLevel::Info, std::setw(16) << std::setprecision(5));
    log_at(LogLevel::Info, icd.norm_du << " |");
    log_at(LogLevel::Info, std::setw(16) << std::setprecision(5));
    log_at(LogLevel::Info, icd.violation << " |");
    if (!_reduced_output)
    {
        log_at(LogLevel::Info, std::setw(16));
        if (icd.first_violation_stage < 0.0)
        {
            log_at(LogLevel::Info, "-" << " |");
        }
        else
        {
            log_at(LogLevel::Info, icd.first_violation_stage << " |");
        }
        log_at(LogLevel::Info, std::setw(16));
        log_at(LogLevel::Info, icd.qp_iters << " |");
        log_at(LogLevel::Info, std::setw(16));
        log_at(LogLevel::Info, icd.qp_duration << " |");
        log_at(LogLevel::Info, std::setw(16));
        logln_at(LogLevel::Info, icd.step_size);
    }
    else
    {
        log_at(LogLevel::Info, std::endl);
    }
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

    return sum + costs[horizon]->evaluate(x.stage(horizon), _emptyVec)(0);
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

    costs[horizon]->gradient(x.stage(horizon), _emptyVec, dxk, duk);
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

    const uint nq = _model.nq();
    const uint nv = _model.nv();

    bool kinematic = _model.solverMode() == model::SolverMode::Kinematics;

    MatX dXk, dUk;
    for (uint k = 0; k < horizon; ++k)
    {
        VecX f   = dynamics[k]->evaluate(x.stage(k), u.stage(k));
        VecX r_k = dynamics[k]->state_right_minus(x.stage(k + 1), f);

        VecX a = (_params.merit_grad_softmax_beta * r_k.array().abs()).exp();
        r_k    = (r_k.array().sign() * a.array()) / a.sum();

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
        MatX J_r;
        if (kinematic)
        {
            J_r = Jr_inv_wrt_f;
        }
        else
        {
            J_r.setZero(2 * nv, 2 * nv);
            J_r.topLeftCorner(nv, nv) = Jr_inv_wrt_f;
            J_r.bottomRightCorner(nv, nv).diagonal().setConstant(-1.0);

            dx.stage(k + 1).tail(nv) += r_k.tail(nv);
        }

        dx.stage(k) += (J_r * dXk).transpose() * r_k;
        du.stage(k) += (J_r * dUk).transpose() * r_k;

        dx.stage(k + 1).head(nv) += Jr_inv_wrt_x.transpose() * r_k.head(nv);
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
        c    = constraints[horizon]->evaluate(x.stage(horizon), _emptyVec);
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
        c = constraints[horizon]->evaluate(x.stage(horizon), _emptyVec);

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

    VecX c_H = constraints[horizon]->evaluate(x.stage(horizon), _emptyVec);
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

    constraints[horizon]->gradient(x.stage(horizon), _emptyVec, dXk, dUk);

    dx.stage(horizon).noalias() += dXk.transpose() * c_H;
}
}
}