#include "tests/include/FDGrad.hpp"

namespace ssqp
{
namespace tests
{
FDGradCost::FDGradCost(const std::string                  &name,
                       const std::shared_ptr<model::Model> model,
                       const uint stage, const uint ninputs, const double h)
  : Test(name)
  , _model(model)
  , _cost(model->costs()[stage])
  , _stage(stage)
  , _ndx(model->dynamics()[stage]->state_diff_size())
  , _nu(model->dynamics()[stage]->control_size())
  , _ninputs(ninputs)
  , _h(h)
{
}

void FDGradCost::_fd_grad(const VecX &x, const VecX &u, VecX &dx,
                          VecX &du) const
{
    dx.resize(_ndx);
    du.resize(_nu);

    for (uint i = 0; i < _ndx; ++i)
    {
        VecX e(_ndx);
        e.setUnit(i);

        dx[i] = (_cost->evaluate(
                   _model->dynamics()[_stage]->state_right_plus(x, _h * e),
                   u)[0] -
                 _cost->evaluate(
                   _model->dynamics()[_stage]->state_right_plus(x, -_h * e),
                   u)[0]) /
                (2 * _h);
    }

    for (uint i = 0; i < _nu; ++i)
    {
        VecX e(_nu);
        e.setUnit(i);
        du[i] = (_cost->evaluate(x, u + _h * e)[0] -
                 _cost->evaluate(x, u - _h * e)[0]) /
                (2 * _h);
    }
}

bool FDGradCost::run()
{
    bool pass = true;

    for (uint i = 0; i < _ninputs; ++i)
    {
        VecX x = _model->randomState(_stage);
        VecX u = _model->randomControls(_stage);

        VecX dx, du, dxfd, dufd;
        _cost->gradient(x, u, dx, du);
        _fd_grad(x, u, dxfd, dufd);

        double dxdiff = (dx - dxfd).lpNorm<Eigen::Infinity>();
        double dudiff = (du - dufd).lpNorm<Eigen::Infinity>();

        if (std::isnan(dxdiff) || std::isnan(dudiff) || dxdiff > _h ||
            dudiff > _h)
        {
            double max_diff = 0.0;
            int    idx_dx   = -1;
            for (uint j = 0; j < dx.size(); ++j)
            {
                double diff = std::abs(dx[j] - dxfd[j]);
                if (diff > max_diff)
                {
                    max_diff = diff;
                    idx_dx   = j;
                }
            }

            max_diff   = 0.0;
            int idx_du = -1;
            for (uint j = 0; j < du.size(); ++j)
            {
                double diff = std::abs(du[j] - dufd[j]);
                if (diff > max_diff)
                {
                    max_diff = diff;
                    idx_du   = j;
                }
            }

            std::stringstream ss;
            ss << "Run " << i << " failed" << std::endl
               << "Run input:" << std::endl
               << "Norm dX: " << (dx - dxfd).lpNorm<Eigen::Infinity>()
               << " at index: " << idx_dx << std::endl
               << "Norm dU: " << (du - dufd).lpNorm<Eigen::Infinity>()
               << " at index: " << idx_du << std::endl
               << "x: " << x.transpose() << std::endl
               << "u: " << u.transpose() << std::endl
               << "Gradient dx: " << dx.transpose() << std::endl
               << "FD gradient dx: " << dxfd.transpose() << std::endl
               << "Gradient du: " << du.transpose() << std::endl
               << "FD gradient du: " << dufd.transpose() << std::endl;

            _failed_logs.push_back(ss.str());
            pass = false;
        }
    }

    return pass;
}

FDGradDynamics::FDGradDynamics(const std::string                   &name,
                               const std::shared_ptr<model::Model> &model,
                               const uint stage, const uint ninputs,
                               const double h)
  : Test(name)
  , _model(model)
  , _stage(stage)
  , _dynamics(model->dynamics()[stage])
  , _next_dynamics(model->dynamics()[stage + 1])
  , _ninputs(ninputs)
  , _h(h)
{
}

void FDGradDynamics::_fd_grad(const VecX &x, const VecX &u, MatX &dX,
                              MatX &dU) const
{
    uint ndx = _dynamics->state_diff_size();
    uint nu  = _dynamics->control_size();

    dX.resize(ndx, ndx);
    dU.resize(ndx, nu);

    for (uint i = 0; i < ndx; ++i)
    {
        VecX e(ndx);
        e.setUnit(i);
        dX.col(i) =
          _next_dynamics->state_right_minus(
            _dynamics->evaluate(_dynamics->state_right_plus(x, _h * e), u),
            _dynamics->evaluate(_dynamics->state_right_plus(x, -_h * e), u)) /
          (2 * _h);
    }

    for (uint i = 0; i < nu; ++i)
    {
        VecX e(nu);
        e.setUnit(i);
        dU.col(i) = _next_dynamics->state_right_minus(
                      _dynamics->evaluate(x, u + _h * e),
                      _dynamics->evaluate(x, u - _h * e)) /
                    (2 * _h);
    }
}

bool FDGradDynamics::run()
{
    bool pass = true;

    for (uint i = 0; i < _ninputs; ++i)
    {
        VecX x = _model->randomState(_stage);
        VecX u = _model->randomControls(_stage);

        MatX dX, dU, dXfd, dUfd;
        _dynamics->gradient(x, u, dX, dU);
        _fd_grad(x, u, dXfd, dUfd);

        double dXdiff = (dX - dXfd).lpNorm<Eigen::Infinity>();
        double dUdiff = (dU - dUfd).lpNorm<Eigen::Infinity>();

        if (std::isnan(dXdiff) || std::isnan(dUdiff) || dXdiff > _h ||
            dUdiff > _h)
        {
            double max_diff   = 0.0;
            int    idx_dx_row = -1, idx_dx_col = -1;

            for (uint r = 0; r < dX.rows(); ++r)
            {
                for (uint c = 0; c < dX.cols(); ++c)
                {
                    double diff = std::abs(dX(r, c) - dXfd(r, c));
                    if (diff > max_diff)
                    {
                        max_diff   = diff;
                        idx_dx_row = r;
                        idx_dx_col = c;
                    }
                }
            }

            max_diff       = 0.0;
            int idx_du_row = -1, idx_du_col = -1;

            for (uint r = 0; r < dU.rows(); ++r)
            {
                for (uint c = 0; c < dU.cols(); ++c)
                {
                    double diff = std::abs(dU(r, c) - dUfd(r, c));
                    if (diff > max_diff)
                    {
                        max_diff   = diff;
                        idx_du_row = r;
                        idx_du_col = c;
                    }
                }
            }

            std::stringstream ss;
            ss << "Run " << i << " failed" << std::endl
               << "Run input:" << std::endl
               << "Norm dX: " << (dX - dXfd).lpNorm<Eigen::Infinity>()
               << " at index: " << idx_dx_row << ", " << idx_dx_col << std::endl
               << "Norm dU: " << (dU - dUfd).lpNorm<Eigen::Infinity>()
               << " at index: " << idx_du_row << ", " << idx_du_col << std::endl
               << std::endl
               << "x: " << x.transpose() << std::endl
               << "u: " << u.transpose() << std::endl
               << "Gradient dX: " << std::endl
               << dX << std::endl
               << "FD gradient dX: " << std::endl
               << dXfd << std::endl
               << "Gradient dU: " << std::endl
               << dU << std::endl
               << "FD gradient dU: " << std::endl
               << dUfd << std::endl;

            _failed_logs.push_back(ss.str());
            pass = false;
        }
    }

    return pass;
}

FDGradConstraints::FDGradConstraints(const std::string                   &name,
                                     const std::shared_ptr<model::Model> &model,
                                     const uint stage, const uint ninputs,
                                     const double h)
  : Test(name)
  , _model(model)
  , _stage(stage)
  , _constraints(_model->constraints()[stage])
  , _dynamics(_model->dynamics()[stage])
  , _ndc(_constraints->constraint_diff_size())
  , _ndx(_dynamics->state_diff_size())
  , _nu(_dynamics->control_size())
  , _ninputs(ninputs)
  , _h(h)
{
}

void FDGradConstraints::_fd_grad(const VecX &x, const VecX &u, MatX &dX,
                                 MatX &dU) const
{
    dX.resize(_ndc, _ndx);
    dU.resize(_ndc, _nu);

    for (uint i = 0; i < _ndx; ++i)
    {
        VecX e(_ndx);
        e.setUnit(i);
        dX.col(i) =
          _constraints->right_minus(
            _constraints->evaluate(_dynamics->state_right_plus(x, _h * e), u),
            _constraints->evaluate(_dynamics->state_right_plus(x, -_h * e),
                                   u)) /
          (2 * _h);
    }

    for (uint i = 0; i < _nu; ++i)
    {
        VecX e(_nu);
        e.setUnit(i);
        dU.col(i) =
          _constraints->right_minus(_constraints->evaluate(x, u + _h * e),
                                    _constraints->evaluate(x, u - _h * e)) /
          (2 * _h);
    }
}

bool FDGradConstraints::run()
{
    bool pass = true;

    for (uint i = 0; i < _ninputs; ++i)
    {
        VecX x = _model->randomState(_stage);
        VecX u = _model->randomControls(_stage);

        MatX dX, dU, dXfd, dUfd;
        _constraints->gradient(x, u, dX, dU);
        _fd_grad(x, u, dXfd, dUfd);

        double dXdiff = (dX - dXfd).lpNorm<Eigen::Infinity>();
        double dUdiff = (dU - dUfd).lpNorm<Eigen::Infinity>();

        if (std::isnan(dXdiff) || std::isnan(dUdiff) || dXdiff > _h ||
            dUdiff > _h)
        {
            double max_diff   = 0.0;
            int    idx_dx_row = -1, idx_dx_col = -1;

            for (uint r = 0; r < dX.rows(); ++r)
            {
                for (uint c = 0; c < dX.cols(); ++c)
                {
                    double diff = std::abs(dX(r, c) - dXfd(r, c));
                    if (diff > max_diff)
                    {
                        max_diff   = diff;
                        idx_dx_row = r;
                        idx_dx_col = c;
                    }
                }
            }

            max_diff       = 0.0;
            int idx_du_row = -1, idx_du_col = -1;

            for (uint r = 0; r < dU.rows(); ++r)
            {
                for (uint c = 0; c < dU.cols(); ++c)
                {
                    double diff = std::abs(dU(r, c) - dUfd(r, c));
                    if (diff > max_diff)
                    {
                        max_diff   = diff;
                        idx_du_row = r;
                        idx_du_col = c;
                    }
                }
            }

            std::stringstream ss;
            ss << "Run " << i << " failed" << std::endl
               << "Run input:" << std::endl
               << "Norm dX: " << (dX - dXfd).lpNorm<Eigen::Infinity>()
               << " at index: " << idx_dx_row << ", " << idx_dx_col << std::endl
               << std::endl
               << "Norm dU: " << (dU - dUfd).lpNorm<Eigen::Infinity>()
               << " at index: " << idx_du_row << ", " << idx_du_col << std::endl
               << std::endl
               << "x: " << x.transpose() << std::endl
               << "u: " << u.transpose() << std::endl
               << "Gradient dX: " << std::endl
               << dX << std::endl
               << "FD gradient dX: " << std::endl
               << dXfd << std::endl
               << "Gradient dU: " << std::endl
               << dU << std::endl
               << "FD gradient dU: " << std::endl
               << dUfd << std::endl;

            _failed_logs.push_back(ss.str());
            pass = false;
        }
    }

    return pass;
}

FDGradFullTrajectoryCost::FDGradFullTrajectoryCost(
  const std::string &name, const std::shared_ptr<model::Model> &model,
  const uint ninputs, const double h)
  : Test(name)
  , _ssqp(*model.get())
  , _model(model)
  , _ninputs(ninputs)
  , _h(h)
{
}

void FDGradFullTrajectoryCost::_fd_grad(const solvers::FlatVec &x,
                                        const solvers::FlatVec &u,
                                        solvers::FlatVec       &dx,
                                        solvers::FlatVec       &du)
{
    const model::DynamicsContainer dynamics = _model->dynamics();

    for (uint k = 0; k < x.stages(); ++k)
    {
        VecX x_k = x.stage(k);

        const uint ndx = dx.stage(k).size();

        for (uint i = 0; i < ndx; ++i)
        {
            VecX e(ndx);
            e.setUnit(i);

            solvers::FlatVec xp(x), xm(x);
            xp.stage(k) = dynamics[k]->state_right_plus(x_k, _h * e);
            xm.stage(k) = dynamics[k]->state_right_plus(x_k, -_h * e);

            dx.stage(k)[i] = (_ssqp._lf(xp, u) - _ssqp._lf(xm, u)) / (2 * _h);
        }
    }

    for (uint k = 0; k < u.stages(); ++k)
    {
        VecX u_k = u.stage(k);

        const uint nu = du.stage(k).size();

        for (uint i = 0; i < nu; ++i)
        {
            VecX e(nu);
            e.setUnit(i);

            solvers::FlatVec up(u), um(u);
            up.stage(k) = u_k + _h * e;
            um.stage(k) = u_k - _h * e;

            du.stage(k)[i] = (_ssqp._lf(x, up) - _ssqp._lf(x, um)) / (2 * _h);
        }
    }
}

bool FDGradFullTrajectoryCost::run()
{
    bool pass = true;

    const model::Dimensions dims     = _model->dims();
    const model::Dimensions lin_dims = _model->linear_dims();

    solvers::FlatVec x(dims.state, dims.stage_state);
    solvers::FlatVec u(dims.control, dims.stage_control);

    solvers::FlatVec dx(lin_dims.state, lin_dims.stage_state);
    solvers::FlatVec du(lin_dims.control, lin_dims.stage_control);

    solvers::FlatVec dxfd(dx, false);
    solvers::FlatVec dufd(du, false);

    x.setPrintColumns(-1);
    u.setPrintColumns(-1);
    dx.setPrintColumns(-1);
    du.setPrintColumns(-1);
    dxfd.setPrintColumns(-1);
    dufd.setPrintColumns(-1);

    for (uint i = 0; i < _ninputs; ++i)
    {
        for (uint k = 0; k < x.stages(); ++k)
        {
            x.stage(k) = _model->randomState(k);
        }

        for (uint k = 0; k < u.stages(); ++k)
        {
            u.stage(k) = _model->randomControls(k);
        }

        _ssqp._lf_grad(x, u, dx, du);
        _fd_grad(x, u, dxfd, dufd);

        double dxdiff = (dx - dxfd).vec.lpNorm<Eigen::Infinity>();
        double dudiff = (du - dufd).vec.lpNorm<Eigen::Infinity>();

        if (std::isnan(dxdiff) || std::isnan(dudiff) || dxdiff > _h ||
            dudiff > _h)
        {
            std::stringstream ss;
            ss << "Run " << i << " failed" << std::endl
               << "Run input:" << std::endl
               << "Norm dX: " << (dx - dxfd).vec.lpNorm<Eigen::Infinity>()
               << std::endl
               << "Norm dU: " << (du - dufd).vec.lpNorm<Eigen::Infinity>()
               << std::endl
               << "x:" << std::endl
               << x << std::endl
               << "u:" << std::endl
               << u << std::endl
               << "Gradient dx:" << std::endl
               << dx << std::endl
               << "FD gradient dx:" << std::endl
               << dxfd << std::endl
               << "Gradient du:" << std::endl
               << du << std::endl
               << "FD gradient du:" << std::endl
               << dufd << std::endl;

            _failed_logs.push_back(ss.str());
            pass = false;
        }
    }

    return pass;
}

FDGradFullTrajectoryStageGap::FDGradFullTrajectoryStageGap(
  const std::string &name, const std::shared_ptr<model::Model> &model,
  const uint ninputs, const double h)
  : Test(name)
  , _ssqp(*model.get())
  , _model(model)
  , _ninputs(ninputs)
  , _h(h)
{
}

void FDGradFullTrajectoryStageGap::_fd_grad(const solvers::FlatVec &x,
                                            const solvers::FlatVec &u,
                                            solvers::FlatVec       &dx,
                                            solvers::FlatVec       &du)
{
    const model::DynamicsContainer dynamics = _model->dynamics();

    for (uint k = 0; k < x.stages(); ++k)
    {
        VecX x_k = x.stage(k);

        const uint ndx = dx.stage(k).size();

        for (uint i = 0; i < ndx; ++i)
        {
            VecX e(ndx);
            e.setUnit(i);

            solvers::FlatVec xp(x), xm(x);
            xp.stage(k) = dynamics[k]->state_right_plus(x_k, _h * e);
            xm.stage(k) = dynamics[k]->state_right_plus(x_k, -_h * e);

            dx.stage(k)[i] =
              (_ssqp._gammaf_softmax(xp, u) - _ssqp._gammaf_softmax(xm, u)) /
              (2 * _h);
        }
    }

    for (uint k = 0; k < u.stages(); ++k)
    {
        VecX u_k = u.stage(k);

        const uint nu = du.stage(k).size();

        for (uint i = 0; i < nu; ++i)
        {
            VecX e(nu);
            e.setUnit(i);

            solvers::FlatVec up(u), um(u);
            up.stage(k) = u_k + _h * e;
            um.stage(k) = u_k - _h * e;

            du.stage(k)[i] =
              (_ssqp._gammaf_softmax(x, up) - _ssqp._gammaf_softmax(x, um)) /
              (2 * _h);
        }
    }
}

bool FDGradFullTrajectoryStageGap::run()
{
    bool pass = true;

    const model::Dimensions dims     = _model->dims();
    const model::Dimensions lin_dims = _model->linear_dims();

    solvers::FlatVec x(dims.state, dims.stage_state);
    solvers::FlatVec u(dims.control, dims.stage_control);

    solvers::FlatVec dx(lin_dims.state, lin_dims.stage_state);
    solvers::FlatVec du(lin_dims.control, lin_dims.stage_control);

    solvers::FlatVec dxfd(dx, false);
    solvers::FlatVec dufd(du, false);

    x.setPrintColumns(-1);
    u.setPrintColumns(-1);
    dx.setPrintColumns(-1);
    du.setPrintColumns(-1);
    dxfd.setPrintColumns(-1);
    dufd.setPrintColumns(-1);

    for (uint i = 0; i < _ninputs; ++i)
    {
        for (uint k = 0; k < x.stages(); ++k)
        {
            x.stage(k) = _model->randomState(k);
        }

        for (uint k = 0; k < u.stages(); ++k)
        {
            u.stage(k) = _model->randomControls(k);
        }

        dx.vec.setZero();
        du.vec.setZero();

        dx.vec.setZero();
        du.vec.setZero();

        _ssqp._gammaf_grad(x, u, dx, du);
        _fd_grad(x, u, dxfd, dufd);

        double dxdiff = (dx - dxfd).vec.lpNorm<Eigen::Infinity>();
        double dudiff = (du - dufd).vec.lpNorm<Eigen::Infinity>();

        if (std::isnan(dxdiff) || std::isnan(dudiff) || dxdiff > _h ||
            dudiff > _h)
        {
            std::stringstream ss;
            ss << "Run " << i << " failed" << std::endl
               << "Run input:" << std::endl
               << "Norm dX: " << (dx - dxfd).vec.lpNorm<Eigen::Infinity>()
               << std::endl
               << "Norm dU: " << (du - dufd).vec.lpNorm<Eigen::Infinity>()
               << std::endl
               << "x:" << std::endl
               << x << std::endl
               << "u:" << std::endl
               << u << std::endl
               << "Gradient dx:" << std::endl
               << dx << std::endl
               << "FD gradient dx:" << std::endl
               << dxfd << std::endl
               << "Gradient du:" << std::endl
               << du << std::endl
               << "FD gradient du:" << std::endl
               << dufd << std::endl;

            _failed_logs.push_back(ss.str());
            pass = false;
        }
    }

    return pass;
}

FDGradFullTrajectoryConstraintViolation::
  FDGradFullTrajectoryConstraintViolation(
    const std::string &name, const std::shared_ptr<model::Model> &model,
    const uint ninputs, const double h)
  : Test(name)
  , _ssqp(*model.get())
  , _model(model)
  , _ninputs(ninputs)
  , _h(h)
{
}

void FDGradFullTrajectoryConstraintViolation::_fd_grad(
  const solvers::FlatVec &x, const solvers::FlatVec &u, solvers::FlatVec &dx,
  solvers::FlatVec &du)
{
    const model::DynamicsContainer dynamics = _model->dynamics();

    for (uint k = 0; k < x.stages(); ++k)
    {
        VecX x_k = x.stage(k);

        const uint ndx = dx.stage(k).size();

        for (uint i = 0; i < ndx; ++i)
        {
            VecX e(ndx);
            e.setUnit(i);

            solvers::FlatVec xp(x), xm(x);
            xp.stage(k) = dynamics[k]->state_right_plus(x_k, _h * e);
            xm.stage(k) = dynamics[k]->state_right_plus(x_k, -_h * e);

            dx.stage(k)[i] =
              (_ssqp._cf_softmax(xp, u) - _ssqp._cf_softmax(xm, u)) / (2 * _h);
        }
    }

    for (uint k = 0; k < u.stages(); ++k)
    {
        VecX u_k = u.stage(k);

        const uint nu = du.stage(k).size();

        for (uint i = 0; i < nu; ++i)
        {
            VecX e(nu);
            e.setUnit(i);

            solvers::FlatVec up(u), um(u);
            up.stage(k) = u_k + _h * e;
            um.stage(k) = u_k - _h * e;

            du.stage(k)[i] =
              (_ssqp._cf_softmax(x, up) - _ssqp._cf_softmax(x, um)) / (2 * _h);
        }
    }
}

bool FDGradFullTrajectoryConstraintViolation::run()
{
    bool pass = true;

    const model::Dimensions dims     = _model->dims();
    const model::Dimensions lin_dims = _model->linear_dims();

    solvers::FlatVec x(dims.state, dims.stage_state);
    solvers::FlatVec u(dims.control, dims.stage_control);

    solvers::FlatVec dx(lin_dims.state, lin_dims.stage_state);
    solvers::FlatVec du(lin_dims.control, lin_dims.stage_control);

    solvers::FlatVec dxfd(dx, false);
    solvers::FlatVec dufd(du, false);

    x.setPrintColumns(-1);
    u.setPrintColumns(-1);
    dx.setPrintColumns(-1);
    du.setPrintColumns(-1);
    dxfd.setPrintColumns(-1);
    dufd.setPrintColumns(-1);

    for (uint i = 0; i < _ninputs; ++i)
    {
        for (uint k = 0; k < x.stages(); ++k)
        {
            x.stage(k) = _model->randomState(k);
        }

        for (uint k = 0; k < u.stages(); ++k)
        {
            u.stage(k) = _model->randomControls(k);
        }

        dx.vec.setZero();
        du.vec.setZero();

        _ssqp._cf_grad(x, u, dx, du);
        _fd_grad(x, u, dxfd, dufd);

        double dxdiff = (dx - dxfd).vec.lpNorm<Eigen::Infinity>();
        double dudiff = (du - dufd).vec.lpNorm<Eigen::Infinity>();

        if (std::isnan(dxdiff) || std::isnan(dudiff) || dxdiff > _h ||
            dudiff > _h)
        {
            std::stringstream ss;
            ss << "Run " << i << " failed" << std::endl
               << "Run input:" << std::endl
               << "Norm dX: " << (dx - dxfd).vec.lpNorm<Eigen::Infinity>()
               << std::endl
               << "Norm dU: " << (du - dufd).vec.lpNorm<Eigen::Infinity>()
               << std::endl
               << "x:" << std::endl
               << x << std::endl
               << "u:" << std::endl
               << u << std::endl
               << "Gradient dx:" << std::endl
               << dx << std::endl
               << "FD gradient dx:" << std::endl
               << dxfd << std::endl
               << "Gradient du:" << std::endl
               << du << std::endl
               << "FD gradient du:" << std::endl
               << dufd << std::endl;

            _failed_logs.push_back(ss.str());
            pass = false;
        }
    }

    return pass;
}

}
}