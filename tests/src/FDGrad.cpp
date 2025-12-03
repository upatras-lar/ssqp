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
        Eigen::VectorXd e(_ndx);
        e.setUnit(i);
        dx[i] = (_cost->evaluate(
                   _model->dynamics()[_stage]->state_right_plus(x, _h * e),
                   u)[0] -
                 _cost->evaluate(
                   _model->dynamics()[_stage]->state_right_plus(x, _h * e),
                   u)[0]) /
                (2 * _h);
    }

    for (uint i = 0; i < _nu; ++i)
    {
        Eigen::VectorXd e(_nu);
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

        if ((dx - dxfd).lpNorm<Eigen::Infinity>() > _h ||
            (du - dufd).lpNorm<Eigen::Infinity>() > _h)
        {
            std::stringstream ss;
            ss << "Test " << i << " failed" << std::endl
               << "Test input:" << std::endl
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
    static uint ndx = _dynamics->state_diff_size();
    static uint nu  = _dynamics->control_size();

    dX.resize(ndx, ndx);
    dU.resize(ndx, nu);

    for (uint i = 0; i < ndx; ++i)
    {
        Eigen::VectorXd e(ndx);
        e.setUnit(i);
        dX.col(i) =
          _next_dynamics->state_right_minus(
            _dynamics->evaluate(_dynamics->state_right_plus(x, _h * e), u),
            _dynamics->evaluate(_dynamics->state_right_plus(x, -_h * e), u)) /
          (2 * _h);
    }

    for (uint i = 0; i < nu; ++i)
    {
        Eigen::VectorXd e(nu);
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

        if ((dX - dXfd).lpNorm<Eigen::Infinity>() > _h ||
            (dU - dUfd).lpNorm<Eigen::Infinity>() > _h)
        {
            std::stringstream ss;
            ss << "Run " << i << " failed" << std::endl
               << "Run input:" << std::endl
               << "Norm dX: " << (dX - dXfd).lpNorm<Eigen::Infinity>()
               << std::endl
               << "Norm dU: " << (dU - dUfd).lpNorm<Eigen::Infinity>()
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
        Eigen::VectorXd e(_ndx);
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
        Eigen::VectorXd e(_nu);
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

        if ((dX - dXfd).lpNorm<Eigen::Infinity>() > _h ||
            (dU - dUfd).lpNorm<Eigen::Infinity>() > _h)
        {
            std::stringstream ss;
            ss << "Test " << i << " failed" << std::endl
               << "Test input:" << std::endl
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
}
}