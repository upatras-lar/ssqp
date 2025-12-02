#include "tests/include/FDGrad.hpp"

namespace ssqp
{
namespace tests
{
FDGradCost::FDGradCost(const std::shared_ptr<model::stagewise::StageCost> &cost,
                       const uint ndx, const uint nu, const uint ninputs,
                       const double h)
  : _cost(cost)
  , _ndx(ndx)
  , _nu(nu)
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
        dx[i] = (_cost->evaluate(x + _h * e, u)[0] -
                 _cost->evaluate(x - _h * e, u)[0]) /
                2 * _h;
    }

    for (uint i = 0; i < _nu; ++i)
    {
        Eigen::VectorXd e(_nu);
        e.setUnit(i);
        du[i] = (_cost->evaluate(x, u + _h * e)[0] -
                 _cost->evaluate(x, u - _h * e)[0]) /
                2 * _h;
    }
}

bool FDGradCost::run()
{
    bool pass = true;

    for (uint i = 0; i < _ninputs; ++i)
    {
        VecX x = random_vector(_ndx);
        VecX u = random_vector(_nu);

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

FDGradDynamics::FDGradDynamics(
  const std::shared_ptr<model::stagewise::StageDynamics> &dynamics,
  const uint ninputs = 100, const double h = 1e-7)
  : _dynamics(dynamics)
  , _ndx(dynamics->state_diff_size())
  , _nu(_dynamics->control_size())
  , _ninputs(ninputs)
  , _h(h)
{
}

void FDGradDynamics::_fd_grad(const VecX &x, const VecX &u, MatX &dX,
                              MatX &dU) const
{
    dX.resize(_ndx, _ndx);
    dU.resize(_ndx, _nu);

    for (uint i = 0; i < _ndx; ++i)
    {
        Eigen::VectorXd e(_ndx);
        e.setUnit(i);
        dX.col(i) = (_dynamics->evaluate(x + _h * e, u) -
                     _dynamics->evaluate(x - _h * e, u)) /
                    2 * _h;
    }

    for (uint i = 0; i < _nu; ++i)
    {
        Eigen::VectorXd e(_nu);
        e.setUnit(i);
        dU.col(i) = (_dynamics->evaluate(x, u + _h * e) -
                     _dynamics->evaluate(x, u - _h * e)) /
                    2 * _h;
    }
}

bool FDGradDynamics::run()
{
    bool pass = true;

    for (uint i = 0; i < _ninputs; ++i)
    {
        VecX x = random_vector(_ndx);
        VecX u = random_vector(_nu);

        MatX dX, dU, dXfd, dUfd;
        _dynamics->gradient(x, u, dX, dU);
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