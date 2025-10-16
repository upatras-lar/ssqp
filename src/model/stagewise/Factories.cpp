#include "model/stagewise/Factories.hpp"

namespace ssqp
{
namespace model
{
namespace stagewise
{
namespace factory
{
std::shared_ptr<StageCost> make_cost(const EvaluationFunction &eval,
                                     const GradientFunction   &grad,
                                     const HessianFunction    &hess_xx,
                                     const HessianFunction    &hess_xu,
                                     const HessianFunction    &hess_uu)
{
    return std::make_shared<FunctionStageCost>(eval,
                                               grad,
                                               hess_xx,
                                               hess_xu,
                                               hess_uu);
}
std::shared_ptr<StageDynamics> make_dynamics(
  const uint nx, const uint nu, const uint nout, const uint ndx,
  const VecX &init_state, const ManifoldOperators &ops,
  const EvaluationFunction &eval, const multiGradientFunction &grad,
  const HessprodFunction &hess_xx, const HessprodFunction &hess_xu,
  const HessprodFunction &hess_uu)
{
    return std::make_shared<FunctionStageDynamics>(nx,
                                                   nu,
                                                   nout,
                                                   ndx,
                                                   init_state,
                                                   ops,
                                                   eval,
                                                   grad,
                                                   hess_xx,
                                                   hess_xu,
                                                   hess_uu);
}

std::shared_ptr<StageConstraints> make_constraints(
  const int nout, const VecX &lb, const VecX &ub,
  const EvaluationFunction &eval, const multiGradientFunction &grad,
  const HessprodFunction &hess_xx, const HessprodFunction &hess_xu,
  const HessprodFunction &hess_uu)
{
    return std::make_shared<FunctionStageConstraints>(nout,
                                                      lb,
                                                      ub,
                                                      eval,
                                                      grad,
                                                      hess_xx,
                                                      hess_xu,
                                                      hess_uu);
}

FunctionStageCost::FunctionStageCost(const EvaluationFunction &eval,
                                     const GradientFunction   &grad,
                                     const HessianFunction    &hess_xx,
                                     const HessianFunction    &hess_xu,
                                     const HessianFunction    &hess_uu)
  : _eval(eval)
  , _grad(grad)
  , _hess_xx(hess_xx)
  , _hess_xu(hess_xu)
  , _hess_uu(hess_uu)
{
}

VecX FunctionStageCost::evaluate(const VecX &x, const VecX &u) const
{
    return _eval(x, u);
}

void FunctionStageCost::gradient(const VecX &x, const VecX &u, VecX &dx,
                                 VecX &du) const
{
    return _grad(x, u, dx, du);
}
void FunctionStageCost::hessian_xx(const VecX &x, const VecX &u,
                                   MatX &Hxx) const
{
    return _hess_xx(x, u, Hxx);
}
void FunctionStageCost::hessian_xu(const VecX &x, const VecX &u,
                                   MatX &Hxu) const
{
    return _hess_xu(x, u, Hxu);
}
void FunctionStageCost::hessian_uu(const VecX &x, const VecX &u,
                                   MatX &Huu) const
{
    return _hess_uu(x, u, Huu);
}

FunctionStageDynamics::FunctionStageDynamics(
  const uint nx, const uint nu, const uint nout, const uint ndx,
  const VecX &init_state, const ManifoldOperators &ops,
  const EvaluationFunction &eval, const multiGradientFunction &grad,
  const HessprodFunction &hess_xx, const HessprodFunction &hess_xu,
  const HessprodFunction &hess_uu)
  : _nx(nx)
  , _nu(nu)
  , _nout(nout)
  , _ndx(ndx)
  , _init_state(init_state)
  , _ops(ops)
  , _eval(eval)
  , _grad(grad)
  , _hess_xx(hess_xx)
  , _hess_xu(hess_xu)
  , _hess_uu(hess_uu)
{
}

uint FunctionStageDynamics::output_size() const
{
    return _nout;
}

uint FunctionStageDynamics::state_size() const
{
    return _nx;
}

uint FunctionStageDynamics::control_size() const
{
    return _nu;
}

uint FunctionStageDynamics::state_diff_size() const
{
    return _ndx;
}

VecX FunctionStageDynamics::init_state() const
{
    return _init_state;
}

VecX FunctionStageDynamics::state_right_plus(const VecX &x,
                                             const VecX &dx) const
{
    return _ops.right_plus(x, dx);
}

VecX FunctionStageDynamics::state_right_minus(const VecX &x,
                                              const VecX &y) const
{
    return _ops.right_minus(x, y);
}

VecX FunctionStageDynamics::evaluate(const VecX &x, const VecX &u) const
{
    return _eval(x, u);
}

void FunctionStageDynamics::gradient(const VecX &x, const VecX &u, MatX &dX,
                                     MatX &dU) const
{
    return _grad(x, u, dX, dU);
}

void FunctionStageDynamics::hessprod_xx(const VecX &x, const VecX &u,
                                        const VecX &lambda, MatX &Pxx) const
{
    return _hess_xx(x, u, lambda, Pxx);
}

void FunctionStageDynamics::hessprod_xu(const VecX &x, const VecX &u,
                                        const VecX &lambda, MatX &Pxu) const
{
    return _hess_xu(x, u, lambda, Pxu);
}

void FunctionStageDynamics::hessprod_uu(const VecX &x, const VecX &u,
                                        const VecX &lambda, MatX &Puu) const
{
    return _hess_uu(x, u, lambda, Puu);
}

FunctionStageConstraints::FunctionStageConstraints(
  const int nout, const VecX &lb, const VecX &ub,
  const EvaluationFunction &eval, const multiGradientFunction &grad,
  const HessprodFunction &hess_xx, const HessprodFunction &hess_xu,
  const HessprodFunction &hess_uu)
  : _nout(nout)
  , _lb(lb)
  , _ub(ub)
  , _eval(eval)
  , _grad(grad)
  , _hess_xx(hess_xx)
  , _hess_xu(hess_xu)
  , _hess_uu(hess_uu)
{
}

uint FunctionStageConstraints::output_size() const
{
    return _nout;
}

VecX FunctionStageConstraints::lower_bound() const
{
    return _lb;
}

VecX FunctionStageConstraints::upper_bound() const
{
    return _ub;
}

VecX FunctionStageConstraints::evaluate(const VecX &x, const VecX &u) const
{
    return _eval(x, u);
}

void FunctionStageConstraints::gradient(const VecX &x, const VecX &u, MatX &dX,
                                        MatX &dU) const
{
    return _grad(x, u, dX, dU);
}

void FunctionStageConstraints::hessprod_xx(const VecX &x, const VecX &u,
                                           const VecX &lambda, MatX &Pxx) const
{
    return _hess_xx(x, u, lambda, Pxx);
}

void FunctionStageConstraints::hessprod_xu(const VecX &x, const VecX &u,
                                           const VecX &lambda, MatX &Pxu) const
{
    return _hess_xu(x, u, lambda, Pxu);
}

void FunctionStageConstraints::hessprod_uu(const VecX &x, const VecX &u,
                                           const VecX &lambda, MatX &Puu) const
{
    return _hess_uu(x, u, lambda, Puu);
}

}
}
}
}