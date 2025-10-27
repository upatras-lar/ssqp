#ifndef SSQP_FACTORIES_HPP
#define SSQP_FACTORIES_HPP

#include <memory>

#include "types.hpp"
#include "model/stagewise/StageCost.hpp"
#include "model/stagewise/StageDynamics.hpp"
#include "model/stagewise/StageConstraints.hpp"

namespace ssqp
{
namespace model
{
namespace stagewise
{
namespace factory
{
std::shared_ptr<StageCost> make_cost(
  const EvaluationFunction &eval, const GradientFunction &grad,
  const HessianFunction &hess_xx = ZeroHessian_xx,
  const HessianFunction &hess_xu = ZeroHessian_xu,
  const HessianFunction &hess_uu = ZeroHessian_uu);

std::shared_ptr<StageDynamics> make_dynamics(
  const uint nx, const uint nu, const uint nout, const uint ndx,
  const VecX &init_state, const ManifoldOperators &ops,
  const EvaluationFunction &eval, const multiGradientFunction &grad,
  const HessprodFunction &hess_xx = ZeroHessprod_xx,
  const HessprodFunction &hess_xu = ZeroHessprod_xu,
  const HessprodFunction &hess_uu = ZeroHessprod_uu);

std::shared_ptr<StageConstraints> make_constraints(
  const int nout, const VecX &lb, const VecX &ub,
  const EvaluationFunction &eval, const multiGradientFunction &grad,
  const HessprodFunction &hess_xx = ZeroHessprod_xx,
  const HessprodFunction &hess_xu = ZeroHessprod_xu,
  const HessprodFunction &hess_uu = ZeroHessprod_uu);

std::shared_ptr<StageConstraints> merge_constraints(
  const std::shared_ptr<StageConstraints> &c1,
  const std::shared_ptr<StageConstraints> &c2);

class FunctionStageCost : public StageCost
{
  public:
    FunctionStageCost(const EvaluationFunction &eval,
                      const GradientFunction   &grad,
                      const HessianFunction    &hess_xx,
                      const HessianFunction    &hess_xu,
                      const HessianFunction    &hess_uu);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const EvaluationFunction _eval;
    const GradientFunction   _grad;
    const HessianFunction    _hess_xx, _hess_xu, _hess_uu;
};
class FunctionStageDynamics : public StageDynamics
{
  public:
    FunctionStageDynamics(const uint nx, const uint nu, const uint nout,
                          const uint ndx, const VecX &init_state,
                          const ManifoldOperators     &ops,
                          const EvaluationFunction    &eval,
                          const multiGradientFunction &grad,
                          const HessprodFunction      &hess_xx,
                          const HessprodFunction      &hess_xu,
                          const HessprodFunction      &hess_uu);

    uint output_size() const override;

    uint state_size() const override;

    uint control_size() const override;

    uint state_diff_size() const override;

    VecX init_state() const override;

    void set_init_state(const VecX &init_state) override;

    VecX state_right_plus(const VecX &x, const VecX &dx) const override;

    VecX state_right_minus(const VecX &x, const VecX &y) const override;

    VecX evaluate(const VecX &x, const VecX &u) const override;

    void gradient(const VecX &x, const VecX &u, MatX &dX,
                  MatX &dU) const override;

    void hessprod_xx(const VecX &x, const VecX &u, const VecX &lambda,
                     MatX &Pxx) const override;

    void hessprod_xu(const VecX &x, const VecX &u, const VecX &lambda,
                     MatX &Pxu) const override;

    void hessprod_uu(const VecX &x, const VecX &u, const VecX &lambda,
                     MatX &Puu) const override;

  protected:
    const uint                  _nx, _nu, _nout, _ndx;
    VecX                        _init_state;
    const ManifoldOperators     _ops;
    const EvaluationFunction    _eval;
    const multiGradientFunction _grad;
    const HessprodFunction      _hess_xx, _hess_xu, _hess_uu;
};
class FunctionStageConstraints : public StageConstraints
{
  public:
    FunctionStageConstraints(const int nout, const VecX &lb, const VecX &ub,
                             const EvaluationFunction    &eval,
                             const multiGradientFunction &grad,
                             const HessprodFunction      &hess_xx,
                             const HessprodFunction      &hess_xu,
                             const HessprodFunction      &hess_uu);

    uint output_size() const override;

    VecX lower_bound() const override;
    VecX upper_bound() const override;

    VecX evaluate(const VecX &x, const VecX &u) const override;

    void gradient(const VecX &x, const VecX &u, MatX &dX,
                  MatX &dU) const override;

    void hessprod_xx(const VecX &x, const VecX &u, const VecX &lambda,
                     MatX &Pxx) const override;

    void hessprod_xu(const VecX &x, const VecX &u, const VecX &lambda,
                     MatX &Pxu) const override;

    void hessprod_uu(const VecX &x, const VecX &u, const VecX &lambda,
                     MatX &Puu) const override;

  protected:
    const uint                  _nout;
    const VecX                  _lb, _ub;
    const EvaluationFunction    _eval;
    const multiGradientFunction _grad;
    const HessprodFunction      _hess_xx, _hess_xu, _hess_uu;
};
}
}
}
}

#endif