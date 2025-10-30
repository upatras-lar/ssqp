#include "model/stagewise/Factories.hpp"

namespace ssqp
{
namespace model
{
namespace stagewise
{
namespace factory
{
template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
std::shared_ptr<StageCost> make_cost(const EvaluationFunction &eval,
                                     const GradientFunction   &grad,
                                     const HessianFunction1   &hess_xx,
                                     const HessianFunction2   &hess_xu,
                                     const HessianFunction3   &hess_uu)
{
    return std::make_shared<FunctionStageCost<EvaluationFunction,
                                              GradientFunction,
                                              HessianFunction1,
                                              HessianFunction2,
                                              HessianFunction3>>(eval,
                                                                 grad,
                                                                 hess_xx,
                                                                 hess_xu,
                                                                 hess_uu);
}

template<typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
std::shared_ptr<StageDynamics> make_dynamics(
  const uint nx, const uint nu, const uint nout, const uint ndx,
  const VecX &init_state, const ManifoldOperators &ops,
  const EvaluationFunction &eval, const multiGradientFunction &grad,
  const HessprodFunction1 &hess_xx, const HessprodFunction2 &hess_xu,
  const HessprodFunction3 &hess_uu)
{
    return std::make_shared<FunctionStageDynamics<EvaluationFunction,
                                                  multiGradientFunction,
                                                  HessprodFunction1,
                                                  HessprodFunction2,
                                                  HessprodFunction3>>(
      nx,
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

template<typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
std::shared_ptr<StageConstraints> make_constraints(
  const int nout, const VecX &lb, const VecX &ub,
  const EvaluationFunction &eval, const multiGradientFunction &grad,
  const HessprodFunction1 &hess_xx, const HessprodFunction2 &hess_xu,
  const HessprodFunction3 &hess_uu)
{
    return std::make_shared<FunctionStageConstraints<EvaluationFunction,
                                                     multiGradientFunction,
                                                     HessprodFunction1,
                                                     HessprodFunction2,
                                                     HessprodFunction3>>(
      nout,
      lb,
      ub,
      eval,
      grad,
      hess_xx,
      hess_xu,
      hess_uu);
}

std::shared_ptr<StageConstraints> merge_constraints(
  const std::shared_ptr<StageConstraints> &c1,
  const std::shared_ptr<StageConstraints> &c2)
{
    uint nc = c1->output_size() + c2->output_size();

    VecX lb(nc), ub(nc);
    lb << c1->lower_bound(), c2->lower_bound();
    ub << c1->upper_bound(), c2->upper_bound();

    auto eval = [nc, c1, c2](const VecX &x, const VecX &u)
    {
        VecX c(nc);
        c.head(c1->output_size()) = c1->evaluate(x, u);
        c.tail(c2->output_size()) = c2->evaluate(x, u);

        return c;
    };

    auto grad = [nc, c1, c2](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
    {
        dX = MatX::Zero(nc, x.size());
        dU = MatX::Zero(nc, u.size());

        MatX dX1(c1->output_size(), x.size());
        MatX dU1(c1->output_size(), u.size());
        c1->gradient(x, u, dX1, dU1);
        dX.block(0, 0, c1->output_size(), x.size()) = dX1;
        dU.block(0, 0, c1->output_size(), u.size()) = dU1;

        MatX dX2(c2->output_size(), x.size());
        MatX dU2(c2->output_size(), u.size());
        c2->gradient(x, u, dX2, dU2);
        dX.block(c1->output_size(), 0, c2->output_size(), x.size()) = dX2;
        dU.block(c1->output_size(), 0, c2->output_size(), u.size()) = dU2;
    };

    auto hessprod_xx =
      [c1, c2](const VecX &x, const VecX &u, const VecX &mu, MatX &Pxx)
    {
        Pxx = MatX::Zero(x.size(), x.size());
        c1->hessprod_xx(x, u, mu.head(c1->output_size()), Pxx);

        MatX tmp = MatX::Zero(x.size(), x.size());
        c2->hessprod_xx(x, u, mu.tail(c2->output_size()), tmp);
        Pxx += tmp;
    };

    auto hessprod_xu =
      [c1, c2](const VecX &x, const VecX &u, const VecX &mu, MatX &Pxu)
    {
        Pxu = MatX::Zero(x.size(), u.size());
        c1->hessprod_xu(x, u, mu.head(c1->output_size()), Pxu);

        MatX tmp = MatX::Zero(x.size(), u.size());
        c2->hessprod_xu(x, u, mu.tail(c2->output_size()), tmp);
        Pxu += tmp;
    };

    auto hessprod_uu =
      [c1, c2](const VecX &x, const VecX &u, const VecX &mu, MatX &Puu)
    {
        Puu = MatX::Zero(u.size(), u.size());
        c1->hessprod_uu(x, u, mu.head(c1->output_size()), Puu);

        MatX tmp = MatX::Zero(u.size(), u.size());
        c2->hessprod_uu(x, u, mu.tail(c2->output_size()), tmp);
        Puu += tmp;
    };

    return make_constraints(nc,
                            lb,
                            ub,
                            eval,
                            grad,
                            hessprod_xx,
                            hessprod_xu,
                            hessprod_uu);
}

template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
FunctionStageCost<
  EvaluationFunction, GradientFunction, HessianFunction1, HessianFunction2,
  HessianFunction3>::FunctionStageCost(const EvaluationFunction &eval,
                                       const GradientFunction   &grad,
                                       const HessianFunction1   &hess_xx,
                                       const HessianFunction2   &hess_xu,
                                       const HessianFunction3   &hess_uu)
  : _eval(eval)
  , _grad(grad)
  , _hess_xx(hess_xx)
  , _hess_xu(hess_xu)
  , _hess_uu(hess_uu)
{
}

template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
VecX FunctionStageCost<EvaluationFunction, GradientFunction, HessianFunction1,
                       HessianFunction2,
                       HessianFunction3>::evaluate(const VecX &x,
                                                   const VecX &u) const
{
    return _eval(x, u);
}

template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
void FunctionStageCost<EvaluationFunction, GradientFunction, HessianFunction1,
                       HessianFunction2,
                       HessianFunction3>::gradient(const VecX &x, const VecX &u,
                                                   VecX &dx, VecX &du) const
{
    return _grad(x, u, dx, du);
}

template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
void FunctionStageCost<EvaluationFunction, GradientFunction, HessianFunction1,
                       HessianFunction2,
                       HessianFunction3>::hessian_xx(const VecX &x,
                                                     const VecX &u,
                                                     MatX       &Hxx) const
{
    return _hess_xx(x, u, Hxx);
}

template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
void FunctionStageCost<EvaluationFunction, GradientFunction, HessianFunction1,
                       HessianFunction2,
                       HessianFunction3>::hessian_xu(const VecX &x,
                                                     const VecX &u,
                                                     MatX       &Hxu) const
{
    return _hess_xu(x, u, Hxu);
}

template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
void FunctionStageCost<EvaluationFunction, GradientFunction, HessianFunction1,
                       HessianFunction2,
                       HessianFunction3>::hessian_uu(const VecX &x,
                                                     const VecX &u,
                                                     MatX       &Huu) const
{
    return _hess_uu(x, u, Huu);
}

FunctionStageDynamicsBase::FunctionStageDynamicsBase(
  const uint nx, const uint nu, const uint nout, const uint ndx,
  const VecX &init_state, const ManifoldOperators &ops)
  : _nx(nx)
  , _nu(nu)
  , _nout(nout)
  , _ndx(ndx)
  , _init_state(init_state)
  , _ops(ops)
{
}

template<typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
FunctionStageDynamics<EvaluationFunction, multiGradientFunction,
                      HessprodFunction1, HessprodFunction2, HessprodFunction3>::
  FunctionStageDynamics(const uint nx, const uint nu, const uint nout,
                        const uint ndx, const VecX &init_state,
                        const ManifoldOperators     &ops,
                        const EvaluationFunction    &eval,
                        const multiGradientFunction &grad,
                        const HessprodFunction1     &hess_xx,
                        const HessprodFunction2     &hess_xu,
                        const HessprodFunction3     &hess_uu)
  : FunctionStageDynamicsBase(nx, nu, nout, ndx, init_state, ops)
  , _eval(eval)
  , _grad(grad)
  , _hess_xx(hess_xx)
  , _hess_xu(hess_xu)
  , _hess_uu(hess_uu)
{
    assert(init_state.size() == nx &&
           "FunctionStageDynamics: nx must match init_state.size()");
}

uint FunctionStageDynamicsBase::output_size() const
{
    return _nout;
}

uint FunctionStageDynamicsBase::state_size() const
{
    return _nx;
}

uint FunctionStageDynamicsBase::control_size() const
{
    return _nu;
}

uint FunctionStageDynamicsBase::state_diff_size() const
{
    return _ndx;
}

VecX FunctionStageDynamicsBase::init_state() const
{
    return _init_state;
}

void FunctionStageDynamicsBase::set_init_state(const VecX &init_state)
{
    assert(init_state.size() == _nx &&
           "FunctionStageDynamics: nx must match init_state.size()");
    _init_state = init_state;
}

VecX FunctionStageDynamicsBase::state_right_plus(const VecX &x,
                                                 const VecX &dx) const
{
    return _ops.right_plus(x, dx);
}

VecX FunctionStageDynamicsBase::state_right_minus(const VecX &x,
                                                  const VecX &y) const
{
    return _ops.right_minus(x, y);
}

template<typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
VecX FunctionStageDynamics<EvaluationFunction, multiGradientFunction,
                           HessprodFunction1, HessprodFunction2,
                           HessprodFunction3>::evaluate(const VecX &x,
                                                        const VecX &u) const
{
    return _eval(x, u);
}

template<typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
void FunctionStageDynamics<EvaluationFunction, multiGradientFunction,
                           HessprodFunction1, HessprodFunction2,
                           HessprodFunction3>::gradient(const VecX &x,
                                                        const VecX &u, MatX &dX,
                                                        MatX &dU) const
{
    return _grad(x, u, dX, dU);
}

template<typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
void FunctionStageDynamics<EvaluationFunction, multiGradientFunction,
                           HessprodFunction1, HessprodFunction2,
                           HessprodFunction3>::hessprod_xx(const VecX &x,
                                                           const VecX &u,
                                                           const VecX &lambda,
                                                           MatX &Pxx) const
{
    return _hess_xx(x, u, lambda, Pxx);
}

template<typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
void FunctionStageDynamics<EvaluationFunction, multiGradientFunction,
                           HessprodFunction1, HessprodFunction2,
                           HessprodFunction3>::hessprod_xu(const VecX &x,
                                                           const VecX &u,
                                                           const VecX &lambda,
                                                           MatX &Pxu) const
{
    return _hess_xu(x, u, lambda, Pxu);
}

template<typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
void FunctionStageDynamics<EvaluationFunction, multiGradientFunction,
                           HessprodFunction1, HessprodFunction2,
                           HessprodFunction3>::hessprod_uu(const VecX &x,
                                                           const VecX &u,
                                                           const VecX &lambda,
                                                           MatX &Puu) const
{
    return _hess_uu(x, u, lambda, Puu);
}

FunctionStageConstraintsBase::FunctionStageConstraintsBase(const int   nout,
                                                           const VecX &lb,
                                                           const VecX &ub)
  : _nout(nout)
  , _lb(lb)
  , _ub(ub)
{
}

template<typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
FunctionStageConstraints<EvaluationFunction, multiGradientFunction,
                         HessprodFunction1, HessprodFunction2,
                         HessprodFunction3>::
  FunctionStageConstraints(const int nout, const VecX &lb, const VecX &ub,
                           const EvaluationFunction    &eval,
                           const multiGradientFunction &grad,
                           const HessprodFunction1     &hess_xx,
                           const HessprodFunction2     &hess_xu,
                           const HessprodFunction3     &hess_uu)
  : FunctionStageConstraintsBase(nout, lb, ub)
  , _eval(eval)
  , _grad(grad)
  , _hess_xx(hess_xx)
  , _hess_xu(hess_xu)
  , _hess_uu(hess_uu)
{
    assert(lb.size() == ub.size() && ub.size() == nout &&
           "FunctionStageConstraints: nout must match lb.size() and ub.size()");
}

uint FunctionStageConstraintsBase::output_size() const
{
    return _nout;
}

VecX FunctionStageConstraintsBase::lower_bound() const
{
    return _lb;
}

VecX FunctionStageConstraintsBase::upper_bound() const
{
    return _ub;
}

template<typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
VecX FunctionStageConstraints<EvaluationFunction, multiGradientFunction,
                              HessprodFunction1, HessprodFunction2,
                              HessprodFunction3>::evaluate(const VecX &x,
                                                           const VecX &u) const
{
    return _eval(x, u);
}

template<typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
void FunctionStageConstraints<EvaluationFunction, multiGradientFunction,
                              HessprodFunction1, HessprodFunction2,
                              HessprodFunction3>::gradient(const VecX &x,
                                                           const VecX &u,
                                                           MatX       &dX,
                                                           MatX       &dU) const
{
    return _grad(x, u, dX, dU);
}

template<typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
void FunctionStageConstraints<
  EvaluationFunction, multiGradientFunction, HessprodFunction1,
  HessprodFunction2, HessprodFunction3>::hessprod_xx(const VecX &x,
                                                     const VecX &u,
                                                     const VecX &lambda,
                                                     MatX       &Pxx) const
{
    return _hess_xx(x, u, lambda, Pxx);
}

template<typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
void FunctionStageConstraints<
  EvaluationFunction, multiGradientFunction, HessprodFunction1,
  HessprodFunction2, HessprodFunction3>::hessprod_xu(const VecX &x,
                                                     const VecX &u,
                                                     const VecX &lambda,
                                                     MatX       &Pxu) const
{
    return _hess_xu(x, u, lambda, Pxu);
}

template<typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
void FunctionStageConstraints<
  EvaluationFunction, multiGradientFunction, HessprodFunction1,
  HessprodFunction2, HessprodFunction3>::hessprod_uu(const VecX &x,
                                                     const VecX &u,
                                                     const VecX &lambda,
                                                     MatX       &Puu) const
{
    return _hess_uu(x, u, lambda, Puu);
}

}
}
}
}