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
std::shared_ptr<StageCost> make_cost(EvaluationFunction eval,
                                     GradientFunction   grad,
                                     HessianFunction1   hess_xx,
                                     HessianFunction2   hess_xu,
                                     HessianFunction3   hess_uu)
{
    return std::make_shared<FunctionStageCost<EvaluationFunction,
                                              GradientFunction,
                                              HessianFunction1,
                                              HessianFunction2,
                                              HessianFunction3>>(
      std::move(eval),
      std::move(grad),
      std::move(hess_xx),
      std::move(hess_xu),
      std::move(hess_uu));
}

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
std::shared_ptr<StageDynamics> make_dynamics(
  const uint nx, const uint nu, const uint nout, const uint ndx,
  const VecX                                             &init_state,
  ManifoldOperators<ManifoldOperator1, ManifoldOperator2> ops,
  EvaluationFunction eval, multiGradientFunction grad,
  HessprodFunction1 hess_xx, HessprodFunction2 hess_xu,
  HessprodFunction3 hess_uu)
{
    return std::make_shared<FunctionStageDynamics<ManifoldOperator1,
                                                  ManifoldOperator2,
                                                  EvaluationFunction,
                                                  multiGradientFunction,
                                                  HessprodFunction1,
                                                  HessprodFunction2,
                                                  HessprodFunction3>>(
      nx,
      nu,
      nout,
      ndx,
      init_state,
      std::move(ops),
      std::move(eval),
      std::move(grad),
      std::move(hess_xx),
      std::move(hess_xu),
      std::move(hess_uu));
}

template<typename EvaluationFunction, typename multiGradientFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
std::shared_ptr<StageConstraints> make_constraints(
  const uint nout, const uint ndc, const VecX &lb, const VecX &ub,
  EvaluationFunction eval, multiGradientFunction grad, ManifoldOperator op,
  HessprodFunction1 hess_xx, HessprodFunction2 hess_xu,
  HessprodFunction3 hess_uu)
{
    return std::make_shared<FunctionStageConstraints<EvaluationFunction,
                                                     multiGradientFunction,
                                                     ManifoldOperator,
                                                     HessprodFunction1,
                                                     HessprodFunction2,
                                                     HessprodFunction3>>(
      nout,
      ndc,
      lb,
      ub,
      std::move(eval),
      std::move(grad),
      std::move(op),
      std::move(hess_xx),
      std::move(hess_xu),
      std::move(hess_uu));
}

// TODO: instead of merging, possibly hold multiple?
std::shared_ptr<StageCost> merge_costs(const std::shared_ptr<StageCost> &l1,
                                       const std::shared_ptr<StageCost> &l2)
{
    auto eval = [l1, l2](const VecX &x, const VecX &u)
    {
        return (Scalar() << l1->evaluate(x, u) + l2->evaluate(x, u)).finished();
    };

    auto grad = [l1, l2](const VecX &x, const VecX &u, VecX &dx, VecX &du)
    {
        VecX tmp_x, tmp_u;
        l1->gradient(x, u, dx, du);
        l2->gradient(x, u, tmp_x, tmp_u);

        dx += tmp_x;
        du += tmp_u;
    };

    auto hess_xx = [l1, l2](const VecX &x, const VecX &u, MatX &Hxx)
    {
        MatX tmp_Hxx;
        l1->hessian_xx(x, u, Hxx);
        l2->hessian_xx(x, u, tmp_Hxx);

        Hxx += tmp_Hxx;
    };

    auto hess_xu = [l1, l2](const VecX &x, const VecX &u, MatX &Hxu)
    {
        MatX tmp_Hxu;
        l1->hessian_xu(x, u, Hxu);
        l2->hessian_xu(x, u, tmp_Hxu);

        Hxu += tmp_Hxu;
    };

    auto hess_uu = [l1, l2](const VecX &x, const VecX &u, MatX &Huu)
    {
        MatX tmp_Huu;
        l1->hessian_uu(x, u, Huu);
        l2->hessian_uu(x, u, tmp_Huu);

        Huu += tmp_Huu;
    };

    return make_cost(eval, grad, hess_xx, hess_xu, hess_uu);
}

std::shared_ptr<StageConstraints> merge_constraints(
  const std::shared_ptr<StageConstraints> &c1,
  const std::shared_ptr<StageConstraints> &c2)
{
    uint nc  = c1->output_size() + c2->output_size();
    uint ndc = c1->constraint_diff_size() + c2->constraint_diff_size();

    VecX lb(nc), ub(nc);
    lb << c1->lower_bound(), c2->lower_bound();
    ub << c1->upper_bound(), c2->upper_bound();

    auto op = [nc, ndc, c1, c2](const VecX &x, const VecX &y)
    {
        VecX c(ndc);
        c.head(c1->constraint_diff_size()) =
          c1->right_minus(x.head(c1->output_size()), y.head(c1->output_size()));
        c.tail(c2->constraint_diff_size()) =
          c2->right_minus(x.tail(c2->output_size()), y.tail(c2->output_size()));

        return c;
    };

    auto eval = [nc, c1, c2](const VecX &x, const VecX &u)
    {
        VecX c(nc);
        c.head(c1->output_size()) = c1->evaluate(x, u);
        c.tail(c2->output_size()) = c2->evaluate(x, u);

        return c;
    };

    auto grad = [ndc, c1, c2](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
    {
        MatX dX1, dU1;
        c1->gradient(x, u, dX1, dU1);

        uint ndx = dX1.cols();
        dX       = MatX::Zero(ndc, ndx);
        dU       = MatX::Zero(ndc, u.size());

        dX.block(0, 0, c1->constraint_diff_size(), ndx)      = dX1;
        dU.block(0, 0, c1->constraint_diff_size(), u.size()) = dU1;

        MatX dX2, dU2;
        c2->gradient(x, u, dX2, dU2);
        dX.block(c1->constraint_diff_size(),
                 0,
                 c2->constraint_diff_size(),
                 ndx)      = dX2;
        dU.block(c1->constraint_diff_size(),
                 0,
                 c2->constraint_diff_size(),
                 u.size()) = dU2;
    };

    auto hessprod_xx =
      [c1, c2](const VecX &x, const VecX &u, const VecX &mu, MatX &Pxx)
    {
        MatX tmp;
        c1->hessprod_xx(x, u, mu.head(c1->constraint_diff_size()), Pxx);
        c2->hessprod_xx(x, u, mu.tail(c2->constraint_diff_size()), tmp);
        Pxx += tmp;
    };

    auto hessprod_xu =
      [c1, c2](const VecX &x, const VecX &u, const VecX &mu, MatX &Pxu)
    {
        MatX tmp;
        c1->hessprod_xu(x, u, mu.head(c1->constraint_diff_size()), Pxu);
        c2->hessprod_xu(x, u, mu.tail(c2->constraint_diff_size()), tmp);

        Pxu += tmp;
    };

    auto hessprod_uu =
      [c1, c2](const VecX &x, const VecX &u, const VecX &mu, MatX &Puu)
    {
        MatX tmp;
        c1->hessprod_uu(x, u, mu.head(c1->constraint_diff_size()), Puu);
        c2->hessprod_uu(x, u, mu.tail(c2->constraint_diff_size()), tmp);

        Puu += tmp;
    };

    return make_constraints(nc,
                            ndc,
                            lb,
                            ub,
                            eval,
                            grad,
                            op,
                            hessprod_xx,
                            hessprod_xu,
                            hessprod_uu);
}

template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
FunctionStageCost<EvaluationFunction, GradientFunction, HessianFunction1,
                  HessianFunction2,
                  HessianFunction3>::FunctionStageCost(EvaluationFunction eval,
                                                       GradientFunction   grad,
                                                       HessianFunction1 hess_xx,
                                                       HessianFunction2 hess_xu,
                                                       HessianFunction3 hess_uu)
  : _eval(std::move(eval))
  , _grad(std::move(grad))
  , _hess_xx(std::move(hess_xx))
  , _hess_xu(std::move(hess_xu))
  , _hess_uu(std::move(hess_uu))
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

FunctionStageDynamicsBase::FunctionStageDynamicsBase(const uint  nx,
                                                     const uint  nu,
                                                     const uint  nout,
                                                     const uint  ndx,
                                                     const VecX &init_state)
  : _nx(nx)
  , _nu(nu)
  , _nout(nout)
  , _ndx(ndx)
  , _init_state(init_state)
{
}

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
FunctionStageDynamics<ManifoldOperator1, ManifoldOperator2, EvaluationFunction,
                      multiGradientFunction, HessprodFunction1,
                      HessprodFunction2, HessprodFunction3>::
  FunctionStageDynamics(
    const uint nx, const uint nu, const uint nout, const uint ndx,
    const VecX                                             &init_state,
    ManifoldOperators<ManifoldOperator1, ManifoldOperator2> ops,
    EvaluationFunction eval, multiGradientFunction grad,
    HessprodFunction1 hess_xx, HessprodFunction2 hess_xu,
    HessprodFunction3 hess_uu)
  : FunctionStageDynamicsBase(nx, nu, nout, ndx, init_state)
  , _ops(std::move(ops))
  , _eval(std::move(eval))
  , _grad(std::move(grad))
  , _hess_xx(std::move(hess_xx))
  , _hess_xu(std::move(hess_xu))
  , _hess_uu(std::move(hess_uu))
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

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
VecX FunctionStageDynamics<
  ManifoldOperator1, ManifoldOperator2, EvaluationFunction,
  multiGradientFunction, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::state_right_plus(const VecX &x, const VecX &dx) const
{
    return _ops.right_plus(x, dx);
}

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
VecX FunctionStageDynamics<
  ManifoldOperator1, ManifoldOperator2, EvaluationFunction,
  multiGradientFunction, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::state_right_minus(const VecX &x, const VecX &y) const
{
    return _ops.right_minus(x, y);
}

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
VecX FunctionStageDynamics<ManifoldOperator1, ManifoldOperator2,
                           EvaluationFunction, multiGradientFunction,
                           HessprodFunction1, HessprodFunction2,
                           HessprodFunction3>::evaluate(const VecX &x,
                                                        const VecX &u) const
{
    return _eval(x, u);
}

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
void FunctionStageDynamics<ManifoldOperator1, ManifoldOperator2,
                           EvaluationFunction, multiGradientFunction,
                           HessprodFunction1, HessprodFunction2,
                           HessprodFunction3>::gradient(const VecX &x,
                                                        const VecX &u, MatX &dX,
                                                        MatX &dU) const
{
    return _grad(x, u, dX, dU);
}

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
void FunctionStageDynamics<ManifoldOperator1, ManifoldOperator2,
                           EvaluationFunction, multiGradientFunction,
                           HessprodFunction1, HessprodFunction2,
                           HessprodFunction3>::hessprod_xx(const VecX &x,
                                                           const VecX &u,
                                                           const VecX &lambda,
                                                           MatX &Pxx) const
{
    return _hess_xx(x, u, lambda, Pxx);
}

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
void FunctionStageDynamics<ManifoldOperator1, ManifoldOperator2,
                           EvaluationFunction, multiGradientFunction,
                           HessprodFunction1, HessprodFunction2,
                           HessprodFunction3>::hessprod_xu(const VecX &x,
                                                           const VecX &u,
                                                           const VecX &lambda,
                                                           MatX &Pxu) const
{
    return _hess_xu(x, u, lambda, Pxu);
}

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
void FunctionStageDynamics<ManifoldOperator1, ManifoldOperator2,
                           EvaluationFunction, multiGradientFunction,
                           HessprodFunction1, HessprodFunction2,
                           HessprodFunction3>::hessprod_uu(const VecX &x,
                                                           const VecX &u,
                                                           const VecX &lambda,
                                                           MatX &Puu) const
{
    return _hess_uu(x, u, lambda, Puu);
}

FunctionStageConstraintsBase::FunctionStageConstraintsBase(const uint  nout,
                                                           const uint  ndc,
                                                           const VecX &lb,
                                                           const VecX &ub)
  : _nout(nout)
  , _ndc(ndc)
  , _lb(lb)
  , _ub(ub)
{
}

template<typename EvaluationFunction, typename multiGradientFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
FunctionStageConstraints<
  EvaluationFunction, multiGradientFunction, ManifoldOperator,
  HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::FunctionStageConstraints(const uint nout, const uint ndc,
                                               const VecX &lb, const VecX &ub,
                                               const EvaluationFunction    eval,
                                               const multiGradientFunction grad,
                                               const ManifoldOperator      op,
                                               const HessprodFunction1 hess_xx,
                                               const HessprodFunction2 hess_xu,
                                               const HessprodFunction3 hess_uu)
  : FunctionStageConstraintsBase(nout, ndc, lb, ub)
  , _eval(std::move(eval))
  , _grad(std::move(grad))
  , _op(std::move(op))
  , _hess_xx(std::move(hess_xx))
  , _hess_xu(std::move(hess_xu))
  , _hess_uu(std::move(hess_uu))
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
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
uint
FunctionStageConstraints<EvaluationFunction, multiGradientFunction,
                         ManifoldOperator, HessprodFunction1, HessprodFunction2,
                         HessprodFunction3>::constraint_diff_size() const
{
    return _ndc;
}

template<typename EvaluationFunction, typename multiGradientFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
VecX
FunctionStageConstraints<EvaluationFunction, multiGradientFunction,
                         ManifoldOperator, HessprodFunction1, HessprodFunction2,
                         HessprodFunction3>::right_minus(const VecX &x,
                                                         const VecX &y) const
{
    return _op(x, y);
}

template<typename EvaluationFunction, typename multiGradientFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
VecX
FunctionStageConstraints<EvaluationFunction, multiGradientFunction,
                         ManifoldOperator, HessprodFunction1, HessprodFunction2,
                         HessprodFunction3>::evaluate(const VecX &x,
                                                      const VecX &u) const
{
    return _eval(x, u);
}

template<typename EvaluationFunction, typename multiGradientFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
void
FunctionStageConstraints<EvaluationFunction, multiGradientFunction,
                         ManifoldOperator, HessprodFunction1, HessprodFunction2,
                         HessprodFunction3>::gradient(const VecX &x,
                                                      const VecX &u, MatX &dX,
                                                      MatX &dU) const
{
    return _grad(x, u, dX, dU);
}

template<typename EvaluationFunction, typename multiGradientFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
void
FunctionStageConstraints<EvaluationFunction, multiGradientFunction,
                         ManifoldOperator, HessprodFunction1, HessprodFunction2,
                         HessprodFunction3>::hessprod_xx(const VecX &x,
                                                         const VecX &u,
                                                         const VecX &lambda,
                                                         MatX       &Pxx) const
{
    return _hess_xx(x, u, lambda, Pxx);
}

template<typename EvaluationFunction, typename multiGradientFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
void
FunctionStageConstraints<EvaluationFunction, multiGradientFunction,
                         ManifoldOperator, HessprodFunction1, HessprodFunction2,
                         HessprodFunction3>::hessprod_xu(const VecX &x,
                                                         const VecX &u,
                                                         const VecX &lambda,
                                                         MatX       &Pxu) const
{
    return _hess_xu(x, u, lambda, Pxu);
}

template<typename EvaluationFunction, typename multiGradientFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
void
FunctionStageConstraints<EvaluationFunction, multiGradientFunction,
                         ManifoldOperator, HessprodFunction1, HessprodFunction2,
                         HessprodFunction3>::hessprod_uu(const VecX &x,
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