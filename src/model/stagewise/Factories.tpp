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
  const VecX &init_state, const VecX &init_control,
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
      init_control,
      std::move(ops),
      std::move(eval),
      std::move(grad),
      std::move(hess_xx),
      std::move(hess_xu),
      std::move(hess_uu));
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
std::shared_ptr<StageConstraints> make_constraints(
  NOutFunction nout, NdcFunction ndc, const LowerBoundFunction lb,
  const UpperBoundFunction ub, EvaluationFunction eval,
  multiGradientFunction grad, ManifoldOperator op, HessprodFunction1 hess_xx,
  HessprodFunction2 hess_xu, HessprodFunction3 hess_uu)
{
    return std::make_shared<FunctionStageConstraints<NOutFunction,
                                                     NdcFunction,
                                                     LowerBoundFunction,
                                                     UpperBoundFunction,
                                                     EvaluationFunction,
                                                     multiGradientFunction,
                                                     ManifoldOperator,
                                                     HessprodFunction1,
                                                     HessprodFunction2,
                                                     HessprodFunction3>>(
      std::move(nout),
      std::move(ndc),
      std::move(lb),
      std::move(ub),
      std::move(eval),
      std::move(grad),
      std::move(op),
      std::move(hess_xx),
      std::move(hess_xu),
      std::move(hess_uu));
}

std::shared_ptr<StageCost> merge_costs(const std::shared_ptr<StageCost> &l1,
                                       const std::shared_ptr<StageCost> &l2)
{
    auto eval = [l1, l2](const VecX &x, const VecX &u)
    {
        Scalar out;

        if (l1->enabled())
        {
            out += l1->evaluate(x, u);

            if (l2->enabled())
            {
                out += l2->evaluate(x, u);
            }
        }
        else
        {
            out += l2->evaluate(x, u);
        }

        return (Scalar() << l1->evaluate(x, u) + l2->evaluate(x, u)).finished();
    };

    auto grad = [l1, l2](const VecX &x, const VecX &u, VecX &dx, VecX &du)
    {
        if (l1->enabled())
        {
            l1->gradient(x, u, dx, du);

            if (l2->enabled())
            {
                VecX tmp_x, tmp_u;
                l2->gradient(x, u, tmp_x, tmp_u);
                dx += tmp_x;
                du += tmp_u;
            }
        }
        else
        {
            l2->gradient(x, u, dx, du);
        }
    };

    auto hess_xx = [l1, l2](const VecX &x, const VecX &u, MatX &Hxx)
    {
        if (l1->enabled())
        {
            l1->hessian_xx(x, u, Hxx);

            if (l2->enabled())
            {
                MatX tmp_Hxx;
                l2->hessian_xx(x, u, tmp_Hxx);
                Hxx += tmp_Hxx;
            }
        }
        else
        {
            l2->hessian_xx(x, u, Hxx);
        }
    };

    auto hess_xu = [l1, l2](const VecX &x, const VecX &u, MatX &Hxu)
    {
        if (l1->enabled())
        {
            l1->hessian_xu(x, u, Hxu);

            if (l2->enabled())
            {
                MatX tmp_Hxu;
                l2->hessian_xu(x, u, tmp_Hxu);
                Hxu += tmp_Hxu;
            }
        }
        else
        {
            l2->hessian_xu(x, u, Hxu);
        }
    };

    auto hess_uu = [l1, l2](const VecX &x, const VecX &u, MatX &Huu)
    {
        if (l1->enabled())
        {
            l1->hessian_uu(x, u, Huu);

            if (l2->enabled())
            {
                MatX tmp_Huu;
                l2->hessian_uu(x, u, tmp_Huu);
                Huu += tmp_Huu;
            }
        }
        else
        {
            l2->hessian_uu(x, u, Huu);
        }
    };

    return make_cost(eval, grad, hess_xx, hess_xu, hess_uu);
}

std::shared_ptr<StageConstraints> merge_constraints(
  const std::shared_ptr<StageConstraints> &c1,
  const std::shared_ptr<StageConstraints> &c2)
{
    auto nout = [c1, c2]()
    {
        uint nc = 0;

        if (c1->enabled())
        {
            nc += c1->output_size();
        }

        if (c2->enabled())
        {
            nc += c2->output_size();
        }

        return nc;
    };

    auto ndc = [c1, c2]()
    {
        uint ndc = 0;

        if (c1->enabled())
        {
            ndc += c1->constraint_diff_size();
        }

        if (c2->enabled())
        {
            ndc += c2->constraint_diff_size();
        }

        return ndc;
    };

    auto lb = [c1, c2]()
    {
        uint nc = 0;

        if (c1->enabled())
        {
            nc += c1->output_size();
        }

        if (c2->enabled())
        {
            nc += c2->output_size();
        }

        VecX lb(nc);

        if (c1->enabled())
        {
            lb.head(c1->output_size()) = c1->lower_bound();
        }

        if (c2->enabled())
        {
            lb.tail(c2->output_size()) = c2->lower_bound();
        }

        return lb;
    };

    auto ub = [c1, c2]()
    {
        uint nc = 0;

        if (c1->enabled())
        {
            nc += c1->output_size();
        }

        if (c2->enabled())
        {
            nc += c2->output_size();
        }

        VecX ub(nc);

        if (c1->enabled())
        {
            ub.head(c1->output_size()) = c1->upper_bound();
        }

        if (c2->enabled())
        {
            ub.tail(c2->output_size()) = c2->upper_bound();
        }

        return ub;
    };

    auto op = [c1, c2](const VecX &x, const VecX &y)
    {
        uint ndc = 0;

        if (c1->enabled())
        {
            ndc += c1->constraint_diff_size();
        }

        if (c2->enabled())
        {
            ndc += c2->constraint_diff_size();
        }

        VecX c(ndc);

        if (c1->enabled())
        {
            c.head(c1->constraint_diff_size()) =
              c1->right_minus(x.head(c1->output_size()),
                              y.head(c1->output_size()));
        }

        if (c2->enabled())
        {
            c.tail(c2->constraint_diff_size()) =
              c2->right_minus(x.tail(c2->output_size()),
                              y.tail(c2->output_size()));
        }

        return c;
    };

    auto eval = [c1, c2](const VecX &x, const VecX &u)
    {
        uint nc = 0;

        if (c1->enabled())
        {
            nc += c1->output_size();
        }

        if (c2->enabled())
        {
            nc += c2->output_size();
        }

        VecX c(nc);

        if (c1->enabled())
        {
            c.head(c1->output_size()) = c1->evaluate(x, u);
        }

        if (c2->enabled())
        {
            c.tail(c2->output_size()) = c2->evaluate(x, u);
        }

        return c;
    };

    auto grad = [c1, c2](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
    {
        uint ndc = 0;

        if (c1->enabled())
        {
            ndc += c1->constraint_diff_size();
        }

        if (c2->enabled())
        {
            ndc += c2->constraint_diff_size();
        }

        if (c1->enabled())
        {
            MatX tmp_dX, tmp_dU;
            c1->gradient(x, u, tmp_dX, tmp_dU);

            uint ndx = tmp_dX.cols();
            dX       = MatX::Zero(ndc, ndx);
            dU       = MatX::Zero(ndc, u.size());

            dX.block(0, 0, c1->constraint_diff_size(), ndx)      = tmp_dX;
            dU.block(0, 0, c1->constraint_diff_size(), u.size()) = tmp_dU;

            if (c2->enabled())
            {
                c2->gradient(x, u, tmp_dX, tmp_dU);
                dX.block(c1->constraint_diff_size(),
                         0,
                         c2->constraint_diff_size(),
                         ndx)      = tmp_dX;
                dU.block(c1->constraint_diff_size(),
                         0,
                         c2->constraint_diff_size(),
                         u.size()) = tmp_dU;
            }
        }
        else
        {
            c2->gradient(x, u, dX, dU);
        }
    };

    auto hessprod_xx =
      [c1, c2](const VecX &x, const VecX &u, const VecX &mu, MatX &Pxx)
    {
        if (c1->enabled())
        {
            c1->hessprod_xx(x, u, mu.head(c1->constraint_diff_size()), Pxx);
            if (c2->enabled())
            {
                MatX tmp_Pxx;
                c2->hessprod_xx(x,
                                u,
                                mu.tail(c2->constraint_diff_size()),
                                tmp_Pxx);
                Pxx += tmp_Pxx;
            }
        }
        else
        {
            c2->hessprod_xx(x, u, mu, Pxx);
        }
    };

    auto hessprod_xu =
      [c1, c2](const VecX &x, const VecX &u, const VecX &mu, MatX &Pxu)
    {
        if (c1->enabled())
        {
            c1->hessprod_xu(x, u, mu.head(c1->constraint_diff_size()), Pxu);
            if (c2->enabled())
            {
                MatX tmp_Pxu;
                c2->hessprod_xu(x,
                                u,
                                mu.tail(c2->constraint_diff_size()),
                                tmp_Pxu);
                Pxu += tmp_Pxu;
            }
        }
        else
        {
            c2->hessprod_xu(x, u, mu, Pxu);
        }
    };

    auto hessprod_uu =
      [c1, c2](const VecX &x, const VecX &u, const VecX &mu, MatX &Puu)
    {
        if (c1->enabled())
        {
            c1->hessprod_uu(x, u, mu.head(c1->constraint_diff_size()), Puu);
            if (c2->enabled())
            {
                MatX tmp_Puu;
                c2->hessprod_uu(x,
                                u,
                                mu.tail(c2->constraint_diff_size()),
                                tmp_Puu);
                Puu += tmp_Puu;
            }
        }
        else
        {
            c2->hessprod_uu(x, u, mu, Puu);
        }
    };

    return make_constraints(nout,
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
    _grad(x, u, dx, du);
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
    _hess_xx(x, u, Hxx);
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
    _hess_xu(x, u, Hxu);
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
    _hess_uu(x, u, Huu);
}

FunctionStageDynamicsBase::FunctionStageDynamicsBase(
  const uint nx, const uint nu, const uint nout, const uint ndx,
  const VecX &init_state, const VecX &init_control)
  : _nx(nx)
  , _nu(nu)
  , _nout(nout)
  , _ndx(ndx)
  , _init_state(init_state)
  , _init_control(init_control)
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
    const VecX &init_state, const VecX &init_control,
    ManifoldOperators<ManifoldOperator1, ManifoldOperator2> ops,
    EvaluationFunction eval, multiGradientFunction grad,
    HessprodFunction1 hess_xx, HessprodFunction2 hess_xu,
    HessprodFunction3 hess_uu)
  : FunctionStageDynamicsBase(nx, nu, nout, ndx, init_state, init_control)
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

VecX FunctionStageDynamicsBase::init_control() const
{
    return _init_control;
}

void FunctionStageDynamicsBase::set_init_control(const VecX &init_control)
{
    assert(init_control.size() == _nu &&
           "FunctionStageDynamics: nu must match init_control.size()");
    _init_control = init_control;
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
    _grad(x, u, dX, dU);
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
    _hess_xx(x, u, lambda, Pxx);
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
    _hess_xu(x, u, lambda, Pxu);
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
    _hess_uu(x, u, lambda, Puu);
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
FunctionStageConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, ManifoldOperator,
  HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::FunctionStageConstraints(const NOutFunction          nout,
                                               const NdcFunction           ndc,
                                               const LowerBoundFunction    lb,
                                               const UpperBoundFunction    ub,
                                               const EvaluationFunction    eval,
                                               const multiGradientFunction grad,
                                               const ManifoldOperator      op,
                                               const HessprodFunction1 hess_xx,
                                               const HessprodFunction2 hess_xu,
                                               const HessprodFunction3 hess_uu)
  : _nout(std::move(nout))
  , _ndc(std::move(ndc))
  , _lb(std::move(lb))
  , _ub(std::move(ub))
  , _eval(std::move(eval))
  , _grad(std::move(grad))
  , _op(std::move(op))
  , _hess_xx(std::move(hess_xx))
  , _hess_xu(std::move(hess_xu))
  , _hess_uu(std::move(hess_uu))
{
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
uint FunctionStageConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, ManifoldOperator,
  HessprodFunction1, HessprodFunction2, HessprodFunction3>::output_size() const
{
    return _nout();
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
uint FunctionStageConstraints<NOutFunction, NdcFunction, LowerBoundFunction,
                              UpperBoundFunction, EvaluationFunction,
                              multiGradientFunction, ManifoldOperator,
                              HessprodFunction1, HessprodFunction2,
                              HessprodFunction3>::constraint_diff_size() const
{
    return _ndc();
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
VecX FunctionStageConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, ManifoldOperator,
  HessprodFunction1, HessprodFunction2, HessprodFunction3>::lower_bound() const
{
    return _lb();
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
VecX FunctionStageConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, ManifoldOperator,
  HessprodFunction1, HessprodFunction2, HessprodFunction3>::upper_bound() const
{
    return _ub();
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
VecX FunctionStageConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, ManifoldOperator,
  HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::right_minus(const VecX &x, const VecX &y) const
{
    return _op(x, y);
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
VecX FunctionStageConstraints<NOutFunction, NdcFunction, LowerBoundFunction,
                              UpperBoundFunction, EvaluationFunction,
                              multiGradientFunction, ManifoldOperator,
                              HessprodFunction1, HessprodFunction2,
                              HessprodFunction3>::evaluate(const VecX &x,
                                                           const VecX &u) const
{
    return _eval(x, u);
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
void FunctionStageConstraints<NOutFunction, NdcFunction, LowerBoundFunction,
                              UpperBoundFunction, EvaluationFunction,
                              multiGradientFunction, ManifoldOperator,
                              HessprodFunction1, HessprodFunction2,
                              HessprodFunction3>::gradient(const VecX &x,
                                                           const VecX &u,
                                                           MatX       &dX,
                                                           MatX       &dU) const
{
    _grad(x, u, dX, dU);
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
void FunctionStageConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, ManifoldOperator,
  HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::hessprod_xx(const VecX &x, const VecX &u,
                                  const VecX &lambda, MatX &Pxx) const
{
    _hess_xx(x, u, lambda, Pxx);
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
void FunctionStageConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, ManifoldOperator,
  HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::hessprod_xu(const VecX &x, const VecX &u,
                                  const VecX &lambda, MatX &Pxu) const
{
    _hess_xu(x, u, lambda, Pxu);
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
void FunctionStageConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, ManifoldOperator,
  HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::hessprod_uu(const VecX &x, const VecX &u,
                                  const VecX &lambda, MatX &Puu) const
{
    _hess_uu(x, u, lambda, Puu);
}

}
}
}
}