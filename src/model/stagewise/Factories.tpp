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
std::shared_ptr<StageCost> make_cost(const std::vector<std::string> names,
                                     EvaluationFunction             eval,
                                     GradientFunction               grad,
                                     HessianFunction1               hess_xx,
                                     HessianFunction2               hess_xu,
                                     HessianFunction3               hess_uu)
{
    return std::make_shared<FunctionStageCost<EvaluationFunction,
                                              GradientFunction,
                                              HessianFunction1,
                                              HessianFunction2,
                                              HessianFunction3>>(
      names,
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
  const std::vector<std::string> names, const uint nx, const uint nu,
  const uint nout, const uint ndx, const VecX &init_state,
  const VecX                                             &init_control,
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
      names,
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
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
std::shared_ptr<StageConstraints> make_constraints(
  const std::vector<std::string> names, NOutFunction nout, NdcFunction ndc,
  const LowerBoundFunction lb, const UpperBoundFunction ub,
  EvaluationFunction eval, multiGradientFunction grad,
  StateOnlyRowsFunction state_only_rows, BoxRowsFunction box_rows,
  ManifoldOperator op, HessprodFunction1 hess_xx, HessprodFunction2 hess_xu,
  HessprodFunction3 hess_uu)
{
    return std::make_shared<FunctionStageConstraints<NOutFunction,
                                                     NdcFunction,
                                                     LowerBoundFunction,
                                                     UpperBoundFunction,
                                                     EvaluationFunction,
                                                     multiGradientFunction,
                                                     StateOnlyRowsFunction,
                                                     BoxRowsFunction,
                                                     ManifoldOperator,
                                                     HessprodFunction1,
                                                     HessprodFunction2,
                                                     HessprodFunction3>>(
      names,
      std::move(nout),
      std::move(ndc),
      std::move(lb),
      std::move(ub),
      std::move(eval),
      std::move(grad),
      std::move(state_only_rows),
      std::move(box_rows),
      std::move(op),
      std::move(hess_xx),
      std::move(hess_xu),
      std::move(hess_uu));
}

template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
std::shared_ptr<StageCost> make_add_cost(
  const std::vector<std::string> names, const uint ndx, EvaluationFunction eval,
  GradientFunction grad, HessianFunction1 hess_xx, HessianFunction2 hess_xu,
  HessianFunction3 hess_uu)
{
    return std::make_shared<FunctionStageAddCost<EvaluationFunction,
                                                 GradientFunction,
                                                 HessianFunction1,
                                                 HessianFunction2,
                                                 HessianFunction3>>(
      names,
      ndx,
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
std::shared_ptr<StageDynamics> make_add_dynamics(
  const std::vector<std::string> names, const uint nx, const uint nu,
  const uint nout, const uint ndx, const VecX &init_state,
  const VecX                                             &init_control,
  ManifoldOperators<ManifoldOperator1, ManifoldOperator2> ops,
  EvaluationFunction eval, multiGradientFunction grad,
  HessprodFunction1 hess_xx, HessprodFunction2 hess_xu,
  HessprodFunction3 hess_uu)
{
    return std::make_shared<FunctionStageAddDynamics<ManifoldOperator1,
                                                     ManifoldOperator2,
                                                     EvaluationFunction,
                                                     multiGradientFunction,
                                                     HessprodFunction1,
                                                     HessprodFunction2,
                                                     HessprodFunction3>>(
      names,
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
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
std::shared_ptr<StageConstraints> make_add_constraints(
  const std::vector<std::string> names, const uint ndx, NOutFunction nout,
  NdcFunction ndc, const LowerBoundFunction lb, const UpperBoundFunction ub,
  EvaluationFunction eval, multiGradientFunction grad,
  StateOnlyRowsFunction state_only_rows, BoxRowsFunction box_rows,
  ManifoldOperator op, HessprodFunction1 hess_xx, HessprodFunction2 hess_xu,
  HessprodFunction3 hess_uu)
{
    return std::make_shared<FunctionStageAddConstraints<NOutFunction,
                                                        NdcFunction,
                                                        LowerBoundFunction,
                                                        UpperBoundFunction,
                                                        EvaluationFunction,
                                                        multiGradientFunction,
                                                        StateOnlyRowsFunction,
                                                        BoxRowsFunction,
                                                        ManifoldOperator,
                                                        HessprodFunction1,
                                                        HessprodFunction2,
                                                        HessprodFunction3>>(
      names,
      ndx,
      std::move(nout),
      std::move(ndc),
      std::move(lb),
      std::move(ub),
      std::move(eval),
      std::move(grad),
      std::move(state_only_rows),
      std::move(box_rows),
      std::move(op),
      std::move(hess_xx),
      std::move(hess_xu),
      std::move(hess_uu));
}

template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
FunctionStageCost<
  EvaluationFunction, GradientFunction, HessianFunction1, HessianFunction2,
  HessianFunction3>::FunctionStageCost(const std::vector<std::string> names,
                                       EvaluationFunction             eval,
                                       GradientFunction               grad,
                                       HessianFunction1               hess_xx,
                                       HessianFunction2               hess_xu,
                                       HessianFunction3               hess_uu)
  : _eval(std::move(eval))
  , _grad(std::move(grad))
  , _hess_xx(std::move(hess_xx))
  , _hess_xu(std::move(hess_xu))
  , _hess_uu(std::move(hess_uu))
{
    _names = names;
}

template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
void
FunctionStageCost<EvaluationFunction, GradientFunction, HessianFunction1,
                  HessianFunction2, HessianFunction3>::evaluate(ConstVecXRef x,
                                                                ConstVecXRef u,
                                                                VecX &out) const
{
    _eval(x, u, out);
}

template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
void
FunctionStageCost<EvaluationFunction, GradientFunction, HessianFunction1,
                  HessianFunction2, HessianFunction3>::gradient(ConstVecXRef x,
                                                                ConstVecXRef u,
                                                                VecX        &dx,
                                                                VecX &du) const
{
    _grad(x, u, dx, du);
}

template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
void FunctionStageCost<EvaluationFunction, GradientFunction, HessianFunction1,
                       HessianFunction2,
                       HessianFunction3>::hessian_xx(ConstVecXRef x,
                                                     ConstVecXRef u,
                                                     MatX        &Hxx) const
{
    _hess_xx(x, u, Hxx);
}

template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
void FunctionStageCost<EvaluationFunction, GradientFunction, HessianFunction1,
                       HessianFunction2,
                       HessianFunction3>::hessian_xu(ConstVecXRef x,
                                                     ConstVecXRef u,
                                                     MatX        &Hxu) const
{
    _hess_xu(x, u, Hxu);
}

template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
void FunctionStageCost<EvaluationFunction, GradientFunction, HessianFunction1,
                       HessianFunction2,
                       HessianFunction3>::hessian_uu(ConstVecXRef x,
                                                     ConstVecXRef u,
                                                     MatX        &Huu) const
{
    _hess_uu(x, u, Huu);
}

FunctionStageDynamicsBase::FunctionStageDynamicsBase(
  const std::vector<std::string> names, const uint nx, const uint nu,
  const uint nout, const uint ndx, const VecX &init_state,
  const VecX &init_control)
  : _nx(nx)
  , _nu(nu)
  , _nout(nout)
  , _ndx(ndx)
{
    _names = names;
    set_init_state(init_state);
    set_init_control(init_control);
}

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
FunctionStageDynamics<ManifoldOperator1, ManifoldOperator2, EvaluationFunction,
                      multiGradientFunction, HessprodFunction1,
                      HessprodFunction2, HessprodFunction3>::
  FunctionStageDynamics(
    const std::vector<std::string> names, const uint nx, const uint nu,
    const uint nout, const uint ndx, const VecX &init_state,
    const VecX                                             &init_control,
    ManifoldOperators<ManifoldOperator1, ManifoldOperator2> ops,
    EvaluationFunction eval, multiGradientFunction grad,
    HessprodFunction1 hess_xx, HessprodFunction2 hess_xu,
    HessprodFunction3 hess_uu)
  : FunctionStageDynamicsBase(names, nx, nu, nout, ndx, init_state,
                              init_control)
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

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
void FunctionStageDynamics<ManifoldOperator1, ManifoldOperator2,
                           EvaluationFunction, multiGradientFunction,
                           HessprodFunction1, HessprodFunction2,
                           HessprodFunction3>::state_right_plus(ConstVecXRef x,
                                                                ConstVecXRef dx,
                                                                VecX &out) const
{
    _ops.right_plus(x, dx, out);
}

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
void FunctionStageDynamics<
  ManifoldOperator1, ManifoldOperator2, EvaluationFunction,
  multiGradientFunction, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::state_right_minus(ConstVecXRef x, ConstVecXRef y,
                                        VecX &out) const
{
    _ops.right_minus(x, y, out);
}

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
void FunctionStageDynamics<ManifoldOperator1, ManifoldOperator2,
                           EvaluationFunction, multiGradientFunction,
                           HessprodFunction1, HessprodFunction2,
                           HessprodFunction3>::evaluate(ConstVecXRef x,
                                                        ConstVecXRef u,
                                                        VecX        &out) const
{
    _eval(x, u, out);
}

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
void FunctionStageDynamics<ManifoldOperator1, ManifoldOperator2,
                           EvaluationFunction, multiGradientFunction,
                           HessprodFunction1, HessprodFunction2,
                           HessprodFunction3>::gradient(ConstVecXRef x,
                                                        ConstVecXRef u,
                                                        MatX        &dX,
                                                        MatX        &dU) const
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
                           HessprodFunction3>::hessprod_xx(ConstVecXRef x,
                                                           ConstVecXRef u,
                                                           ConstVecXRef lambda,
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
                           HessprodFunction3>::hessprod_xu(ConstVecXRef x,
                                                           ConstVecXRef u,
                                                           ConstVecXRef lambda,
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
                           HessprodFunction3>::hessprod_uu(ConstVecXRef x,
                                                           ConstVecXRef u,
                                                           ConstVecXRef lambda,
                                                           MatX &Puu) const
{
    _hess_uu(x, u, lambda, Puu);
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
FunctionStageConstraints<NOutFunction, NdcFunction, LowerBoundFunction,
                         UpperBoundFunction, EvaluationFunction,
                         multiGradientFunction, StateOnlyRowsFunction,
                         BoxRowsFunction, ManifoldOperator, HessprodFunction1,
                         HessprodFunction2, HessprodFunction3>::
  FunctionStageConstraints(
    const std::vector<std::string> names, const NOutFunction nout,
    const NdcFunction ndc, const LowerBoundFunction lb,
    const UpperBoundFunction ub, const EvaluationFunction eval,
    const multiGradientFunction grad,
    const StateOnlyRowsFunction state_only_rows, const BoxRowsFunction box_rows,
    const ManifoldOperator op, const HessprodFunction1 hess_xx,
    const HessprodFunction2 hess_xu, const HessprodFunction3 hess_uu)
  : _nout(std::move(nout))
  , _ndc(std::move(ndc))
  , _lb(std::move(lb))
  , _ub(std::move(ub))
  , _eval(std::move(eval))
  , _grad(std::move(grad))
  , _state_only_rows(std::move(state_only_rows))
  , _box_rows(std::move(box_rows))
  , _op(std::move(op))
  , _hess_xx(std::move(hess_xx))
  , _hess_xu(std::move(hess_xu))
  , _hess_uu(std::move(hess_uu))
{
    _names = names;
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
uint FunctionStageConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, StateOnlyRowsFunction,
  BoxRowsFunction, ManifoldOperator, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::output_size() const
{
    return _nout();
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
uint FunctionStageConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, StateOnlyRowsFunction,
  BoxRowsFunction, ManifoldOperator, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::constraint_diff_size() const
{
    return _ndc();
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
const VecX &FunctionStageConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, StateOnlyRowsFunction,
  BoxRowsFunction, ManifoldOperator, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::lower_bound() const
{
    _lb_cache = _lb();

    return _lb_cache;
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
const VecX &FunctionStageConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, StateOnlyRowsFunction,
  BoxRowsFunction, ManifoldOperator, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::upper_bound() const
{
    _ub_cache = _ub();

    return _ub_cache;
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
void FunctionStageConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, StateOnlyRowsFunction,
  BoxRowsFunction, ManifoldOperator, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::state_only_rows(std::vector<uint> &out) const
{
    _state_only_rows(out);
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
void FunctionStageConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, StateOnlyRowsFunction,
  BoxRowsFunction, ManifoldOperator, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::box_rows(std::vector<BoxRow> &out) const
{
    _box_rows(out);
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
void FunctionStageConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, StateOnlyRowsFunction,
  BoxRowsFunction, ManifoldOperator, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::right_minus(ConstVecXRef x, ConstVecXRef y,
                                  VecX &out) const
{
    _op(x, y, out);
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
void FunctionStageConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, StateOnlyRowsFunction,
  BoxRowsFunction, ManifoldOperator, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::evaluate(ConstVecXRef x, ConstVecXRef u, VecX &out) const
{
    _eval(x, u, out);
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
void FunctionStageConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, StateOnlyRowsFunction,
  BoxRowsFunction, ManifoldOperator, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::gradient(ConstVecXRef x, ConstVecXRef u, MatX &dX,
                               MatX &dU) const
{
    _grad(x, u, dX, dU);
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
void FunctionStageConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, StateOnlyRowsFunction,
  BoxRowsFunction, ManifoldOperator, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::hessprod_xx(ConstVecXRef x, ConstVecXRef u,
                                  ConstVecXRef lambda, MatX &Pxx) const
{
    _hess_xx(x, u, lambda, Pxx);
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
void FunctionStageConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, StateOnlyRowsFunction,
  BoxRowsFunction, ManifoldOperator, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::hessprod_xu(ConstVecXRef x, ConstVecXRef u,
                                  ConstVecXRef lambda, MatX &Pxu) const
{
    _hess_xu(x, u, lambda, Pxu);
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
void FunctionStageConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, StateOnlyRowsFunction,
  BoxRowsFunction, ManifoldOperator, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::hessprod_uu(ConstVecXRef x, ConstVecXRef u,
                                  ConstVecXRef lambda, MatX &Puu) const
{
    _hess_uu(x, u, lambda, Puu);
}

// FunctionStageAddCost
template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
FunctionStageAddCost<
  EvaluationFunction, GradientFunction, HessianFunction1, HessianFunction2,
  HessianFunction3>::FunctionStageAddCost(const std::vector<std::string> names,
                                          const uint                     ndx,
                                          EvaluationFunction             eval,
                                          GradientFunction               grad,
                                          HessianFunction1 hess_xx,
                                          HessianFunction2 hess_xu,
                                          HessianFunction3 hess_uu)
  : FunctionStageCost<EvaluationFunction, GradientFunction, HessianFunction1,
                      HessianFunction2, HessianFunction3>(
      names, std::move(eval), std::move(grad), std::move(hess_xx),
      std::move(hess_xu), std::move(hess_uu))
  , _ndx(ndx)
{
}

template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
void FunctionStageAddCost<EvaluationFunction, GradientFunction,
                          HessianFunction1, HessianFunction2,
                          HessianFunction3>::evaluate(ConstVecXRef x,
                                                      ConstVecXRef u,
                                                      VecX        &out) const
{
    out.resize(1);
    evaluate_into(x, u, out);
}

template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
void FunctionStageAddCost<EvaluationFunction, GradientFunction,
                          HessianFunction1, HessianFunction2,
                          HessianFunction3>::gradient(ConstVecXRef x,
                                                      ConstVecXRef u, VecX &dx,
                                                      VecX &du) const
{
    dx.setZero(_ndx);
    du.setZero(u.size());
    add_gradient(x, u, dx, du);
}

template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
void FunctionStageAddCost<EvaluationFunction, GradientFunction,
                          HessianFunction1, HessianFunction2,
                          HessianFunction3>::hessian_xx(ConstVecXRef x,
                                                        ConstVecXRef u,
                                                        MatX        &Hxx) const
{
    Hxx.setZero(_ndx, _ndx);
    add_hessian_xx(x, u, Hxx);
}

template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
void FunctionStageAddCost<EvaluationFunction, GradientFunction,
                          HessianFunction1, HessianFunction2,
                          HessianFunction3>::hessian_xu(ConstVecXRef x,
                                                        ConstVecXRef u,
                                                        MatX        &Hxu) const
{
    Hxu.setZero(_ndx, u.size());
    add_hessian_xu(x, u, Hxu);
}

template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
void FunctionStageAddCost<EvaluationFunction, GradientFunction,
                          HessianFunction1, HessianFunction2,
                          HessianFunction3>::hessian_uu(ConstVecXRef x,
                                                        ConstVecXRef u,
                                                        MatX        &Huu) const
{
    Huu.setZero(u.size(), u.size());
    add_hessian_uu(x, u, Huu);
}

template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
void FunctionStageAddCost<EvaluationFunction, GradientFunction,
                          HessianFunction1, HessianFunction2,
                          HessianFunction3>::evaluate_into(ConstVecXRef x,
                                                           ConstVecXRef u,
                                                           VecXRef out) const
{
    this->_eval(x, u, out);
}

template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
void FunctionStageAddCost<EvaluationFunction, GradientFunction,
                          HessianFunction1, HessianFunction2,
                          HessianFunction3>::add_gradient(ConstVecXRef x,
                                                          ConstVecXRef u,
                                                          VecXRef      dx,
                                                          VecXRef      du) const
{
    this->_grad(x, u, dx, du);
}

template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
void FunctionStageAddCost<EvaluationFunction, GradientFunction,
                          HessianFunction1, HessianFunction2,
                          HessianFunction3>::add_hessian_xx(ConstVecXRef x,
                                                            ConstVecXRef u,
                                                            MatXRef Hxx) const
{
    this->_hess_xx(x, u, Hxx);
}

template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
void FunctionStageAddCost<EvaluationFunction, GradientFunction,
                          HessianFunction1, HessianFunction2,
                          HessianFunction3>::add_hessian_xu(ConstVecXRef x,
                                                            ConstVecXRef u,
                                                            MatXRef Hxu) const
{
    this->_hess_xu(x, u, Hxu);
}

template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
void FunctionStageAddCost<EvaluationFunction, GradientFunction,
                          HessianFunction1, HessianFunction2,
                          HessianFunction3>::add_hessian_uu(ConstVecXRef x,
                                                            ConstVecXRef u,
                                                            MatXRef Huu) const
{
    this->_hess_uu(x, u, Huu);
}

// FunctionStageAddDynamics
template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
void FunctionStageAddDynamics<
  ManifoldOperator1, ManifoldOperator2, EvaluationFunction,
  multiGradientFunction, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::state_right_plus(ConstVecXRef x, ConstVecXRef dx,
                                       VecX &out) const
{
    out.resize(this->_nx);
    state_right_plus_into(x, dx, out);
}

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
void FunctionStageAddDynamics<
  ManifoldOperator1, ManifoldOperator2, EvaluationFunction,
  multiGradientFunction, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::state_right_minus(ConstVecXRef x, ConstVecXRef y,
                                        VecX &out) const
{
    out.resize(this->_ndx);
    state_right_minus_into(x, y, out);
}

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
void FunctionStageAddDynamics<ManifoldOperator1, ManifoldOperator2,
                              EvaluationFunction, multiGradientFunction,
                              HessprodFunction1, HessprodFunction2,
                              HessprodFunction3>::evaluate(ConstVecXRef x,
                                                           ConstVecXRef u,
                                                           VecX &out) const
{
    out.resize(this->_nout);
    evaluate_into(x, u, out);
}

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
void FunctionStageAddDynamics<
  ManifoldOperator1, ManifoldOperator2, EvaluationFunction,
  multiGradientFunction, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::hessprod_xx(ConstVecXRef x, ConstVecXRef u,
                                  ConstVecXRef lambda, MatX &Pxx) const
{
    Pxx.setZero(this->_ndx, this->_ndx);
    add_hessprod_xx(x, u, lambda, Pxx);
}

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
void FunctionStageAddDynamics<
  ManifoldOperator1, ManifoldOperator2, EvaluationFunction,
  multiGradientFunction, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::hessprod_xu(ConstVecXRef x, ConstVecXRef u,
                                  ConstVecXRef lambda, MatX &Pxu) const
{
    Pxu.setZero(this->_ndx, this->_nu);
    add_hessprod_xu(x, u, lambda, Pxu);
}

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
void FunctionStageAddDynamics<
  ManifoldOperator1, ManifoldOperator2, EvaluationFunction,
  multiGradientFunction, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::hessprod_uu(ConstVecXRef x, ConstVecXRef u,
                                  ConstVecXRef lambda, MatX &Puu) const
{
    Puu.setZero(this->_nu, this->_nu);
    add_hessprod_uu(x, u, lambda, Puu);
}

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
void FunctionStageAddDynamics<
  ManifoldOperator1, ManifoldOperator2, EvaluationFunction,
  multiGradientFunction, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::state_right_plus_into(ConstVecXRef x, ConstVecXRef dx,
                                            VecXRef out) const
{
    this->_ops.right_plus(x, dx, out);
}

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
void FunctionStageAddDynamics<
  ManifoldOperator1, ManifoldOperator2, EvaluationFunction,
  multiGradientFunction, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::state_right_minus_into(ConstVecXRef x, ConstVecXRef y,
                                             VecXRef out) const
{
    this->_ops.right_minus(x, y, out);
}

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
void FunctionStageAddDynamics<
  ManifoldOperator1, ManifoldOperator2, EvaluationFunction,
  multiGradientFunction, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::evaluate_into(ConstVecXRef x, ConstVecXRef u,
                                    VecXRef out) const
{
    this->_eval(x, u, out);
}

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
void FunctionStageAddDynamics<
  ManifoldOperator1, ManifoldOperator2, EvaluationFunction,
  multiGradientFunction, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::add_hessprod_xx(ConstVecXRef x, ConstVecXRef u,
                                      ConstVecXRef lambda, MatXRef Pxx) const
{
    this->_hess_xx(x, u, lambda, Pxx);
}

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
void FunctionStageAddDynamics<
  ManifoldOperator1, ManifoldOperator2, EvaluationFunction,
  multiGradientFunction, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::add_hessprod_xu(ConstVecXRef x, ConstVecXRef u,
                                      ConstVecXRef lambda, MatXRef Pxu) const
{
    this->_hess_xu(x, u, lambda, Pxu);
}

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
void FunctionStageAddDynamics<
  ManifoldOperator1, ManifoldOperator2, EvaluationFunction,
  multiGradientFunction, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::add_hessprod_uu(ConstVecXRef x, ConstVecXRef u,
                                      ConstVecXRef lambda, MatXRef Puu) const
{
    this->_hess_uu(x, u, lambda, Puu);
}

// FunctionStageAddConstraints
template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
FunctionStageAddConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, StateOnlyRowsFunction,
  BoxRowsFunction, ManifoldOperator, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::
  FunctionStageAddConstraints(
    const std::vector<std::string> names, const uint ndx, NOutFunction nout,
    NdcFunction ndc, const LowerBoundFunction lb, const UpperBoundFunction ub,
    const EvaluationFunction eval, const multiGradientFunction grad,
    const StateOnlyRowsFunction state_only_rows, const BoxRowsFunction box_rows,
    const ManifoldOperator op, const HessprodFunction1 hess_xx,
    const HessprodFunction2 hess_xu, const HessprodFunction3 hess_uu)
  : FunctionStageConstraints<
      NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
      EvaluationFunction, multiGradientFunction, StateOnlyRowsFunction,
      BoxRowsFunction, ManifoldOperator, HessprodFunction1, HessprodFunction2,
      HessprodFunction3>(names, std::move(nout), std::move(ndc), std::move(lb),
                         std::move(ub), std::move(eval), std::move(grad),
                         std::move(state_only_rows), std::move(box_rows),
                         std::move(op), std::move(hess_xx), std::move(hess_xu),
                         std::move(hess_uu))
  , _ndx(ndx)
{
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
void FunctionStageAddConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, StateOnlyRowsFunction,
  BoxRowsFunction, ManifoldOperator, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::right_minus(ConstVecXRef x, ConstVecXRef y,
                                  VecX &out) const
{
    out.resize(this->constraint_diff_size());
    right_minus_into(x, y, out);
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
void FunctionStageAddConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, StateOnlyRowsFunction,
  BoxRowsFunction, ManifoldOperator, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::evaluate(ConstVecXRef x, ConstVecXRef u,
                               VecX &out) const
{
    out.resize(this->output_size());
    evaluate_into(x, u, out);
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
void FunctionStageAddConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, StateOnlyRowsFunction,
  BoxRowsFunction, ManifoldOperator, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::gradient(ConstVecXRef x, ConstVecXRef u, MatX &dX,
                               MatX &dU) const
{
    const uint ndc = this->constraint_diff_size();
    dX.setZero(ndc, _ndx);
    dU.setZero(ndc, u.size());
    add_gradient(x, u, dX, dU);
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
void FunctionStageAddConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, StateOnlyRowsFunction,
  BoxRowsFunction, ManifoldOperator, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::hessprod_xx(ConstVecXRef x, ConstVecXRef u,
                                  ConstVecXRef lambda, MatX &Pxx) const
{
    Pxx.setZero(_ndx, _ndx);
    add_hessprod_xx(x, u, lambda, Pxx);
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
void FunctionStageAddConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, StateOnlyRowsFunction,
  BoxRowsFunction, ManifoldOperator, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::hessprod_xu(ConstVecXRef x, ConstVecXRef u,
                                  ConstVecXRef lambda, MatX &Pxu) const
{
    Pxu.setZero(_ndx, u.size());
    add_hessprod_xu(x, u, lambda, Pxu);
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
void FunctionStageAddConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, StateOnlyRowsFunction,
  BoxRowsFunction, ManifoldOperator, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::hessprod_uu(ConstVecXRef x, ConstVecXRef u,
                                  ConstVecXRef lambda, MatX &Puu) const
{
    Puu.setZero(u.size(), u.size());
    add_hessprod_uu(x, u, lambda, Puu);
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
void FunctionStageAddConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, StateOnlyRowsFunction,
  BoxRowsFunction, ManifoldOperator, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::right_minus_into(ConstVecXRef x, ConstVecXRef y,
                                       VecXRef out) const
{
    this->_op(x, y, out);
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
void FunctionStageAddConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, StateOnlyRowsFunction,
  BoxRowsFunction, ManifoldOperator, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::evaluate_into(ConstVecXRef x, ConstVecXRef u,
                                    VecXRef out) const
{
    this->_eval(x, u, out);
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
void FunctionStageAddConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, StateOnlyRowsFunction,
  BoxRowsFunction, ManifoldOperator, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::add_gradient(ConstVecXRef x, ConstVecXRef u, MatXRef dX,
                                   MatXRef dU) const
{
    this->_grad(x, u, dX, dU);
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
void FunctionStageAddConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, StateOnlyRowsFunction,
  BoxRowsFunction, ManifoldOperator, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::add_hessprod_xx(ConstVecXRef x, ConstVecXRef u,
                                      ConstVecXRef lambda, MatXRef Pxx) const
{
    this->_hess_xx(x, u, lambda, Pxx);
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
void FunctionStageAddConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, StateOnlyRowsFunction,
  BoxRowsFunction, ManifoldOperator, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::add_hessprod_xu(ConstVecXRef x, ConstVecXRef u,
                                      ConstVecXRef lambda, MatXRef Pxu) const
{
    this->_hess_xu(x, u, lambda, Pxu);
}

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
void FunctionStageAddConstraints<
  NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
  EvaluationFunction, multiGradientFunction, StateOnlyRowsFunction,
  BoxRowsFunction, ManifoldOperator, HessprodFunction1, HessprodFunction2,
  HessprodFunction3>::add_hessprod_uu(ConstVecXRef x, ConstVecXRef u,
                                      ConstVecXRef lambda, MatXRef Puu) const
{
    this->_hess_uu(x, u, lambda, Puu);
}

}
}
}
}