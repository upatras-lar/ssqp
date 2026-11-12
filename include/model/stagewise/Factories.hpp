#ifndef SSQP_MODEL_STAGEWISE_FACTORIES_HPP
#define SSQP_MODEL_STAGEWISE_FACTORIES_HPP

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
// Make stage functions from functions that set their outputs, sizing them
template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
inline std::shared_ptr<StageCost> make_cost(
  const std::vector<std::string> names, EvaluationFunction eval,
  GradientFunction grad, HessianFunction1 hess_xx, HessianFunction2 hess_xu,
  HessianFunction3 hess_uu);

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
inline std::shared_ptr<StageDynamics> make_dynamics(
  const std::vector<std::string> names, const uint nx, const uint nu,
  const uint nout, const uint ndx, const VecX &init_state,
  const VecX                                             &init_control,
  ManifoldOperators<ManifoldOperator1, ManifoldOperator2> ops,
  EvaluationFunction eval, multiGradientFunction grad,
  HessprodFunction1 hess_xx, HessprodFunction2 hess_xu,
  HessprodFunction3 hess_uu);

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
inline std::shared_ptr<StageConstraints> make_constraints(
  const std::vector<std::string> names, NOutFunction nout, NdcFunction ndc,
  LowerBoundFunction lb, UpperBoundFunction ub, EvaluationFunction eval,
  multiGradientFunction grad, StateOnlyRowsFunction state_only_rows,
  BoxRowsFunction box_rows, ManifoldOperator op, HessprodFunction1 hess_xx,
  HessprodFunction2 hess_xu, HessprodFunction3 hess_uu);

// Make stage functions from functions whose outputs are already sized, so the
// Hessians default to zero
// The evaluation and manifold operator functions write their outputs, while
// the derivative functions add to them
// ndx is the size of the state difference, which sizes the outputs when the
// stage functions set them
// The dynamics gradient still sets its outputs, as for make_dynamics
template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1 = decltype(ZeroHessian),
         typename HessianFunction2 = decltype(ZeroHessian),
         typename HessianFunction3 = decltype(ZeroHessian)>
inline std::shared_ptr<StageCost> make_add_cost(
  const std::vector<std::string> names, const uint ndx, EvaluationFunction eval,
  GradientFunction grad, HessianFunction1 hess_xx = ZeroHessian,
  HessianFunction2 hess_xu = ZeroHessian,
  HessianFunction3 hess_uu = ZeroHessian);

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1 = decltype(ZeroHessprod),
         typename HessprodFunction2 = decltype(ZeroHessprod),
         typename HessprodFunction3 = decltype(ZeroHessprod)>
inline std::shared_ptr<StageDynamics> make_add_dynamics(
  const std::vector<std::string> names, const uint nx, const uint nu,
  const uint nout, const uint ndx, const VecX &init_state,
  const VecX &init_control,
  ManifoldOperators<ManifoldOperator1, ManifoldOperator2> ops,
  EvaluationFunction eval, multiGradientFunction grad,
  HessprodFunction1 hess_xx = ZeroHessprod,
  HessprodFunction2 hess_xu = ZeroHessprod,
  HessprodFunction3 hess_uu = ZeroHessprod);

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction = decltype(NoStateOnlyRows),
         typename BoxRowsFunction       = decltype(NoBoxRows),
         typename ManifoldOperator      = decltype(DefaultRightMinus),
         typename HessprodFunction1     = decltype(ZeroHessprod),
         typename HessprodFunction2     = decltype(ZeroHessprod),
         typename HessprodFunction3     = decltype(ZeroHessprod)>
inline std::shared_ptr<StageConstraints> make_add_constraints(
  const std::vector<std::string> names, const uint ndx, NOutFunction nout,
  NdcFunction ndc, LowerBoundFunction lb, UpperBoundFunction ub,
  EvaluationFunction eval, multiGradientFunction grad,
  StateOnlyRowsFunction state_only_rows = NoStateOnlyRows,
  BoxRowsFunction box_rows = NoBoxRows, ManifoldOperator op = DefaultRightMinus,
  HessprodFunction1 hess_xx = ZeroHessprod,
  HessprodFunction2 hess_xu = ZeroHessprod,
  HessprodFunction3 hess_uu = ZeroHessprod);

template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
class FunctionStageCost : public StageCost
{
  public:
    inline FunctionStageCost(const std::vector<std::string> names,
                             EvaluationFunction eval, GradientFunction grad,
                             HessianFunction1 hess_xx, HessianFunction2 hess_xu,
                             HessianFunction3 hess_uu);

    inline void evaluate(ConstVecXRef x, ConstVecXRef u,
                         VecX &out) const override;
    inline void gradient(ConstVecXRef x, ConstVecXRef u, VecX &dx,
                         VecX &du) const override;
    inline void hessian_xx(ConstVecXRef x, ConstVecXRef u,
                           MatX &Hxx) const override;
    inline void hessian_xu(ConstVecXRef x, ConstVecXRef u,
                           MatX &Hxu) const override;
    inline void hessian_uu(ConstVecXRef x, ConstVecXRef u,
                           MatX &Huu) const override;

  protected:
    const EvaluationFunction _eval;
    const GradientFunction   _grad;
    const HessianFunction1   _hess_xx;
    const HessianFunction2   _hess_xu;
    const HessianFunction3   _hess_uu;
};

class FunctionStageDynamicsBase : public StageDynamics
{
  public:
    inline FunctionStageDynamicsBase(const std::vector<std::string> names,
                                     const uint nx, const uint nu,
                                     const uint nout, const uint ndx,
                                     const VecX &init_state,
                                     const VecX &init_control);

    inline uint output_size() const override;
    inline uint state_size() const override;
    inline uint control_size() const override;
    inline uint state_diff_size() const override;

  protected:
    const uint _nx, _nu, _nout, _ndx;
};

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
class FunctionStageDynamics : public FunctionStageDynamicsBase
{
  public:
    inline FunctionStageDynamics(
      const std::vector<std::string> names, const uint nx, const uint nu,
      const uint nout, const uint ndx, const VecX &init_state,
      const VecX                                             &init_control,
      ManifoldOperators<ManifoldOperator1, ManifoldOperator2> ops,
      EvaluationFunction eval, const multiGradientFunction grad,
      HessprodFunction1 hess_xx, const HessprodFunction2 hess_xu,
      HessprodFunction3 hess_uu);

    inline void state_right_plus(ConstVecXRef x, ConstVecXRef dx,
                                 VecX &out) const override;
    inline void state_right_minus(ConstVecXRef x, ConstVecXRef y,
                                  VecX &out) const override;

    inline void evaluate(ConstVecXRef x, ConstVecXRef u,
                         VecX &out) const override;

    inline void gradient(ConstVecXRef x, ConstVecXRef u, MatX &dX,
                         MatX &dU) const override;

    inline void hessprod_xx(ConstVecXRef x, ConstVecXRef u, ConstVecXRef lambda,
                            MatX &Pxx) const override;

    inline void hessprod_xu(ConstVecXRef x, ConstVecXRef u, ConstVecXRef lambda,
                            MatX &Pxu) const override;

    inline void hessprod_uu(ConstVecXRef x, ConstVecXRef u, ConstVecXRef lambda,
                            MatX &Puu) const override;

  protected:
    const ManifoldOperators<ManifoldOperator1, ManifoldOperator2> _ops;
    const EvaluationFunction                                      _eval;
    const multiGradientFunction                                   _grad;
    const HessprodFunction1                                       _hess_xx;
    const HessprodFunction2                                       _hess_xu;
    const HessprodFunction3                                       _hess_uu;
};

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
class FunctionStageConstraints : public StageConstraints
{
  public:
    inline FunctionStageConstraints(
      const std::vector<std::string> names, NOutFunction nout, NdcFunction ndc,
      const LowerBoundFunction lb, const UpperBoundFunction ub,
      const EvaluationFunction eval, const multiGradientFunction grad,
      const StateOnlyRowsFunction state_only_rows,
      const BoxRowsFunction box_rows, const ManifoldOperator op,
      const HessprodFunction1 hess_xx, const HessprodFunction2 hess_xu,
      const HessprodFunction3 hess_uu);

    inline uint output_size() const override;

    inline uint constraint_diff_size() const override;

    inline void right_minus(ConstVecXRef x, ConstVecXRef y,
                            VecX &out) const override;

    inline const VecX &lower_bound() const override;
    inline const VecX &upper_bound() const override;

    inline void state_only_rows(std::vector<uint> &out) const override;
    inline void box_rows(std::vector<BoxRow> &out) const override;

    inline void evaluate(ConstVecXRef x, ConstVecXRef u,
                         VecX &out) const override;

    inline void gradient(ConstVecXRef x, ConstVecXRef u, MatX &dX,
                         MatX &dU) const override;

    inline void hessprod_xx(ConstVecXRef x, ConstVecXRef u, ConstVecXRef lambda,
                            MatX &Pxx) const override;

    inline void hessprod_xu(ConstVecXRef x, ConstVecXRef u, ConstVecXRef lambda,
                            MatX &Pxu) const override;

    inline void hessprod_uu(ConstVecXRef x, ConstVecXRef u, ConstVecXRef lambda,
                            MatX &Puu) const override;

  protected:
    const NOutFunction          _nout;
    const NdcFunction           _ndc;
    const LowerBoundFunction    _lb;
    const UpperBoundFunction    _ub;
    const EvaluationFunction    _eval;
    const multiGradientFunction _grad;
    const StateOnlyRowsFunction _state_only_rows;
    const BoxRowsFunction       _box_rows;
    const ManifoldOperator      _op;
    const HessprodFunction1     _hess_xx;
    const HessprodFunction2     _hess_xu;
    const HessprodFunction3     _hess_uu;

  private:
    // Scratch backing lower_bound()/upper_bound(); recomputed on every call
    // (bounds may depend on mutable state) but reused across calls to avoid a
    // fresh allocation
    mutable VecX _lb_cache, _ub_cache;
};

// A FunctionStageCost whose gradient and Hessian functions add to their
// outputs
// Setting an output sizes it and zeroes it before adding
template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
class FunctionStageAddCost
  : public FunctionStageCost<EvaluationFunction, GradientFunction,
                             HessianFunction1, HessianFunction2,
                             HessianFunction3>
{
  public:
    inline FunctionStageAddCost(const std::vector<std::string> names,
                                const uint ndx, EvaluationFunction eval,
                                GradientFunction grad, HessianFunction1 hess_xx,
                                HessianFunction2 hess_xu,
                                HessianFunction3 hess_uu);

    inline void evaluate(ConstVecXRef x, ConstVecXRef u,
                         VecX &out) const override;
    inline void gradient(ConstVecXRef x, ConstVecXRef u, VecX &dx,
                         VecX &du) const override;
    inline void hessian_xx(ConstVecXRef x, ConstVecXRef u,
                           MatX &Hxx) const override;
    inline void hessian_xu(ConstVecXRef x, ConstVecXRef u,
                           MatX &Hxu) const override;
    inline void hessian_uu(ConstVecXRef x, ConstVecXRef u,
                           MatX &Huu) const override;

    inline void evaluate_into(ConstVecXRef x, ConstVecXRef u,
                              VecXRef out) const override;
    inline void add_gradient(ConstVecXRef x, ConstVecXRef u, VecXRef dx,
                             VecXRef du) const override;
    inline void add_hessian_xx(ConstVecXRef x, ConstVecXRef u,
                               MatXRef Hxx) const override;
    inline void add_hessian_xu(ConstVecXRef x, ConstVecXRef u,
                               MatXRef Hxu) const override;
    inline void add_hessian_uu(ConstVecXRef x, ConstVecXRef u,
                               MatXRef Huu) const override;

  protected:
    const uint _ndx;
};

// A FunctionStageDynamics whose Hessian product functions add to their
// outputs
// Setting an output sizes it and zeroes it before adding; the gradient
// function still sets its outputs
template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
class FunctionStageAddDynamics
  : public FunctionStageDynamics<ManifoldOperator1, ManifoldOperator2,
                                 EvaluationFunction, multiGradientFunction,
                                 HessprodFunction1, HessprodFunction2,
                                 HessprodFunction3>
{
  public:
    using FunctionStageDynamics<ManifoldOperator1, ManifoldOperator2,
                                EvaluationFunction, multiGradientFunction,
                                HessprodFunction1, HessprodFunction2,
                                HessprodFunction3>::FunctionStageDynamics;

    inline void state_right_plus(ConstVecXRef x, ConstVecXRef dx,
                                 VecX &out) const override;
    inline void state_right_minus(ConstVecXRef x, ConstVecXRef y,
                                  VecX &out) const override;
    inline void evaluate(ConstVecXRef x, ConstVecXRef u,
                         VecX &out) const override;
    inline void hessprod_xx(ConstVecXRef x, ConstVecXRef u, ConstVecXRef lambda,
                            MatX &Pxx) const override;
    inline void hessprod_xu(ConstVecXRef x, ConstVecXRef u, ConstVecXRef lambda,
                            MatX &Pxu) const override;
    inline void hessprod_uu(ConstVecXRef x, ConstVecXRef u, ConstVecXRef lambda,
                            MatX &Puu) const override;

    inline void state_right_plus_into(ConstVecXRef x, ConstVecXRef dx,
                                      VecXRef out) const override;
    inline void state_right_minus_into(ConstVecXRef x, ConstVecXRef y,
                                       VecXRef out) const override;
    inline void evaluate_into(ConstVecXRef x, ConstVecXRef u,
                              VecXRef out) const override;
    inline void add_hessprod_xx(ConstVecXRef x, ConstVecXRef u,
                                ConstVecXRef lambda,
                                MatXRef      Pxx) const override;
    inline void add_hessprod_xu(ConstVecXRef x, ConstVecXRef u,
                                ConstVecXRef lambda,
                                MatXRef      Pxu) const override;
    inline void add_hessprod_uu(ConstVecXRef x, ConstVecXRef u,
                                ConstVecXRef lambda,
                                MatXRef      Puu) const override;
};

// A FunctionStageConstraints whose gradient and Hessian product functions add
// to their outputs
// Setting an output sizes it and zeroes it before adding
template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename StateOnlyRowsFunction, typename BoxRowsFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
class FunctionStageAddConstraints
  : public FunctionStageConstraints<
      NOutFunction, NdcFunction, LowerBoundFunction, UpperBoundFunction,
      EvaluationFunction, multiGradientFunction, StateOnlyRowsFunction,
      BoxRowsFunction, ManifoldOperator, HessprodFunction1, HessprodFunction2,
      HessprodFunction3>
{
  public:
    inline FunctionStageAddConstraints(
      const std::vector<std::string> names, const uint ndx, NOutFunction nout,
      NdcFunction ndc, const LowerBoundFunction lb, const UpperBoundFunction ub,
      const EvaluationFunction eval, const multiGradientFunction grad,
      const StateOnlyRowsFunction state_only_rows,
      const BoxRowsFunction box_rows, const ManifoldOperator op,
      const HessprodFunction1 hess_xx, const HessprodFunction2 hess_xu,
      const HessprodFunction3 hess_uu);

    inline void right_minus(ConstVecXRef x, ConstVecXRef y,
                            VecX &out) const override;
    inline void evaluate(ConstVecXRef x, ConstVecXRef u,
                         VecX &out) const override;
    inline void gradient(ConstVecXRef x, ConstVecXRef u, MatX &dX,
                         MatX &dU) const override;
    inline void hessprod_xx(ConstVecXRef x, ConstVecXRef u, ConstVecXRef lambda,
                            MatX &Pxx) const override;
    inline void hessprod_xu(ConstVecXRef x, ConstVecXRef u, ConstVecXRef lambda,
                            MatX &Pxu) const override;
    inline void hessprod_uu(ConstVecXRef x, ConstVecXRef u, ConstVecXRef lambda,
                            MatX &Puu) const override;

    inline void right_minus_into(ConstVecXRef x, ConstVecXRef y,
                                 VecXRef out) const override;
    inline void evaluate_into(ConstVecXRef x, ConstVecXRef u,
                              VecXRef out) const override;
    inline void add_gradient(ConstVecXRef x, ConstVecXRef u, MatXRef dX,
                             MatXRef dU) const override;
    inline void add_hessprod_xx(ConstVecXRef x, ConstVecXRef u,
                                ConstVecXRef lambda,
                                MatXRef      Pxx) const override;
    inline void add_hessprod_xu(ConstVecXRef x, ConstVecXRef u,
                                ConstVecXRef lambda,
                                MatXRef      Pxu) const override;
    inline void add_hessprod_uu(ConstVecXRef x, ConstVecXRef u,
                                ConstVecXRef lambda,
                                MatXRef      Puu) const override;

  protected:
    const uint _ndx;
};
}
}
}
}

#include "src/model/stagewise/Factories.tpp"

#endif