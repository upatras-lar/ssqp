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
template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1 = decltype(ZeroHessian_xx),
         typename HessianFunction2 = decltype(ZeroHessian_xu),
         typename HessianFunction3 = decltype(ZeroHessian_uu),
         typename EnabledFunction  = decltype(DefaultEnabledFunc)>
inline std::shared_ptr<StageCost> make_cost(
  const std::vector<std::string> names, EvaluationFunction eval,
  GradientFunction grad, HessianFunction1 hess_xx = ZeroHessian_xx,
  HessianFunction2 hess_xu = ZeroHessian_xu,
  HessianFunction3 hess_uu = ZeroHessian_uu,
  EnabledFunction enabled_func = DefaultEnabledFunc);

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1 = decltype(ZeroHessprod_xx),
         typename HessprodFunction2 = decltype(ZeroHessprod_xu),
         typename HessprodFunction3 = decltype(ZeroHessprod_uu)>
inline std::shared_ptr<StageDynamics> make_dynamics(
  const std::vector<std::string> names, const uint nx, const uint nu,
  const uint nout, const uint ndx, const VecX &init_state,
  const VecX &init_control,
  ManifoldOperators<ManifoldOperator1, ManifoldOperator2> ops,
  EvaluationFunction eval, multiGradientFunction grad,
  HessprodFunction1 hess_xx = ZeroHessprod_xx,
  HessprodFunction2 hess_xu = ZeroHessprod_xu,
  HessprodFunction3 hess_uu = ZeroHessprod_uu);

template<typename NOutFunction, typename NdcFunction,
         typename LowerBoundFunction, typename UpperBoundFunction,
         typename EvaluationFunction, typename multiGradientFunction,
         typename ManifoldOperator  = decltype(DefaultRightMinus),
         typename HessprodFunction1 = decltype(ZeroHessprod_xx),
         typename HessprodFunction2 = decltype(ZeroHessprod_xu),
         typename HessprodFunction3 = decltype(ZeroHessprod_uu),
         typename EnabledFunction   = decltype(DefaultEnabledFunc)>
inline std::shared_ptr<StageConstraints> make_constraints(
  const std::vector<std::string> names, NOutFunction nout, NdcFunction ndc,
  LowerBoundFunction lb, UpperBoundFunction ub, EvaluationFunction eval,
  multiGradientFunction grad, ManifoldOperator op = DefaultRightMinus,
  HessprodFunction1 hess_xx    = ZeroHessprod_xx,
  HessprodFunction2 hess_xu    = ZeroHessprod_xu,
  HessprodFunction3 hess_uu    = ZeroHessprod_uu,
  EnabledFunction enabled_func = DefaultEnabledFunc);

inline std::shared_ptr<StageCost> merge_costs(
  const std::shared_ptr<StageCost> &l1, const std::shared_ptr<StageCost> &l2);

inline std::shared_ptr<StageConstraints> merge_constraints(
  const std::shared_ptr<StageConstraints> &c1,
  const std::shared_ptr<StageConstraints> &c2);

template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3, typename EnabledFunction>
class FunctionStageCost : public StageCost
{
  public:
    inline FunctionStageCost(const std::vector<std::string> names,
                             EvaluationFunction eval, GradientFunction grad,
                             HessianFunction1 hess_xx, HessianFunction2 hess_xu,
                             HessianFunction3 hess_uu,
                             EnabledFunction  enabled_func);

    // Not supported for a custom enabled function (merged costs), which is
    // enabled if any of the costs it is made of is; set it on those instead
    inline void set_enabled(const bool enabled) override;
    inline bool enabled() const override;

    inline VecX evaluate(const VecX &x, const VecX &u) const override;
    inline void gradient(const VecX &x, const VecX &u, VecX &dx,
                         VecX &du) const override;
    inline void hessian_xx(const VecX &x, const VecX &u,
                           MatX &Hxx) const override;
    inline void hessian_xu(const VecX &x, const VecX &u,
                           MatX &Hxu) const override;
    inline void hessian_uu(const VecX &x, const VecX &u,
                           MatX &Huu) const override;

  protected:
    const EvaluationFunction _eval;
    const GradientFunction   _grad;
    const HessianFunction1   _hess_xx;
    const HessianFunction2   _hess_xu;
    const HessianFunction3   _hess_uu;
    const EnabledFunction    _enabled_func;
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

    inline VecX state_right_plus(const VecX &x, const VecX &dx) const override;
    inline VecX state_right_minus(const VecX &x, const VecX &y) const override;

    inline VecX evaluate(const VecX &x, const VecX &u) const override;

    inline void gradient(const VecX &x, const VecX &u, MatX &dX,
                         MatX &dU) const override;

    inline void hessprod_xx(const VecX &x, const VecX &u, const VecX &lambda,
                            MatX &Pxx) const override;

    inline void hessprod_xu(const VecX &x, const VecX &u, const VecX &lambda,
                            MatX &Pxu) const override;

    inline void hessprod_uu(const VecX &x, const VecX &u, const VecX &lambda,
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
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3,
         typename EnabledFunction>
class FunctionStageConstraints : public StageConstraints
{
  public:
    inline FunctionStageConstraints(
      const std::vector<std::string> names, NOutFunction nout, NdcFunction ndc,
      const LowerBoundFunction lb, const UpperBoundFunction ub,
      const EvaluationFunction eval, const multiGradientFunction grad,
      const ManifoldOperator op, const HessprodFunction1 hess_xx,
      const HessprodFunction2 hess_xu, const HessprodFunction3 hess_uu,
      const EnabledFunction enabled_func);

    // Not supported: this is always a combination of other constraints, whose
    // enabled state is set on those directly
    inline void set_enabled(const bool enabled) override;

    inline bool enabled() const override;

    inline uint output_size() const override;

    inline uint constraint_diff_size() const override;

    inline VecX right_minus(const VecX &x, const VecX &y) const override;

    inline const VecX &lower_bound() const override;
    inline const VecX &upper_bound() const override;

    inline VecX evaluate(const VecX &x, const VecX &u) const override;

    inline void gradient(const VecX &x, const VecX &u, MatX &dX,
                         MatX &dU) const override;

    inline void hessprod_xx(const VecX &x, const VecX &u, const VecX &lambda,
                            MatX &Pxx) const override;

    inline void hessprod_xu(const VecX &x, const VecX &u, const VecX &lambda,
                            MatX &Pxu) const override;

    inline void hessprod_uu(const VecX &x, const VecX &u, const VecX &lambda,
                            MatX &Puu) const override;

  protected:
    const NOutFunction          _nout;
    const NdcFunction           _ndc;
    const LowerBoundFunction    _lb;
    const UpperBoundFunction    _ub;
    const EvaluationFunction    _eval;
    const multiGradientFunction _grad;
    const ManifoldOperator      _op;
    const HessprodFunction1     _hess_xx;
    const HessprodFunction2     _hess_xu;
    const HessprodFunction3     _hess_uu;
    const EnabledFunction       _enabled_func;

  private:
    // Scratch backing lower_bound()/upper_bound(); recomputed on every call
    // (bounds may depend on mutable state, e.g. enabled() for merged
    // constraints) but reused across calls to avoid a fresh allocation
    mutable VecX _lb_cache, _ub_cache;
};
}
}
}
}

#include "src/model/stagewise/Factories.tpp"

#endif