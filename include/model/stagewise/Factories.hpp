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
         typename HessianFunction3 = decltype(ZeroHessian_uu)>
inline std::shared_ptr<StageCost> make_cost(
  const EvaluationFunction &eval, const GradientFunction &grad,
  const HessianFunction1 &hess_xx = ZeroHessian_xx,
  const HessianFunction2 &hess_xu = ZeroHessian_xu,
  const HessianFunction3 &hess_uu = ZeroHessian_uu);

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1 = decltype(ZeroHessprod_xx),
         typename HessprodFunction2 = decltype(ZeroHessprod_xu),
         typename HessprodFunction3 = decltype(ZeroHessprod_uu)>
inline std::shared_ptr<StageDynamics> make_dynamics(
  const uint nx, const uint nu, const uint nout, const uint ndx,
  const VecX &init_state,
  const ManifoldOperators<ManifoldOperator1, ManifoldOperator2> &ops,
  const EvaluationFunction &eval, const multiGradientFunction &grad,
  const HessprodFunction1 &hess_xx = ZeroHessprod_xx,
  const HessprodFunction2 &hess_xu = ZeroHessprod_xu,
  const HessprodFunction3 &hess_uu = ZeroHessprod_uu);

template<typename EvaluationFunction, typename multiGradientFunction,
         typename ManifoldOperator  = decltype(DefaultRightMinus),
         typename HessprodFunction1 = decltype(ZeroHessprod_xx),
         typename HessprodFunction2 = decltype(ZeroHessprod_xu),
         typename HessprodFunction3 = decltype(ZeroHessprod_uu)>
inline std::shared_ptr<StageConstraints> make_constraints(
  const uint nout, const uint ndc, const VecX &lb, const VecX &ub,
  const EvaluationFunction &eval, const multiGradientFunction &grad,
  const ManifoldOperator &op       = DefaultRightMinus,
  const HessprodFunction1 &hess_xx = ZeroHessprod_xx,
  const HessprodFunction2 &hess_xu = ZeroHessprod_xu,
  const HessprodFunction3 &hess_uu = ZeroHessprod_uu);

inline std::shared_ptr<StageConstraints> merge_constraints(
  const std::shared_ptr<StageConstraints> &c1,
  const std::shared_ptr<StageConstraints> &c2);

template<typename EvaluationFunction, typename GradientFunction,
         typename HessianFunction1, typename HessianFunction2,
         typename HessianFunction3>
class FunctionStageCost : public StageCost
{
  public:
    inline FunctionStageCost(const EvaluationFunction &eval,
                             const GradientFunction   &grad,
                             const HessianFunction1   &hess_xx,
                             const HessianFunction2   &hess_xu,
                             const HessianFunction3   &hess_uu);

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
};

class FunctionStageDynamicsBase : public StageDynamics
{
  public:
    inline FunctionStageDynamicsBase(const uint nx, const uint nu,
                                     const uint nout, const uint ndx,
                                     const VecX &init_state);

    inline uint output_size() const override;
    inline uint state_size() const override;
    inline uint control_size() const override;
    inline uint state_diff_size() const override;
    inline VecX init_state() const override;
    inline void set_init_state(const VecX &init_state) override;

  protected:
    const uint _nx, _nu, _nout, _ndx;
    VecX       _init_state;
};

template<typename ManifoldOperator1, typename ManifoldOperator2,
         typename EvaluationFunction, typename multiGradientFunction,
         typename HessprodFunction1, typename HessprodFunction2,
         typename HessprodFunction3>
class FunctionStageDynamics : public FunctionStageDynamicsBase
{
  public:
    inline FunctionStageDynamics(
      const uint nx, const uint nu, const uint nout, const uint ndx,
      const VecX                                                    &init_state,
      const ManifoldOperators<ManifoldOperator1, ManifoldOperator2> &ops,
      const EvaluationFunction &eval, const multiGradientFunction &grad,
      const HessprodFunction1 &hess_xx, const HessprodFunction2 &hess_xu,
      const HessprodFunction3 &hess_uu);

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

class FunctionStageConstraintsBase : public StageConstraints
{
  public:
    inline FunctionStageConstraintsBase(const uint nout, const uint ndc,
                                        const VecX &lb, const VecX &ub);

    inline uint output_size() const override;
    inline VecX lower_bound() const override;
    inline VecX upper_bound() const override;

  protected:
    const uint _nout, _ndc;
    const VecX _lb, _ub;
};

template<typename EvaluationFunction, typename multiGradientFunction,
         typename ManifoldOperator, typename HessprodFunction1,
         typename HessprodFunction2, typename HessprodFunction3>
class FunctionStageConstraints : public FunctionStageConstraintsBase
{
  public:
    inline FunctionStageConstraints(
      const uint nout, const uint ndc, const VecX &lb, const VecX &ub,
      const EvaluationFunction &eval, const multiGradientFunction &grad,
      const ManifoldOperator &op, const HessprodFunction1 &hess_xx,
      const HessprodFunction2 &hess_xu, const HessprodFunction3 &hess_uu);

    inline uint constraint_diff_size() const override;

    inline VecX right_minus(const VecX &x, const VecX &y) const override;

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
    const EvaluationFunction    _eval;
    const multiGradientFunction _grad;
    const ManifoldOperator      _op;
    const HessprodFunction1     _hess_xx;
    const HessprodFunction2     _hess_xu;
    const HessprodFunction3     _hess_uu;
};
}
}
}
}

#include "src/model/stagewise/Factories.tpp"

#endif