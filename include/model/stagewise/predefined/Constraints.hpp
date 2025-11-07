#ifndef SSQP_MODEL_STAGEWISE_PREDEFINED_CONSTRAINTS_HPP
#define SSQP_MODEL_STAGEWISE_PREDEFINED_CONSTRAINTS_HPP

#include "types.hpp"
#include "model/Model.hpp"
#include "model/stagewise/StageConstraints.hpp"

namespace ssqp
{
namespace model
{
namespace stagewise
{
namespace predefined
{
namespace constraints
{
/* Predefined 3D position bound for a given frame
 * Creates a linear bound constraint on the position of a frame of the given
 * model: $\boldsymbol{b}_l <= \boldsymbol{p}_f <= \boldsymbol{b}_u$
 * where:
 *  - $\boldsymbol{b}_l$, $\boldsymbol{b}_u$: Lower and upper bound vectors (in $\mathcal{R}^3$)
 *  - $\boldsymbol{p}_f$: Frame position vector (in $\mathcal{R^3}$)
 */
class FramePositionBound : public StageConstraints
{
  public:
    // Regular constructor
    FramePositionBound(const Model &model, const std::string &frame,
                       const VecX &lb, const VecX &ub, const uint nq,
                       const uint nv);

    // Terminal constraint constructor (no controls)
    FramePositionBound(const Model &model, const std::string &frame,
                       const VecX &lb, const VecX &ub, const uint nq);

    uint output_size() const override;

    VecX lower_bound() const override;
    VecX upper_bound() const override;

    VecX evaluate(const VecX &x, const VecX &u) const override;

    void gradient(const VecX &x, const VecX &u, MatX &dX,
                  MatX &dU) const override;

    void hessprod_xx(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxx) const override;
    void hessprod_xu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxu) const override;
    void hessprod_uu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Puu) const override;

  protected:
    const std::shared_ptr<Model> _model;
    const std::string            _frame;
    const VecX                   _lb, _ub;
    const uint                   _nq, _nv;
};

class ContactConstraint : public StageConstraints
{
};

}
}
}
}
}

#endif