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
class FramePositionBound : public StageConstraints
{
  public:
    FramePositionBound(const Model &model, const VecX &lb, const VecX &ub,
                       const uint nq, const uint nv);

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
    const VecX                   _lb, _ub;
    const uint                   _nq, _nv;
};

}
}
}
}
}

#endif