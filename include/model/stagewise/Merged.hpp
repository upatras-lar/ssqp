#ifndef SSQP_MODEL_STAGEWISE_MERGED_HPP
#define SSQP_MODEL_STAGEWISE_MERGED_HPP

#include <memory>
#include <vector>

#include "types.hpp"
#include "model/stagewise/StageCost.hpp"
#include "model/stagewise/StageConstraints.hpp"

namespace ssqp
{
namespace model
{
namespace stagewise
{
/** The sum of a stage's costs, in the order they were added
 * Only the enabled costs are summed, or the first cost if none is enabled
 */
class MergedStageCost : public StageCost
{
  public:
    MergedStageCost();

    // Add a cost to the end of the sum
    void add(const std::shared_ptr<StageCost> &cost);

    // Not supported, since this is enabled if any of its costs is; set it on
    // those instead
    void set_enabled(const bool enabled) override;
    bool enabled() const override;

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    using HessianFunc = void (StageCost::*)(const VecX &, const VecX &,
                                            MatX &) const;

    // Whether the i-th cost should be included
    // Reduces to the cost's enabled status, unless all costs are disabled, in
    // which case only the first cost is considered enabled
    bool _is_enabled(const uint i, const bool any_enabled) const;

    // Sum of one of the hessians over the summed costs
    void _sum_hessian(const VecX &x, const VecX &u, MatX &H,
                      const HessianFunc hessian) const;

    std::vector<std::shared_ptr<StageCost>> _costs;
    mutable VecX                            _dx_scratch, _du_scratch;
    mutable MatX                            _hess_scratch;
};

/** A stage's constraints stacked in the order they were added
 * Only the enabled constraints contribute rows
 */
class MergedStageConstraints : public StageConstraints
{
  public:
    MergedStageConstraints();

    // Add constraints to the end of the stack
    void add(const std::shared_ptr<StageConstraints> &constraints);

    // Not supported, since this is enabled if any of its constraints is; set
    // it on those instead
    void set_enabled(const bool enabled) override;
    bool enabled() const override;

    uint output_size() const override;
    uint constraint_diff_size() const override;

    VecX right_minus(const VecX &x, const VecX &y) const override;

    const VecX &lower_bound() const override;
    const VecX &upper_bound() const override;

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
    using HessprodFunc = void (StageConstraints::*)(const VecX &, const VecX &,
                                                    const VecX &, MatX &) const;

    // Sum of one of the hessprods over the enabled constraints, each with its
    // part of mu
    void _sum_hessprod(const VecX &x, const VecX &u, const VecX &mu, MatX &P,
                       const HessprodFunc hessprod) const;

    std::vector<std::shared_ptr<StageConstraints>> _constraints;
    mutable VecX                                   _lb, _ub;
    // Gradients of each of the constraints, kept apart so that each keeps its
    // size between calls
    mutable std::vector<MatX> _dX_scratch, _dU_scratch;
    mutable MatX              _hess_scratch;
};
}
}
}

#endif
