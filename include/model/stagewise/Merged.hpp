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

    // Passes the stage forces on to every summed cost
    void set_stage_forces(
      const std::vector<std::shared_ptr<Force>> &forces,
      const std::vector<std::shared_ptr<Force>> &wrenches) override;

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

    void add_gradient(const VecX &x, const VecX &u, VecXRef dx,
                      VecXRef du) const override;
    void add_hessian_xx(const VecX &x, const VecX &u,
                        MatXRef Hxx) const override;
    void add_hessian_xu(const VecX &x, const VecX &u,
                        MatXRef Hxu) const override;
    void add_hessian_uu(const VecX &x, const VecX &u,
                        MatXRef Huu) const override;

  protected:
    using HessianFunc    = void (StageCost::*)(const VecX &, const VecX &,
                                            MatX &) const;
    using AddHessianFunc = void (StageCost::*)(const VecX &, const VecX &,
                                               MatXRef) const;

    // Whether the i-th cost should be included
    // Reduces to the cost's enabled status, unless all costs are disabled, in
    // which case only the first cost is considered enabled
    bool _is_enabled(const uint i, const bool any_enabled) const;

    // Sum of one of the hessians over the summed costs
    // The first cost sizes H, and the others add to it
    void _sum_hessian(const VecX &x, const VecX &u, MatX &H,
                      const HessianFunc    hessian,
                      const AddHessianFunc add_hessian) const;

    // Add one of the hessians of every summed cost to H
    void _add_hessian(const VecX &x, const VecX &u, MatXRef H,
                      const AddHessianFunc add_hessian) const;

    std::vector<std::shared_ptr<StageCost>> _costs;
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

    // Each enabled constraint adds to its own rows
    void add_gradient(const VecX &x, const VecX &u, MatXRef dX,
                      MatXRef dU) const override;
    void add_hessprod_xx(const VecX &x, const VecX &u, const VecX &mu,
                         MatXRef Pxx) const override;
    void add_hessprod_xu(const VecX &x, const VecX &u, const VecX &mu,
                         MatXRef Pxu) const override;
    void add_hessprod_uu(const VecX &x, const VecX &u, const VecX &mu,
                         MatXRef Puu) const override;

  protected:
    using HessprodFunc = void (StageConstraints::*)(const VecX &, const VecX &,
                                                    const VecX &, MatX &) const;
    using AddHessprodFunc = void (StageConstraints::*)(const VecX &,
                                                       const VecX &,
                                                       const VecX &,
                                                       MatXRef) const;

    // Sum of one of the hessprods over the enabled constraints, each with its
    // part of mu
    // The first constraints size P, and the others add to it
    void _sum_hessprod(const VecX &x, const VecX &u, const VecX &mu, MatX &P,
                       const HessprodFunc    hessprod,
                       const AddHessprodFunc add_hessprod) const;

    // Add one of the hessprods of every enabled constraint to P, each with its
    // part of mu
    void _add_hessprod(const VecX &x, const VecX &u, const VecX &mu, MatXRef P,
                       const AddHessprodFunc add_hessprod) const;

    std::vector<std::shared_ptr<StageConstraints>> _constraints;
    mutable VecX                                   _lb, _ub;
    // Gradients of each of the constraints, kept apart so that each keeps its
    // size between calls
    mutable std::vector<MatX> _dX_scratch, _dU_scratch;
};
}
}
}

#endif
