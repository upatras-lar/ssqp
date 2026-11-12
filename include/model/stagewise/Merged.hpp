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

    void evaluate(ConstVecXRef x, ConstVecXRef u, VecX &out) const override;
    void evaluate_into(ConstVecXRef x, ConstVecXRef u,
                       VecXRef out) const override;
    void gradient(ConstVecXRef x, ConstVecXRef u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(ConstVecXRef x, ConstVecXRef u, MatX &Hxx) const override;
    void hessian_xu(ConstVecXRef x, ConstVecXRef u, MatX &Hxu) const override;
    void hessian_uu(ConstVecXRef x, ConstVecXRef u, MatX &Huu) const override;

    void add_gradient(ConstVecXRef x, ConstVecXRef u, VecXRef dx,
                      VecXRef du) const override;
    void add_hessian_xx(ConstVecXRef x, ConstVecXRef u,
                        MatXRef Hxx) const override;
    void add_hessian_xu(ConstVecXRef x, ConstVecXRef u,
                        MatXRef Hxu) const override;
    void add_hessian_uu(ConstVecXRef x, ConstVecXRef u,
                        MatXRef Huu) const override;

  protected:
    using HessianFunc    = void (StageCost::*)(ConstVecXRef, ConstVecXRef,
                                            MatX &) const;
    using AddHessianFunc = void (StageCost::*)(ConstVecXRef, ConstVecXRef,
                                               MatXRef) const;

    // Whether the i-th cost should be included
    // Reduces to the cost's enabled status, unless all costs are disabled, in
    // which case only the first cost is considered enabled
    bool _is_enabled(const uint i, const bool any_enabled) const;

    // Sum of one of the hessians over the summed costs
    // The first cost sizes H, and the others add to it
    void _sum_hessian(ConstVecXRef x, ConstVecXRef u, MatX &H,
                      const HessianFunc    hessian,
                      const AddHessianFunc add_hessian) const;

    // Add one of the hessians of every summed cost to H
    void _add_hessian(ConstVecXRef x, ConstVecXRef u, MatXRef H,
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

    // Each enabled constraint writes to its own rows
    void right_minus(ConstVecXRef x, ConstVecXRef y, VecX &out) const override;
    void right_minus_into(ConstVecXRef x, ConstVecXRef y,
                          VecXRef out) const override;

    const VecX &lower_bound() const override;
    const VecX &upper_bound() const override;

    // The state-only rows of each enabled constraint, offset to its rows
    void state_only_rows(std::vector<uint> &out) const override;

    // The box rows of each enabled constraint, offset to its rows
    void box_rows(std::vector<BoxRow> &out) const override;

    // Each enabled constraint writes to its own rows
    void evaluate(ConstVecXRef x, ConstVecXRef u, VecX &out) const override;
    void evaluate_into(ConstVecXRef x, ConstVecXRef u,
                       VecXRef out) const override;

    void gradient(ConstVecXRef x, ConstVecXRef u, MatX &dX,
                  MatX &dU) const override;

    void hessprod_xx(ConstVecXRef x, ConstVecXRef u, ConstVecXRef mu,
                     MatX &Pxx) const override;
    void hessprod_xu(ConstVecXRef x, ConstVecXRef u, ConstVecXRef mu,
                     MatX &Pxu) const override;
    void hessprod_uu(ConstVecXRef x, ConstVecXRef u, ConstVecXRef mu,
                     MatX &Puu) const override;

    // Each enabled constraint adds to its own rows
    void add_gradient(ConstVecXRef x, ConstVecXRef u, MatXRef dX,
                      MatXRef dU) const override;
    void add_hessprod_xx(ConstVecXRef x, ConstVecXRef u, ConstVecXRef mu,
                         MatXRef Pxx) const override;
    void add_hessprod_xu(ConstVecXRef x, ConstVecXRef u, ConstVecXRef mu,
                         MatXRef Pxu) const override;
    void add_hessprod_uu(ConstVecXRef x, ConstVecXRef u, ConstVecXRef mu,
                         MatXRef Puu) const override;

  protected:
    using HessprodFunc = void (StageConstraints::*)(ConstVecXRef, ConstVecXRef,
                                                    ConstVecXRef, MatX &) const;
    using AddHessprodFunc = void (StageConstraints::*)(ConstVecXRef,
                                                       ConstVecXRef,
                                                       ConstVecXRef,
                                                       MatXRef) const;

    // Sum of one of the hessprods over the enabled constraints, each with its
    // part of mu
    // The first constraints size P, and the others add to it
    void _sum_hessprod(ConstVecXRef x, ConstVecXRef u, ConstVecXRef mu, MatX &P,
                       const HessprodFunc    hessprod,
                       const AddHessprodFunc add_hessprod) const;

    // Add one of the hessprods of every enabled constraint to P, each with its
    // part of mu
    void _add_hessprod(ConstVecXRef x, ConstVecXRef u, ConstVecXRef mu,
                       MatXRef P, const AddHessprodFunc add_hessprod) const;

    std::vector<std::shared_ptr<StageConstraints>> _constraints;
    mutable VecX                                   _lb, _ub;
    // Gradients of each of the constraints, kept apart so that each keeps its
    // size between calls
    mutable std::vector<MatX> _dX_scratch, _dU_scratch;
    // Rows of one constraint at a time, before they are offset to its rows
    mutable std::vector<uint>   _state_only_rows_scratch;
    mutable std::vector<BoxRow> _box_rows_scratch;
};
}
}
}

#endif
