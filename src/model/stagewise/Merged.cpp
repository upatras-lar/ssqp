#include "model/stagewise/Merged.hpp"

#include <algorithm>
#include <stdexcept>

namespace ssqp
{
namespace model
{
namespace stagewise
{
// MergedStageCost
MergedStageCost::MergedStageCost() : StageCost("MergedStageCost")
{
    _names.clear();
}

void MergedStageCost::add(const std::shared_ptr<StageCost> &cost)
{
    _costs.push_back(cost);

    const std::vector<std::string> names = cost->names();
    _names.insert(_names.end(), names.begin(), names.end());
}

void MergedStageCost::set_enabled(const bool enabled)
{
    throw std::runtime_error("Cannot set the enabled state of a merged cost; "
                             "set it on the costs it was built from instead");
}

bool MergedStageCost::enabled() const
{
    return std::any_of(_costs.begin(),
                       _costs.end(),
                       [](const std::shared_ptr<StageCost> &cost)
                       { return cost->enabled(); });
}

void MergedStageCost::set_stage_forces(
  const std::vector<std::shared_ptr<Force>> &forces,
  const std::vector<std::shared_ptr<Force>> &wrenches)
{
    for (const std::shared_ptr<StageCost> &cost : _costs)
    {
        cost->set_stage_forces(forces, wrenches);
    }
}

VecX MergedStageCost::evaluate(const VecX &x, const VecX &u) const
{
    const bool any_enabled = enabled();

    Scalar out;
    out[0] = 0.0;

    for (uint i = 0; i < _costs.size(); ++i)
    {
        if (_is_enabled(i, any_enabled))
        {
            out += _costs[i]->evaluate(x, u);
        }
    }

    return out;
}

void MergedStageCost::gradient(const VecX &x, const VecX &u, VecX &dx,
                               VecX &du) const
{
    const bool any_enabled = enabled();
    bool       first       = true;

    for (uint i = 0; i < _costs.size(); ++i)
    {
        if (!_is_enabled(i, any_enabled))
        {
            continue;
        }

        if (first)
        {
            _costs[i]->gradient(x, u, dx, du);
            first = false;
        }
        else
        {
            _costs[i]->add_gradient(x, u, dx, du);
        }
    }
}

void MergedStageCost::add_gradient(const VecX &x, const VecX &u, VecXRef dx,
                                   VecXRef du) const
{
    const bool any_enabled = enabled();

    for (uint i = 0; i < _costs.size(); ++i)
    {
        if (_is_enabled(i, any_enabled))
        {
            _costs[i]->add_gradient(x, u, dx, du);
        }
    }
}

void MergedStageCost::hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const
{
    _sum_hessian(x, u, Hxx, &StageCost::hessian_xx, &StageCost::add_hessian_xx);
}

void MergedStageCost::hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const
{
    _sum_hessian(x, u, Hxu, &StageCost::hessian_xu, &StageCost::add_hessian_xu);
}

void MergedStageCost::hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const
{
    _sum_hessian(x, u, Huu, &StageCost::hessian_uu, &StageCost::add_hessian_uu);
}

void MergedStageCost::add_hessian_xx(const VecX &x, const VecX &u,
                                     MatXRef Hxx) const
{
    _add_hessian(x, u, Hxx, &StageCost::add_hessian_xx);
}

void MergedStageCost::add_hessian_xu(const VecX &x, const VecX &u,
                                     MatXRef Hxu) const
{
    _add_hessian(x, u, Hxu, &StageCost::add_hessian_xu);
}

void MergedStageCost::add_hessian_uu(const VecX &x, const VecX &u,
                                     MatXRef Huu) const
{
    _add_hessian(x, u, Huu, &StageCost::add_hessian_uu);
}

bool MergedStageCost::_is_enabled(const uint i, const bool any_enabled) const
{
    return any_enabled ? _costs[i]->enabled() : i == 0;
}

void MergedStageCost::_sum_hessian(const VecX &x, const VecX &u, MatX &H,
                                   const HessianFunc    hessian,
                                   const AddHessianFunc add_hessian) const
{
    const bool any_enabled = enabled();
    bool       first       = true;

    for (uint i = 0; i < _costs.size(); ++i)
    {
        if (!_is_enabled(i, any_enabled))
        {
            continue;
        }

        if (first)
        {
            (_costs[i].get()->*hessian)(x, u, H);
            first = false;
        }
        else
        {
            (_costs[i].get()->*add_hessian)(x, u, H);
        }
    }
}

void MergedStageCost::_add_hessian(const VecX &x, const VecX &u, MatXRef H,
                                   const AddHessianFunc add_hessian) const
{
    const bool any_enabled = enabled();

    for (uint i = 0; i < _costs.size(); ++i)
    {
        if (_is_enabled(i, any_enabled))
        {
            (_costs[i].get()->*add_hessian)(x, u, H);
        }
    }
}

// MergedStageConstraints
MergedStageConstraints::MergedStageConstraints()
  : StageConstraints("MergedStageConstraints")
{
    _names.clear();
}

void MergedStageConstraints::add(
  const std::shared_ptr<StageConstraints> &constraints)
{
    _constraints.push_back(constraints);
    _dX_scratch.emplace_back();
    _dU_scratch.emplace_back();

    const std::vector<std::string> names = constraints->names();
    _names.insert(_names.end(), names.begin(), names.end());
}

void MergedStageConstraints::set_enabled(const bool enabled)
{
    throw std::runtime_error("Cannot set the enabled state of merged "
                             "constraints; set it on the constraints they "
                             "were built from instead");
}

bool MergedStageConstraints::enabled() const
{
    return std::any_of(_constraints.begin(),
                       _constraints.end(),
                       [](const std::shared_ptr<StageConstraints> &constraints)
                       { return constraints->enabled(); });
}

uint MergedStageConstraints::output_size() const
{
    uint nc = 0;

    for (const auto &constraints : _constraints)
    {
        if (constraints->enabled())
        {
            nc += constraints->output_size();
        }
    }

    return nc;
}

uint MergedStageConstraints::constraint_diff_size() const
{
    uint ndc = 0;

    for (const auto &constraints : _constraints)
    {
        if (constraints->enabled())
        {
            ndc += constraints->constraint_diff_size();
        }
    }

    return ndc;
}

VecX MergedStageConstraints::right_minus(const VecX &x, const VecX &y) const
{
    VecX out(constraint_diff_size());
    uint row = 0, diff_row = 0;

    for (const auto &constraints : _constraints)
    {
        if (!constraints->enabled())
        {
            continue;
        }

        const uint nc  = constraints->output_size();
        const uint ndc = constraints->constraint_diff_size();

        out.segment(diff_row, ndc) =
          constraints->right_minus(x.segment(row, nc), y.segment(row, nc));

        row      += nc;
        diff_row += ndc;
    }

    return out;
}

const VecX &MergedStageConstraints::lower_bound() const
{
    _lb.resize(output_size());
    uint row = 0;

    for (const auto &constraints : _constraints)
    {
        if (constraints->enabled())
        {
            const uint nc         = constraints->output_size();
            _lb.segment(row, nc)  = constraints->lower_bound();
            row                  += nc;
        }
    }

    return _lb;
}

const VecX &MergedStageConstraints::upper_bound() const
{
    _ub.resize(output_size());
    uint row = 0;

    for (const auto &constraints : _constraints)
    {
        if (constraints->enabled())
        {
            const uint nc         = constraints->output_size();
            _ub.segment(row, nc)  = constraints->upper_bound();
            row                  += nc;
        }
    }

    return _ub;
}

VecX MergedStageConstraints::evaluate(const VecX &x, const VecX &u) const
{
    VecX out(output_size());
    uint row = 0;

    for (const auto &constraints : _constraints)
    {
        if (constraints->enabled())
        {
            const uint nc         = constraints->output_size();
            out.segment(row, nc)  = constraints->evaluate(x, u);
            row                  += nc;
        }
    }

    return out;
}

void MergedStageConstraints::gradient(const VecX &x, const VecX &u, MatX &dX,
                                      MatX &dU) const
{
    const uint ndc   = constraint_diff_size();
    bool       sized = false;
    uint       row   = 0;

    for (uint i = 0; i < _constraints.size(); ++i)
    {
        if (!_constraints[i]->enabled())
        {
            continue;
        }

        _constraints[i]->gradient(x, u, _dX_scratch[i], _dU_scratch[i]);

        if (!sized)
        {
            dX.setZero(ndc, _dX_scratch[i].cols());
            dU.setZero(ndc, u.size());
            sized = true;
        }

        const uint nc = _constraints[i]->constraint_diff_size();

        dX.middleRows(row, nc) = _dX_scratch[i];
        if (u.size())
        {
            dU.middleRows(row, nc) = _dU_scratch[i];
        }

        row += nc;
    }

    if (!sized)
    {
        // Without rows, the first constraints still give the column count
        _constraints.front()->gradient(x, u, _dX_scratch[0], _dU_scratch[0]);
        dX.setZero(0, _dX_scratch[0].cols());
        dU.setZero(0, u.size());
    }
}

void MergedStageConstraints::add_gradient(const VecX &x, const VecX &u,
                                          MatXRef dX, MatXRef dU) const
{
    uint row = 0;

    for (const auto &constraints : _constraints)
    {
        if (!constraints->enabled())
        {
            continue;
        }

        const uint nc = constraints->constraint_diff_size();
        constraints->add_gradient(x,
                                  u,
                                  dX.middleRows(row, nc),
                                  dU.middleRows(row, nc));
        row += nc;
    }
}

void MergedStageConstraints::hessprod_xx(const VecX &x, const VecX &u,
                                         const VecX &mu, MatX &Pxx) const
{
    _sum_hessprod(x,
                  u,
                  mu,
                  Pxx,
                  &StageConstraints::hessprod_xx,
                  &StageConstraints::add_hessprod_xx);
}

void MergedStageConstraints::hessprod_xu(const VecX &x, const VecX &u,
                                         const VecX &mu, MatX &Pxu) const
{
    _sum_hessprod(x,
                  u,
                  mu,
                  Pxu,
                  &StageConstraints::hessprod_xu,
                  &StageConstraints::add_hessprod_xu);
}

void MergedStageConstraints::hessprod_uu(const VecX &x, const VecX &u,
                                         const VecX &mu, MatX &Puu) const
{
    _sum_hessprod(x,
                  u,
                  mu,
                  Puu,
                  &StageConstraints::hessprod_uu,
                  &StageConstraints::add_hessprod_uu);
}

void MergedStageConstraints::add_hessprod_xx(const VecX &x, const VecX &u,
                                             const VecX &mu, MatXRef Pxx) const
{
    _add_hessprod(x, u, mu, Pxx, &StageConstraints::add_hessprod_xx);
}

void MergedStageConstraints::add_hessprod_xu(const VecX &x, const VecX &u,
                                             const VecX &mu, MatXRef Pxu) const
{
    _add_hessprod(x, u, mu, Pxu, &StageConstraints::add_hessprod_xu);
}

void MergedStageConstraints::add_hessprod_uu(const VecX &x, const VecX &u,
                                             const VecX &mu, MatXRef Puu) const
{
    _add_hessprod(x, u, mu, Puu, &StageConstraints::add_hessprod_uu);
}

void MergedStageConstraints::_sum_hessprod(
  const VecX &x, const VecX &u, const VecX &mu, MatX &P,
  const HessprodFunc hessprod, const AddHessprodFunc add_hessprod) const
{
    bool first = true;
    uint row   = 0;

    for (const auto &constraints : _constraints)
    {
        if (!constraints->enabled())
        {
            continue;
        }

        const uint nc = constraints->constraint_diff_size();

        if (first)
        {
            (constraints.get()->*hessprod)(x, u, mu.segment(row, nc), P);
            first = false;
        }
        else
        {
            (constraints.get()->*add_hessprod)(x, u, mu.segment(row, nc), P);
        }

        row += nc;
    }

    if (first)
    {
        // Without rows, the first constraints still give the product's size
        const auto &constraints = _constraints.front();
        (constraints.get()->*hessprod)(
          x,
          u,
          VecX::Zero(constraints->constraint_diff_size()),
          P);
        P.setZero();
    }
}

void MergedStageConstraints::_add_hessprod(
  const VecX &x, const VecX &u, const VecX &mu, MatXRef P,
  const AddHessprodFunc add_hessprod) const
{
    uint row = 0;

    for (const auto &constraints : _constraints)
    {
        if (!constraints->enabled())
        {
            continue;
        }

        const uint nc = constraints->constraint_diff_size();
        (constraints.get()->*add_hessprod)(x, u, mu.segment(row, nc), P);
        row += nc;
    }
}
}
}
}
