#include "types.hpp"

namespace ssqp
{
namespace solvers
{
FlatVec::FlatVec() {}

FlatVec::FlatVec(const uint total_size, const std::vector<uint> &dims)
  : vec(total_size)
  , _stage_sizes(dims.size())
  , _acc_idx(dims.size())
{
    _stage_sizes[0]                 = dims[0];
    _acc_idx[0]                     = 0;
    vec.segment(0, _stage_sizes[0]) = VecX::Zero(dims[0]);

    for (uint k = 1; k < dims.size(); ++k)
    {
        _stage_sizes[k] = dims[k];
        _acc_idx[k]     = _acc_idx[k - 1] + _stage_sizes[k - 1];
        vec.segment(_acc_idx[k], _stage_sizes[k]) = VecX::Zero(dims[k]);
    }
}

FlatVec::FlatVec(const uint total_size, const std::vector<VecX> &vectors)
  : vec(total_size)
  , _stage_sizes(vectors.size())
  , _acc_idx(vectors.size())
{
    _stage_sizes[0]                 = vectors[0].size();
    _acc_idx[0]                     = 0;
    vec.segment(0, _stage_sizes[0]) = vectors[0];

    for (uint k = 1; k < vectors.size(); ++k)
    {
        _stage_sizes[k] = vectors[k].size();
        _acc_idx[k]     = _acc_idx[k - 1] + _stage_sizes[k - 1];
        vec.segment(_acc_idx[k], _stage_sizes[k]) = vectors[k];
    }
}

FlatVec::FlatVec(const FlatVec &other, bool copy)
  : vec(other.vec.size(), 0)
  , _stage_sizes(other._stage_sizes)
  , _acc_idx(other._acc_idx)
{
    if (copy)
    {
        vec = other.vec;
    }
}

inline Eigen::VectorBlock<VecX> FlatVec::stage(const uint k)
{
    return vec.segment(_acc_idx[k], _stage_sizes[k]);
}

inline const VecX FlatVec::stage(const uint k) const
{
    return vec.segment(_acc_idx[k], _stage_sizes[k]);
}

FlatVec FlatVec::operator+(const FlatVec &p) const
{
    FlatVec n  = FlatVec(*this);
    n.vec     += p.vec;

    return n;
}

FlatVec FlatVec::operator-(const FlatVec &p) const
{
    FlatVec n  = FlatVec(*this);
    n.vec     -= p.vec;

    return n;
}

FlatVec FlatVec::operator-() const
{
    FlatVec n(*this);
    n.vec *= -1;

    return n;
}

FlatVec FlatVec::operator/(double val)
{
    FlatVec n  = FlatVec(*this);
    n.vec     /= val;

    return n;
}

FlatVec &FlatVec::operator+=(const FlatVec &p)
{
    vec += p.vec;

    return *this;
}

FlatVec operator*(double val, const FlatVec &p)
{
    FlatVec n  = FlatVec(p);
    n.vec     *= val;

    return p;
}

const std::string to_string(const SolverState &state)
{
    switch (state)
    {
    case SUCCESS:
        return "Success";
    case LOOSE_SUCCESS:
        return "Loose success";
    case PRIMAL_INFEASIBLE:
        return "Primal infeasibility detected";
    case DUAL_INFEASIBLE:
        return "Dual infeasibility detected";
    case MAX_ITER_REACHED:
        return "Max iterations reached";
    case ONGOING:
        return "Ongoing";
    }

    return "";
}

QPData::QPData(const model::Dimensions &dims) : dims(dims) {}
}
}