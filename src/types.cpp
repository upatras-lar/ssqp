#include "types.hpp"

namespace ssqp
{
namespace model
{
Force::Force(const std::string &frame, const uint force_index)
  : frame(frame)
  , force_index(force_index)
{
}
}

void print_matrix_dimensions(const MatX &mat)
{
    std::cout << mat.rows() << 'x' << mat.cols() << std::endl;
}

int sgn(double val)
{
    return (0.0 < val) - (val < 0.0);
}

double softmax_max(const VecX &v, double alpha)
{
    const double m = v.maxCoeff();
    return m + std::log((alpha * (v.array() - m)).exp().sum()) / alpha;
}

std::string format_double(double val, uint precision)
{
    static uint        SCIENTIFIC_CUTOFF = 1e5;
    std::ostringstream oss;
    // No decimal if value is integer
    if (std::abs(val - std::round(val)) <
        std::numeric_limits<double>::epsilon())
    {
        if (std::abs(val) < SCIENTIFIC_CUTOFF)
        {
            oss << static_cast<long long>(std::round(val));
        }
        else
        {
            oss << std::scientific << val;
        }
    }
    // Fixed if not too small to display with given precision
    else if (std::abs(val) <= std::numeric_limits<double>::epsilon() ||
             std::abs(val) >= std::pow(10, -static_cast<int>(precision)) &&
               std::abs(val) < SCIENTIFIC_CUTOFF)
    {
        oss << std::fixed << std::setprecision(precision) << val;
    }
    // Scientific notation otherwise
    else
    {
        oss << std::scientific << std::setprecision(precision) << val;
    }

    return oss.str();
}

namespace solvers
{
FlatVec::FlatVec(const uint total_size, const std::vector<uint> &dims)
  : vec(total_size)
  , _stage_sizes(dims.size())
  , _acc_idx(dims.size())
{
    _max_dim                        = dims[0];
    _stage_sizes[0]                 = dims[0];
    _acc_idx[0]                     = 0;
    vec.segment(0, _stage_sizes[0]) = VecX::Zero(dims[0]);

    for (uint k = 1; k < dims.size(); ++k)
    {
        _max_dim        = std::max(_max_dim, dims[k]);
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
    _max_dim                        = _stage_sizes[0];
    _acc_idx[0]                     = 0;
    vec.segment(0, _stage_sizes[0]) = vectors[0];

    for (uint k = 1; k < vectors.size(); ++k)
    {
        _stage_sizes[k] = vectors[k].size();
        _max_dim        = std::max(_max_dim, _stage_sizes[k]);
        _acc_idx[k]     = _acc_idx[k - 1] + _stage_sizes[k - 1];
        vec.segment(_acc_idx[k], _stage_sizes[k]) = vectors[k];
    }
}

FlatVec::FlatVec(const FlatVec &other, bool copy)
  : vec(copy ? other.vec : VecX::Zero(other.vec.size()))
  , _max_dim(other._max_dim)
  , _stage_sizes(other._stage_sizes)
  , _acc_idx(other._acc_idx)
{
}

Eigen::VectorBlock<VecX> FlatVec::stage(const uint k)
{
    return vec.segment(_acc_idx[k], _stage_sizes[k]);
}

const VecX FlatVec::stage(const uint k) const
{
    return vec.segment(_acc_idx[k], _stage_sizes[k]);
}

const uint FlatVec::stages() const
{
    return _stage_sizes.size();
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

FlatVec FlatVec::operator-(const FlatVec &p) const
{
    assert(stages() == p.stages() && "FlatVec x - y: stage mismatch");
    FlatVec n  = FlatVec(*this);
    n.vec     -= p.vec;
    return n;
}

FlatVec operator*(double val, const FlatVec &p)
{
    FlatVec n  = FlatVec(p);
    n.vec     *= val;

    return n;
}

std::ostream &operator<<(std::ostream &os, const FlatVec &v)
{
    static uint width     = 10;
    static uint precision = 4;
    uint        max_cols  = v.printColumns();

    bool skip = v.stages() > max_cols;

    std::vector<uint> col_widths(std::min(v.stages(), max_cols), 1);
    std::vector<std::vector<std::string>> formatted_values(
      v._max_dim,
      std::vector<std::string>(col_widths.size()));
    uint i = 0;
    for (uint col = 0; i < col_widths.size();
         col      = (!skip || skip && col != max_cols / 2 - 1)
                      ? col + 1
                      : (v.stages() - max_cols / 2))
    {
        for (uint row = 0; row < v._max_dim; ++row)
        {
            if (row < v._stage_sizes[col])
            {
                formatted_values[row][i] =
                  format_double(v.vec(v._acc_idx[col] + row), precision);
                col_widths[i] = std::max(
                  col_widths[i],
                  static_cast<uint>(formatted_values[row][i].length()));
                continue;
            }
            formatted_values[row][i] = "-";
        }
        ++i;
    }

    for (uint row = 0; row < v._max_dim; ++row)
    {
        for (uint col = 0; col < col_widths.size(); ++col)
        {
            os << std::setw(col_widths[col]) << formatted_values[row][col];
            os << "\t";

            if (skip && col == max_cols / 2 - 1)
            {
                os << std::setw(3) << "...";
                os << "\t";
            }
        }
        os << std::endl;
    }

    return os;
}

void FlatVec::setPrintColumns(const uint cols)
{
    _print_cols = cols;
}

uint FlatVec::printColumns() const
{
    return _print_cols > 0 ? _print_cols : stages() + 1;
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

QPData::QPData(const model::Dimensions &dims) : dims(dims), stages(dims.horizon)
{
}
}
}