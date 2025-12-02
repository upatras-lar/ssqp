#include "tests/Test.hpp"

namespace ssqp
{
namespace tests
{
VecX random_vector(const uint size, const double start = -1e+6,
                   const double end = 1e+6)
{
    return start + (end - start) * 0.5 * (VecX::Random(size).array() + 1.0);
}

std::vector<std::string> Test::what() const
{
    return _failed_logs;
}
}
}