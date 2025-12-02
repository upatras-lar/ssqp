#ifndef SSQP_TESTS_TEST_HPP
#define SSQP_TESTS_TEST_HPP

#include <random>
#include <sstream>

#include "types.hpp"

namespace ssqp
{
namespace tests
{
class Test
{
  public:
    virtual bool run() = 0;

    std::vector<std::string> what() const;

  protected:
    std::vector<std::string> _failed_logs;
};

inline VecX random_vector(const uint size, const double start = -1e+6,
                          const double end = 1e+6);
}
}

#endif