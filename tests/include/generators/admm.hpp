#ifndef SSQP_TESTS_GENERATORS_ADMM_HPP
#define SSQP_TESTS_GENERATORS_ADMM_HPP

#include "Test.hpp"
#include "ADMMTest.hpp"

namespace ssqp
{
namespace tests
{
namespace generators
{
std::vector<std::shared_ptr<Test>> get_admm_problems();
}
}
}

#endif