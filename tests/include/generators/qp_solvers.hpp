#ifndef SSQP_TESTS_GENERATORS_ADMM_HPP
#define SSQP_TESTS_GENERATORS_ADMM_HPP

#include "Test.hpp"
#include "QPSolverTests.hpp"
#include "ADMMInfeasibility.hpp"

namespace ssqp
{
namespace tests
{
namespace generators
{
std::vector<std::shared_ptr<Test>> get_admm_problems();
std::shared_ptr<Test>              get_admm_infeasibility();

std::vector<std::shared_ptr<Test>> get_hpipm_problems();
}
}
}

#endif