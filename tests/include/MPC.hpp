#ifndef SSQP_TESTS_MPC_HPP
#define SSQP_TESTS_MPC_HPP

#include "Test.hpp"
#include "solvers/ssqp.hpp"

namespace ssqp
{
namespace tests
{
class MPCStepTest : public Test
{
  public:
    MPCStepTest(const std::string                &name,
                const solvers::ssqp::QPSolverType qp_solver_type);

    bool run() override;

  protected:
    const solvers::ssqp::QPSolverType _qp_solver_type;
};
}
}

#endif
