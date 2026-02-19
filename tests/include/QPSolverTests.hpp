#ifndef SSQP_TESTS_ADMM_HPP
#define SSQP_TESTS_ADMM_HPP

#include <fstream>

#include "Test.hpp"

#include "types.hpp"
#include "solvers/qp.hpp"
#include "solvers/admm_qp.hpp"
#include "solvers/hpipm_qp.hpp"

#include "nlohmann/json.hpp"

namespace ssqp
{
namespace tests
{
class QPSolverTest : public Test
{
  public:
    QPSolverTest(const std::string &name, const std::string &data_file,
                 const double tolerance);

    virtual bool run() = 0;

  protected:
    QPSolverTest(
      const std::string                                             &name,
      const std::pair<solvers::QPData, solvers::QPSolver::Solution> &pair,
      const double                                                   tolerance);
    static std::pair<solvers::QPData, solvers::QPSolver::Solution>
    _loadQPDataFromFile(const std::string &data_file);

    const solvers::QPData       _qp;
    solvers::QPSolver::Solution _solution;
    const double                _tol;
};

class ADMMTest : public QPSolverTest
{
  public:
    ADMMTest(const std::string &name, const std::string &data_file,
             const double tolerance = 1e-3);

    bool run() override;
};

class HPIPMTest : public QPSolverTest
{
  public:
    HPIPMTest(const std::string &name, const std::string &data_file,
              const double tolerance = 1e-3);

    bool run() override;
};
}
}

#endif