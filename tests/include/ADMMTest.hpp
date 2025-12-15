#ifndef SSQP_TESTS_ADMM_HPP
#define SSQP_TESTS_ADMM_HPP

#include <fstream>

#include "Test.hpp"

#include "types.hpp"
#include "solvers/admm_qp.hpp"

#include "external/nlohmann/json.hpp"

namespace ssqp
{
namespace tests
{
class ADMMTest : public Test
{
  public:
    ADMMTest(const std::string &name, const std::string &data_file,
             const double tolerance = 1e-3);

    bool run() override;

  protected:
    ADMMTest(const std::string                                            &name,
             const std::pair<solvers::QPData, solvers::admm_qp::Solution> &pair,
             const double tolerance);
    static std::pair<solvers::QPData, solvers::admm_qp::Solution>
    _loadQPDataFromFile(const std::string &data_file);

    const solvers::QPData      _qp;
    solvers::admm_qp::Solution _solution;
    const double               _tol;
};
}
}

#endif