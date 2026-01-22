#ifndef SSQP_TESTS_ADMM_INFEASIBILITY_HPP
#define SSQP_TESTS_ADMM_INFEASIBILITY_HPP

#include <fstream>

#include "Test.hpp"

#include "types.hpp"
#include "model/Model.hpp"
#include "solvers/ssqp.hpp"
#include "model/stagewise/predefined/Costs.hpp"
#include "model/stagewise/predefined/Constraints.hpp"

namespace ssqp
{
namespace tests
{
class ADMMInfeasibility : public Test
{
  public:
    ADMMInfeasibility(const std::string &name);

    bool run() override;

  protected:
    model::Model _model;
};
}
}

#endif