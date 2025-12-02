#ifndef SSQP_TESTS_FDGRAD_HPP
#define SSQP_TESTS_FDGRAD_HPP

#include "tests/Test.hpp"

#include "model/stagewise/StageCost.hpp"
#include "model/stagewise/StageDynamics.hpp"
#include "model/stagewise/StageConstraints.hpp"

namespace ssqp
{
namespace tests
{
/* Finite-differences gradient test for Cost functions
 * Tests whether a cost function's gradient is valid by comparing its output
 * with that of the finite-differences gradient on random inputs
 */
class FDGradCost : public Test
{
  public:
    FDGradCost(const std::shared_ptr<model::stagewise::StageCost> &cost,
               const uint ndx, const uint nu, const uint ninputs = 100,
               const double h = 1e-7);

    bool run() override;

  protected:
    const std::shared_ptr<model::stagewise::StageCost> _cost;
    const uint                                         _ndx, _nu;
    const uint                                         _ninputs;
    const double                                       _h;

    void _fd_grad(const VecX &x, const VecX &u, VecX &dx, VecX &du) const;
};

class FDGradDynamics : public Test
{
  public:
    FDGradDynamics(
      const std::shared_ptr<model::stagewise::StageDynamics> &dynamics,
      const uint ninputs = 100, const double h = 1e-7);

    bool run() override;

  protected:
    const std::shared_ptr<model::stagewise::StageDynamics> _dynamics;
    const uint                                             _ndx, _nu;
    const uint                                             _ninputs;
    const double                                           _h;

    void _fd_grad(const VecX &x, const VecX &u, MatX &dX, MatX &dU) const;
};

}
}

#endif