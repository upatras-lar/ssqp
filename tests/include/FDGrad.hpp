#ifndef SSQP_TESTS_FDGRAD_HPP
#define SSQP_TESTS_FDGRAD_HPP

#include "Test.hpp"

#include "model/Model.hpp"
#include "model/stagewise/StageCost.hpp"
#include "model/stagewise/StageDynamics.hpp"
#include "model/stagewise/StageConstraints.hpp"
#include "solvers/ssqp.hpp"

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
    // TODO: Some fails with 1e-5, 1e-6
    FDGradCost(const std::string                  &name,
               const std::shared_ptr<model::Model> model, const uint stage,
               const uint ninputs = 100, const double h = 1e-4);

    bool run() override;

  protected:
    const std::shared_ptr<model::Model>                _model;
    const std::shared_ptr<model::stagewise::StageCost> _cost;
    const uint                                         _stage;
    const uint                                         _ndx, _nu;
    const uint                                         _ninputs;
    const double                                       _h;

    void _fd_grad(const VecX &x, const VecX &u, VecX &dx, VecX &du) const;
};

/* Finite-differences gradient test for Dynamics functions
 * Tests whether a dynamics function's gradient is valid by comparing its output
 * with that of the finite-differences gradient on random inputs
 */
class FDGradDynamics : public Test
{
  public:
    FDGradDynamics(const std::string                   &name,
                   const std::shared_ptr<model::Model> &model, const uint stage,
                   const uint ninputs = 100, const double h = 1e-6);

    bool run() override;

  protected:
    const std::shared_ptr<model::Model>                    _model;
    const std::shared_ptr<model::stagewise::StageDynamics> _dynamics,
      _next_dynamics;
    const uint   _stage;
    const uint   _ninputs;
    const double _h;

    void _fd_grad(const VecX &x, const VecX &u, MatX &dX, MatX &dU) const;
};

class FDGradDynamicsParameters : public Test
{
  public:
    FDGradDynamicsParameters(const std::string                   &name,
                             const std::shared_ptr<model::Model> &model,
                             const uint stage, const uint ninputs = 100,
                             const double h = 1e-6, const uint seed = 42);

    bool run() override;

  protected:
    const std::shared_ptr<model::Model>                    _model;
    const std::shared_ptr<model::stagewise::StageDynamics> _dynamics,
      _next_dynamics;
    const uint   _stage;
    const uint   _ninputs;
    const double _h;
    const uint   _seed;

    MatX _fd_grad(const VecX &x, const VecX &u);
};

/* Finite-differences gradient test for Constraints functions
 * Tests whether a constraints function's gradient is valid by comparing its 
 * output with that of the finite-differences gradient on random inputs
 */
class FDGradConstraints : public Test
{
  public:
    FDGradConstraints(const std::string                   &name,
                      const std::shared_ptr<model::Model> &model,
                      const uint stage, const uint ninputs = 100,
                      const double h = 1e-6);

    bool run() override;

  protected:
    const std::shared_ptr<model::Model>                       _model;
    const std::shared_ptr<model::stagewise::StageConstraints> _constraints;
    const std::shared_ptr<model::stagewise::StageDynamics>    _dynamics;
    const uint                                                _stage;
    const uint                                                _ndc, _ndx, _nu;
    const uint                                                _ninputs;
    const double                                              _h;

    void _fd_grad(const VecX &x, const VecX &u, MatX &dX, MatX &dU) const;
};

/* Finite-differences gradient test for total trajectory cost function _ssqp::_lf
 * Tests whether the function's gradient is valid by comparing its output with
 * that of the finite-differences gradient on random inputs
 */
class FDGradFullTrajectoryCost : public Test
{
  public:
    FDGradFullTrajectoryCost(const std::string                   &name,
                             const std::shared_ptr<model::Model> &model,
                             const uint ninputs = 100, const double h = 1e-4);

    bool run() override;

  protected:
    const solvers::ssqp                 _ssqp;
    const std::shared_ptr<model::Model> _model;
    const uint                          _ninputs;
    const double                        _h;

    void _fd_grad(const solvers::FlatVec &x, const solvers::FlatVec &u,
                  solvers::FlatVec &dx, solvers::FlatVec &du);
};

/* Finite-differences gradient test for total trajectory stage gap function 
 * _ssqp::_gammaf
 * Tests whether the function's gradient is valid by comparing its output with
 * that of the finite-differences gradient on random inputs
 */
class FDGradFullTrajectoryStageGap : public Test
{
  public:
    FDGradFullTrajectoryStageGap(const std::string                   &name,
                                 const std::shared_ptr<model::Model> &model,
                                 const uint ninputs = 100,
                                 const double h     = 1e-4);

    bool run() override;

  protected:
    const solvers::ssqp                 _ssqp;
    const std::shared_ptr<model::Model> _model;
    const uint                          _ninputs;
    const double                        _h;

    void _fd_grad(const solvers::FlatVec &x, const solvers::FlatVec &u,
                  solvers::FlatVec &dx, solvers::FlatVec &du);
};

/* Finite-differences gradient test for total trajectory constraint violation
 * function _ssqp::_cf
 * Tests whether the function's gradient is valid by comparing its output with
 * that of the finite-differences gradient on random inputs
 */
class FDGradFullTrajectoryConstraintViolation : public Test
{
  public:
    FDGradFullTrajectoryConstraintViolation(
      const std::string &name, const std::shared_ptr<model::Model> &model,
      const uint ninputs = 100, const double h = 1e-4);

    bool run() override;

  protected:
    const solvers::ssqp                 _ssqp;
    const std::shared_ptr<model::Model> _model;
    const uint                          _ninputs;
    const double                        _h;

    void _fd_grad(const solvers::FlatVec &x, const solvers::FlatVec &u,
                  solvers::FlatVec &dx, solvers::FlatVec &du);
};

}
}

#endif