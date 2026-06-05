#ifndef SSQP_PARAM_SEARCH_SEARCH_HPP
#define SSQP_PARAM_SEARCH_SEARCH_HPP

#include <omp.h>

#include <vector>

#include "solvers/ssqp.hpp"
#include "solvers/admm_qp.hpp"
#include "model/Model.hpp"
#include "model/stagewise/StageDynamics.hpp"

namespace ssqp
{
namespace param_search
{
class DynamicsParamGaussNewton
{
  public:
    DynamicsParamGaussNewton(
      const std::shared_ptr<model::stagewise::StageDynamics> &dynamics,
      const uint window_size = 100, const double armijo_beta = 0.5,
      const double armijo_c1 = 1e-4);

    void addData(const VecX &x, const VecX &u, const VecX &x_next);

  protected:
    std::shared_ptr<model::stagewise::StageDynamics> _dynamics;

    struct DataPoint
    {
        VecX x_prev;
        VecX u;
        VecX x_actual;

        DataPoint(const VecX &x_prev, const VecX &u, const VecX &x_actual);
    };

    const uint             _state_size;
    const uint             _param_size;
    const uint             _window_size;
    std::vector<DataPoint> _window;

    const double _armijo_beta;
    const double _armijo_c1;

    VecX _r;
    MatX _J;

    double _jacobianSimilarity(const MatX &J1, const MatX &J2);
};

class HyperParamSearch
{
  public:
    HyperParamSearch() = default;
    HyperParamSearch(const std::vector<solvers::ssqp::Params> &hyperparams,
                     const std::vector<model::Model>          &problems);

    void addHyperparams(const solvers::ssqp::Params &sqp_params);

    void addProblem(const model::Model &model);

    const solvers::ssqp::Params search() const;

  protected:
    std::vector<solvers::ssqp::Params> _hyperparams;
    std::vector<model::Model>          _problems;
};
}
}

#endif
