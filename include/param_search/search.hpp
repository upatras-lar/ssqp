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
namespace dynamics_params
{
struct DataPoint
{
    VecX x_prev;
    VecX u;
    VecX x_actual;

    DataPoint(const VecX &x_prev, const VecX &u, const VecX &x_actual);
};

class Search
{
  public:
    Search(const std::shared_ptr<model::stagewise::StageDynamics> &dynamics,
           const double armijo_beta = 0.5, const double armijo_c1 = 1e-4);

    // Adds a new data point
    // Returns: whether the parameters were updated
    bool addData(const VecX &x, const VecX &u, const VecX &x_next);

    // Clears all currently held data
    void clearData();

  protected:
    std::shared_ptr<model::stagewise::StageDynamics> _dynamics;
    const uint                                       _state_size;
    const uint                                       _param_size;
    const double                                     _armijo_beta;
    const double                                     _armijo_c1;

    std::vector<DataPoint> _data;

    // Search function to be overridden by subclasses, implementing their
    // specific search method
    virtual bool _search() = 0;

    /** Performs Armijo line search to update the current dynamics parameters,
     * given:
     *  - `gradient`: Gradient of the cost function
     *  - `dTheta_log`: Descent direction
     *  - `initial_cost`: The initial cost
     */
    void _armijo_line_search(const VecX &gradient, const VecX &dTheta_log,
                             const double initial_cost);
};

class GradientDescent : public Search
{
  public:
    GradientDescent(
      const std::shared_ptr<model::stagewise::StageDynamics> &dynamics,
      const uint min_data_size, const double armijo_beta = 0.5,
      const double armijo_c1 = 1e-4);

    // Sets the minimum required data size in order to perform an update
    void setMinimumDataSize(const uint data_size);

  protected:
    bool _search() override;

    uint _min_data_size;
};

class GaussNewton : public Search
{
  public:
    GaussNewton(
      const std::shared_ptr<model::stagewise::StageDynamics> &dynamics,
      const uint window_size = 100, const double armijo_beta = 0.5,
      const double armijo_c1 = 1e-4);

    /** Sets the window size to the given value
     * Parameters:
     *  - window_size: The size to resize the window to
     *  - prune_last: Whether to remove the most recent data points, in case 
     *    the window currently holds more data than the provided size; if false,
     *    an algorithm is used to instead prune the most similar data points
     *    has no effect if the provided size is larger than the data currently
     *    held
     */
    void setWindowSize(const uint window_size, bool prune_last = false);

  protected:
    bool _search() override;

    uint _window_size;

    VecX _r;
    MatX _J;

    double _jacobianSimilarity(const MatX &J1, const MatX &J2);
};

}

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
