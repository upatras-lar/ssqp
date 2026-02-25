#ifndef SSQP_PARAM_SEARCH_SEARCH_HPP
#define SSQP_PARAM_SEARCH_SEARCH_HPP

#include <omp.h>

#include <vector>

#include "solvers/ssqp.hpp"
#include "solvers/admm_qp.hpp"
#include "model/Model.hpp"

namespace ssqp
{
namespace param_search
{
class ParamSearch
{
  public:
    ParamSearch() = default;
    ParamSearch(const std::vector<solvers::ssqp::Params> &hyperparams,
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
