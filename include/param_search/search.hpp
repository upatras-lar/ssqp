#ifndef SSQP_PARAM_SEARCH_SEARCH_HPP
#define SSQP_PARAM_SEARCH_SEARCH_HPP

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
    ParamSearch(const std::vector<
                  std::pair<solvers::ssqp::Params, solvers::admm_qp::Params>>
                                                &hyperparams,
                const std::vector<model::Model> &problems);

    void addHyperparams(const solvers::ssqp::Params    &sqp_params,
                        const solvers::admm_qp::Params &admm_params);

    void addProblem(const model::Model &model);

    const std::tuple<solvers::ssqp::Params, solvers::admm_qp::Params,
                     model::Model>
    search() const;

  protected:
    std::vector<std::pair<solvers::ssqp::Params, solvers::admm_qp::Params>>
                              _hyperparams;
    std::vector<model::Model> _problems;
};
}
}

#endif
