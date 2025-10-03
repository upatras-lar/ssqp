#include "types.hpp"

namespace ssqp
{
namespace solvers
{
const std::string to_string(const SolverState &state)
{
    switch (state)
    {
    case SUCCESS:
        return "Success";
    case LOOSE_SUCCESS:
        return "Loose success";
    case PRIMAL_INFEASIBLE:
        return "Primal infeasibility detected";
    case DUAL_INFEASIBLE:
        return "Dual infeasibility detected";
    case MAX_ITER_REACHED:
        return "Max iterations reached";
    case ONGOING:
        return "Ongoing";
    }

    return "";
}
}
}