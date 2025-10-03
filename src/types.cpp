#include "types.hpp"

namespace ssqp
{
namespace solvers
{
const std::string to_string(const SolverState &state)
{
    switch (state)
    {
    case 0:
        return "Max iterations reached";
    case 1:
        return "Success";
    case 10:
        return "Primal infeasibility detected";
    case 11:
        return "Dual infeasibility detected";
    case 100:
        return "Ongoing";
    }

    return "";
}
}
}