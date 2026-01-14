#include <nanobind/nanobind.h>

namespace ssqp
{
namespace python
{
namespace nb = nanobind;

void bind_ssqp(nb::module_ &m);
void bind_MPC(nb::module_ &m);
void bind_FlatVec(nb::module_ &m);

void bind_solvers(nb::module_ &m)
{
    bind_ssqp(m);
    bind_MPC(m);
    bind_FlatVec(m);
}
}
}