#include <nanobind/nanobind.h>

namespace ssqp
{
namespace python
{
namespace nb = nanobind;

void bind_Dynamics(nb::module_ &);

void bind_dynamics(nb::module_ &m)
{
    bind_Dynamics(m);
}
}
}
