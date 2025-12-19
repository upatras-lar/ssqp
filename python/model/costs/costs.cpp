#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bind_Costs(nb::module_ &);

void bind_costs(nb::module_ &m)
{
    bind_Costs(m);
}