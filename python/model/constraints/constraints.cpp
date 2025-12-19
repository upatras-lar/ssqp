#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bind_Constraints(nb::module_ &);

void bind_constraints(nb::module_ &m)
{
    bind_Constraints(m);
}