#include <nanobind/nanobind.h>

namespace ssqp
{
namespace python
{
namespace nb = nanobind;

void bind_Custom(nb::module_ &);

void bind_custom(nb::module_ &m)
{
    bind_Custom(m);
}
}
}
