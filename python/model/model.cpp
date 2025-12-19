#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bind_SystemModel(nb::module_ &);

void bind_model(nb::module_ &m)
{
    bind_SystemModel(m);
}