#include <nanobind/nanobind.h>

namespace ssqp
{
namespace python
{
namespace nb = nanobind;

void bind_ssqp(nb::module_ &m);
void bind_MPC(nb::module_ &m);
void bind_FlatVec(nb::module_ &m);
void bind_search(nb::module_ &m);

void bind_solvers(nb::module_ &m)
{
    bind_ssqp(m);
    bind_MPC(m);
    bind_FlatVec(m);

    auto search = m.def_submodule("search", R"doc(
        Submodule containing classes related to various types of parameter
        searching methods.
        )doc");
    bind_search(search);
}
}
}