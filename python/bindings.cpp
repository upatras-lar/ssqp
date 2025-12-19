#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bind_model(nb::module_ &);

NB_MODULE(pyssqp, m)
{
    m.doc() = "Python bindings for SSQP";

    auto model   = m.def_submodule("model");
    auto solvers = m.def_submodule("solvers");

    bind_model(model);
}