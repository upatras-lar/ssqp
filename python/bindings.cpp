#include <nanobind/nanobind.h>

#include "types.hpp"

namespace ssqp
{
namespace python
{
namespace nb = nanobind;

void bind_model(nb::module_ &);
void bind_solvers(nb::module_ &);

NB_MODULE(ssqpy, m)
{
    m.doc() = "Python bindings for SSQP";

    nb::enum_<ssqp::LogLevel>(m,
                              "LogLevel",
                              R"doc(
        The available log levels
        Each is used for a different class of output, and is independent of the 
        others
        )doc",
                              nb::is_arithmetic())
      .value("Info", ssqp::LogLevel::Info)
      .value("Debug", ssqp::LogLevel::Debug)
      .export_values();

    m.def("setSilentMode",
          &ssqp::set_silent_mode,
          "Disable all levels of logging");

    m.def("setLogLevel",
          static_cast<void (*)(const ssqp::LogLevel &)>(&ssqp::set_log_level),
          nb::arg("log_level"),
          "Add a logging level to the global log mask");

    m.def("setLogLevel",
          static_cast<void (*)(const uint)>(&ssqp::set_log_level),
          nb::arg("log_level_mask"),
          "Add a combination of logging levels to the global log mask");

    m.def("unsetLogLevel",
          static_cast<void (*)(const ssqp::LogLevel &)>(&ssqp::unset_log_level),
          nb::arg("log_level"),
          "Remove a logging level from the global log mask");

    m.def("unsetLogLevel",
          static_cast<void (*)(const uint)>(&ssqp::unset_log_level),
          nb::arg("log_level"),
          "Remove a combination of logging levels from the global log mask");

    auto model   = m.def_submodule("model", R"doc(
        Submodule containing classes related to the SSQP model, a respresentation of the
        system used for optimization.
        )doc");
    auto solvers = m.def_submodule("solvers", R"doc(
        Submodule containing classes related to the SSQP solvers.
        )doc");

    bind_model(model);
    bind_solvers(solvers);
}
}
}
