#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/shared_ptr.h>

#include "param_search/search.hpp"
#include "model/stagewise/StageDynamics.hpp"

namespace ssqp
{
namespace python
{
namespace nb = nanobind;

using namespace model::stagewise;

void bind_search(nb::module_ &m)
{
    nb::class_<param_search::dynamics_params::GradientDescent>(
      m,
      "DynamicsParamGradientDescent")
      .def(nb::init<const std::shared_ptr<StageDynamics> &,
                    const uint,
                    const double,
                    const double>(),
           nb::arg("dynamics"),
           nb::arg("min_data_size") = 100,
           nb::arg("armijo_beta")   = 0.5,
           nb::arg("armijo_c1")     = 1e-4)
      .def("addData",
           &param_search::dynamics_params::GradientDescent::addData,
           nb::arg("x"),
           nb::arg("u"),
           nb::arg("x_next"),
           R"doc(
        Adds a new data point
        Returns: whether the parameters were updated
           )doc")
      .def("clearData",
           &param_search::dynamics_params::GradientDescent::clearData,
           "Clears all previously added data")
      .def("setMinimumDataSize",
           &param_search::dynamics_params::GradientDescent::setMinimumDataSize,
           nb::arg("min_data_size"),
           "Sets the minimum required data size in order to perform an update");

    nb::class_<param_search::dynamics_params::GaussNewton>(
      m,
      "DynamicsParamGaussNewton")
      .def(nb::init<const std::shared_ptr<StageDynamics> &,
                    const uint,
                    const double,
                    const double>(),
           nb::arg("dynamics"),
           nb::arg("window_size") = 100,
           nb::arg("armijo_beta") = 0.5,
           nb::arg("armijo_c1")   = 1e-4)
      .def("addData",
           &param_search::dynamics_params::GaussNewton::addData,
           nb::arg("x"),
           nb::arg("u"),
           nb::arg("x_next"),
           R"doc(
        Adds a new data point
        Returns: whether the parameters were updated
           )doc")
      .def("clearData",
           &param_search::dynamics_params::GaussNewton::clearData,
           "Clears window of all previously added data")
      .def("setWindowSize",
           &param_search::dynamics_params::GaussNewton::setWindowSize,
           nb::arg("window_size"),
           nb::arg("prune_last") = false,
           R"doc(
        Sets the window size to the given value
        Parameters:
         - window_size: The size to resize the window to
         - prune_last: Whether to remove the most recent data points, in case 
           the window currently holds more data than the provided size; if false,
           an algorithm is used to instead prune the most similar data points
           has no effect if the provided size is larger than the data currently
           held
           )doc");

    nb::class_<param_search::HyperParamSearch>(m, "HyperParamSearch")
      .def(nb::init())
      .def(nb::init<const std::vector<solvers::ssqp::Params> &,
                    const std::vector<model::Model> &>())
      .def("addHyperparams", &param_search::HyperParamSearch::addHyperparams)
      .def("addProblem", &param_search::HyperParamSearch::addProblem)
      .def("search", &param_search::HyperParamSearch::search);
}
}
}
