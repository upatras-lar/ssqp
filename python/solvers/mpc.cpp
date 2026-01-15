#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>

#include "solvers/mpc.hpp"
#include "model/Model.hpp"

namespace ssqp
{
namespace python
{
namespace nb = nanobind;

using namespace solvers;
using namespace model;

void bind_MPC(nb::module_ &m)
{
    nb::class_<MPC>(m, "MPC")
      .def(
        "__init__",
        [](MPC *self, Model &model, nb::kwargs kwargs)
        {
            MPC::Params params;

            if (kwargs.contains("sqp_iters"))
            {
                params.sqp_iters = nb::cast<uint>(kwargs["sqp_iters"]);
            }
            if (kwargs.contains("admm_iters"))
            {
                params.admm_iters = nb::cast<uint>(kwargs["admm_iters"]);
            }

            new (self) MPC(model, params);
        },
        nb::keep_alive<1, 2>())
      .def("step", &MPC::step, nb::arg("current_state"));
}

}
}