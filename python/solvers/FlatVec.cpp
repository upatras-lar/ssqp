#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/eigen/dense.h>

#include <iostream>

#include "types.hpp"

namespace ssqp
{
namespace python
{
namespace nb = nanobind;

using namespace solvers;
using namespace model;

void bind_FlatVec(nb::module_ &m)
{
    nb::class_<FlatVec>(m, "FlatVec")
      .def(
        "stage",
        static_cast<const VecX (FlatVec::*)(const uint) const>(&FlatVec::stage),
        nb::arg("k"))
      .def("stages", &FlatVec::stages)
      .def("print", [](const FlatVec &self) { std::cout << self; })
      .def("toarray",
           [](const FlatVec &self)
           {
               std::vector<VecX> vec(self.stages());

               for (uint k = 0; k < self.stages(); ++k)
               {
                   vec[k] = self.stage(k);
               }

               return vec;
           });
}
}
}
