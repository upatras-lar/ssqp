#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/eigen/dense.h>

#include "model/terrain/Terrain.hpp"
#include "model/terrain/predefined/FlatTerrain.hpp"

namespace ssqp
{
namespace python
{
namespace nb = nanobind;

using namespace model::terrain;
using namespace model::terrain::predefined;

struct __attribute__((visibility("default"))) PyTerrain : Terrain
{
    NB_TRAMPOLINE(Terrain, 2);

    TerrainPoint at(const double x, const double y) const override
    {
        NB_OVERRIDE_PURE(at, x, y);
    }

    double distance(const double x, const double y,
                    const double z) const override
    {
        NB_OVERRIDE(distance, x, y, z);
    }
};

void bind_Terrain(nb::module_ &m)
{
    nb::class_<Terrain, PyTerrain>(m, "Terrain")
      .def(nb::init())
      .def(
        "at",
        [](const Terrain &self, const double x, const double y)
        {
            Terrain::TerrainPoint p = self.at(x, y);
            return nb::make_tuple(p.height,
                                  p.normal,
                                  p.t1,
                                  p.t2,
                                  p.dhdx,
                                  p.dhdy);
        },
        nb::arg("x"),
        nb::arg("y"))
      .def("distance",
           static_cast<double (Terrain::*)(double, double, double) const>(
             &Terrain::distance),
           nb::arg("x"),
           nb::arg("y"),
           nb::arg("z"));

    nb::class_<FlatTerrain, Terrain>(m, "FlatTerrain")
      .def(nb::init<double>(), nb::arg("base_height") = 0.0);
}
}
}
