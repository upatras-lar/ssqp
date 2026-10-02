#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/eigen/dense.h>

#include "model/terrain/Terrain.hpp"
#include "model/terrain/predefined/FlatTerrain.hpp"
#include "model/terrain/predefined/HeightmapTerrain.hpp"
#include "model/terrain/predefined/SlopeTerrain.hpp"
#include "model/terrain/predefined/StepTerrain.hpp"

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
    nb::class_<Terrain::TerrainPoint>(m, "TerrainPoint")
      .def(nb::init<>())
      .def_rw("height", &Terrain::TerrainPoint::height)
      .def_rw("normal", &Terrain::TerrainPoint::normal)
      .def_rw("t1", &Terrain::TerrainPoint::t1)
      .def_rw("t2", &Terrain::TerrainPoint::t2)
      .def_rw("dhdx", &Terrain::TerrainPoint::dhdx)
      .def_rw("dhdy", &Terrain::TerrainPoint::dhdy)
      .def_rw("d2hdx2", &Terrain::TerrainPoint::d2hdx2)
      .def_rw("d2hdxdy", &Terrain::TerrainPoint::d2hdxdy)
      .def_rw("d2hdy2", &Terrain::TerrainPoint::d2hdy2)
      .def_static("from_height",
                  &Terrain::TerrainPoint::from_height,
                  nb::arg("height"),
                  nb::arg("dhdx"),
                  nb::arg("dhdy"),
                  nb::arg("d2hdx2")  = 0.0,
                  nb::arg("d2hdxdy") = 0.0,
                  nb::arg("d2hdy2")  = 0.0);

    nb::class_<Terrain, PyTerrain>(m, "Terrain")
      .def(nb::init())
      .def("at",
           static_cast<Terrain::TerrainPoint (Terrain::*)(double, double)
                         const>(&Terrain::at),
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

    nb::class_<SlopeTerrain, Terrain>(m, "SlopeTerrain")
      .def(nb::init<double, double, double>(),
           nb::arg("slope_x"),
           nb::arg("slope_y")     = 0.0,
           nb::arg("base_height") = 0.0);

    nb::class_<StepTerrain, Terrain>(m, "StepTerrain")
      .def(nb::init<double, double, double, uint, double>(),
           nb::arg("x_start"),
           nb::arg("step_height"),
           nb::arg("step_length") = 1.0,
           nb::arg("num_steps")   = 1,
           nb::arg("base_height") = 0.0);

    nb::class_<HeightmapTerrain, Terrain>(m, "HeightmapTerrain")
      .def(nb::init<const MatX &, double, double, double>(),
           nb::arg("heights"),
           nb::arg("resolution"),
           nb::arg("x0") = 0.0,
           nb::arg("y0") = 0.0);
}
}
}
