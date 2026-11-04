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
    nb::class_<Terrain::TerrainPoint>(
      m,
      "TerrainPoint",
      "A point on the terrain surface: its height, surface frame and height "
      "derivatives.")
      .def(nb::init<>())
      .def_rw("height",
              &Terrain::TerrainPoint::height,
              "Height of the terrain.")
      .def_rw("normal",
              &Terrain::TerrainPoint::normal,
              "Unit normal of the terrain surface.")
      .def_rw("t1",
              &Terrain::TerrainPoint::t1,
              "First unit tangent of the terrain surface, along x.")
      .def_rw("t2",
              &Terrain::TerrainPoint::t2,
              "Second unit tangent of the terrain surface, the normal crossed "
              "with t1.")
      .def_rw("dhdx",
              &Terrain::TerrainPoint::dhdx,
              "Slope of the height along x.")
      .def_rw("dhdy",
              &Terrain::TerrainPoint::dhdy,
              "Slope of the height along y.")
      .def_rw("d2hdx2",
              &Terrain::TerrainPoint::d2hdx2,
              "Second derivative of the height along x.")
      .def_rw("d2hdxdy",
              &Terrain::TerrainPoint::d2hdxdy,
              "Mixed second derivative of the height along x and y.")
      .def_rw("d2hdy2",
              &Terrain::TerrainPoint::d2hdy2,
              "Second derivative of the height along y.")
      .def_static(
        "from_height",
        &Terrain::TerrainPoint::from_height,
        nb::arg("height"),
        nb::arg("dhdx"),
        nb::arg("dhdy"),
        nb::arg("d2hdx2")  = 0.0,
        nb::arg("d2hdxdy") = 0.0,
        nb::arg("d2hdy2")  = 0.0,
        "Builds a point from the height and its derivatives, using the frame "
        "convention required by contact constraints: normal = (-dhdx, -dhdy, "
        "1) normalized, t1 = (1, 0, dhdx) normalized, t2 = normal x t1.");

    nb::class_<Terrain, PyTerrain>(m,
                                   "Terrain",
                                   "Base class for terrains. Subclasses "
                                   "implement `at`, and optionally `distance`.")
      .def(nb::init())
      .def(
        "at",
        static_cast<Terrain::TerrainPoint (Terrain::*)(double, double) const>(
          &Terrain::at),
        nb::arg("x"),
        nb::arg("y"),
        "Returns the terrain point at the horizontal position (x, y).")
      .def("distance",
           static_cast<double (Terrain::*)(double, double, double) const>(
             &Terrain::distance),
           nb::arg("x"),
           nb::arg("y"),
           nb::arg("z"),
           "Returns the distance of a point above the terrain, by default its "
           "height above the terrain at its horizontal position.");

    nb::class_<FlatTerrain, Terrain>(m,
                                     "FlatTerrain",
                                     "Flat terrain at a constant height.")
      .def(nb::init<double>(), nb::arg("base_height") = 0.0);

    nb::class_<SlopeTerrain, Terrain>(m, "SlopeTerrain", R"doc(
        Slope terrain
        Height is $h = h_0 + s_x x + s_y y$, where $s_x$ and $s_y$ are the slopes
        along the x and y axes (rise over run)    
        )doc")
      .def(nb::init<double, double, double>(),
           nb::arg("slope_x"),
           nb::arg("slope_y")     = 0.0,
           nb::arg("base_height") = 0.0);

    nb::class_<StepTerrain, Terrain>(m, "StepTerrain", R"doc(
        Piecewise flat terrain with a flight of steps along the x axis

        Height is `base_height` for $x < x_{start}$, then rises by `step_height` at
        `x_start` and every `step_length` after that, for `num_steps` steps in total

        The terrain stays at the top step height afterwards; with the default of a
        single step, `step_length` has no effect
        )doc")
      .def(nb::init<double, double, double, uint, double>(),
           nb::arg("x_start"),
           nb::arg("step_height"),
           nb::arg("step_length") = 1.0,
           nb::arg("num_steps")   = 1,
           nb::arg("base_height") = 0.0);

    nb::class_<HeightmapTerrain, Terrain>(m, "HeightmapTerrain", R"doc(
        Smooth terrain interpolated from a uniform grid of heights
        `heights(i, j)` is the height at $(x_0 + i r, y_0 + j r)$, where $r$ is the
        grid resolution

        Heights are interpolated with a bicubic Catmull-Rom spline, which passes 
        through every grid height and has continuous height and slope

        The height second derivatives are piecewise continuous

        Outside the grid, the terrain extends flat from its closest edge value
        )doc")
      .def(nb::init<const MatX &, double, double, double>(),
           nb::arg("heights"),
           nb::arg("resolution"),
           nb::arg("x0") = 0.0,
           nb::arg("y0") = 0.0)
      .def("set_heights",
           &HeightmapTerrain::set_heights,
           nb::arg("heights"),
           nb::arg("x0") = 0.0,
           nb::arg("y0") = 0.0,
           "Replaces the grid of heights and its origin, keeping the "
           "resolution. The grid may change size. Models using this terrain "
           "see the new heights from their next evaluation, so this must not "
           "be called while a solver is using such a model.");
}
}
}
