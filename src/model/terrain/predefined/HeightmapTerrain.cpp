#include "model/terrain/predefined/HeightmapTerrain.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace ssqp
{
namespace model
{
namespace terrain
{
namespace predefined
{
namespace
{
// Catmull-Rom weights of the 4 neighboring samples at local coordinate t, and
// their first and second derivatives with respect to t
void weights(const double t, Vec4 &w, Vec4 &dw, Vec4 &ddw)
{
    const double t2 = t * t, t3 = t2 * t;
    w << -t3 + 2 * t2 - t, 3 * t3 - 5 * t2 + 2, -3 * t3 + 4 * t2 + t,
      t3 - t2;
    dw << -3 * t2 + 4 * t - 1, 9 * t2 - 10 * t, -9 * t2 + 8 * t + 1,
      3 * t2 - 2 * t;
    ddw << -6 * t + 4, 18 * t - 10, -18 * t + 8, 6 * t - 2;
    w *= 0.5;
    dw *= 0.5;
    ddw *= 0.5;
}

// Splits grid coordinate g into a cell index and local coordinate
// Coordinates outside the grid are clamped to its edge
void locate(const double g, const int n, int &cell, double &t, bool &inside)
{
    const double clamped = std::clamp(g, 0.0, double(n - 1));
    inside               = (clamped == g);
    cell                 = std::min(int(std::floor(clamped)), n - 2);
    t                    = clamped - cell;
}
}

HeightmapTerrain::HeightmapTerrain(const MatX &heights, const double resolution,
                                   const double x0, const double y0)
  : _heights(heights), _res(resolution), _x0(x0), _y0(y0)
{
    assert(heights.rows() >= 2 && heights.cols() >= 2 &&
           "Heightmap must have at least 2 samples along each axis");
    assert(resolution > 0.0 && "Heightmap resolution must be positive");
}

void HeightmapTerrain::set_heights(const MatX &heights, const double x0,
                                   const double y0)
{
    assert(heights.rows() >= 2 && heights.cols() >= 2 &&
           "Heightmap must have at least 2 samples along each axis");
    _heights = heights;
    _x0      = x0;
    _y0      = y0;
}

Terrain::TerrainPoint HeightmapTerrain::at(const double x, const double y) const
{
    const int nx = _heights.rows(), ny = _heights.cols();

    int    ci, cj;
    double tx, ty;
    bool   in_x, in_y;
    locate((x - _x0) / _res, nx, ci, tx, in_x);
    locate((y - _y0) / _res, ny, cj, ty, in_y);

    Vec4 wx, dwx, ddwx, wy, dwy, ddwy;
    weights(tx, wx, dwx, ddwx);
    weights(ty, wy, dwy, ddwy);

    Eigen::Matrix4d patch;
    for (int a = 0; a < 4; ++a)
    {
        for (int b = 0; b < 4; ++b)
        {
            patch(a, b) = _heights(std::clamp(ci - 1 + a, 0, nx - 1),
                                   std::clamp(cj - 1 + b, 0, ny - 1));
        }
    }

    // Derivatives along an axis the query was clamped on are zero
    const double sx = in_x ? 1.0 / _res : 0.0, sy = in_y ? 1.0 / _res : 0.0;

    double h    = wx.dot(patch * wy);
    double dhdx = sx * dwx.dot(patch * wy);
    double dhdy = sy * wx.dot(patch * dwy);
    double hxx  = sx * sx * ddwx.dot(patch * wy);
    double hxy  = sx * sy * dwx.dot(patch * dwy);
    double hyy  = sy * sy * wx.dot(patch * ddwy);

    return TerrainPoint::from_height(h, dhdx, dhdy, hxx, hxy, hyy);
}

nlohmann::ordered_json HeightmapTerrain::to_json() const
{
    // Heights are written as rows indexed by the x grid index
    std::vector<std::vector<double>> rows(_heights.rows());
    for (int i = 0; i < _heights.rows(); ++i)
    {
        rows[i].assign(_heights.row(i).begin(), _heights.row(i).end());
    }

    return { { "type", "heightmap" },
             { "heights", rows },
             { "resolution", _res },
             { "x0", _x0 },
             { "y0", _y0 } };
}

}
}
}
}
