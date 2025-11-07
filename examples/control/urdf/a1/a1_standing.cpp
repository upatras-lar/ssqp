#include <iostream>
#include <cmath>
#include <vector>

#include "solvers/ssqp.hpp"
#include "solvers/admm_qp.hpp"
#include "model/Model.hpp"
#include "model/stagewise/predefined/Costs.hpp"
#include "pinocchio_wrapper/PinocchioWrapper.hpp"

#define STATE_SIZE 9
#define HORIZON    400
#define dt         0.01
#define P_WGT      10
#define R_WGT      1
#define V_WGT      1e-5
#define U_WGT      1e-5
#define TERM_WGT   1

using namespace ssqp::model::stagewise::predefined;

int main()
{
    Eigen::Vector3d base_pos;
    base_pos << 0, 0, 0.4;

    ssqp::model::Model model("examples/urdf/a1.urdf", HORIZON, dt);

    for (uint k = 0; k < HORIZON; ++k)
    {
        // model.addCost<costs::FrameSquaredPositionErrorCost>(k,
        // model,
        // base_pos, );
    }

    return 0;
}