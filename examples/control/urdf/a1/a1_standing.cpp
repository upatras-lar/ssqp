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

int main()
{
    return 0;
}