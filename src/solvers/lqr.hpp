#include <vector>

#include <Eigen/Dense>

#include "types.hpp"

namespace ssqp
{
namespace solvers
{
namespace lqr
{
// TODO: change to vector of structs
struct Input
{
    int horizon, nx, nu;

    std::vector<MatX> A, B;
    std::vector<VecX> gamma;
    std::vector<MatX> Qxx, Qxu, Quu;
    std::vector<VecX> qx, qu;

    std::vector<MatX> D, E;

    MatX Qt;
    VecX qt;
    MatX Dt;

    MatX x, u, z, y;

    Input(const int horizon, const int nx, const int nu)
      : horizon(horizon)
      , nx(nx)
      , nu(nu)
      , A(std::vector<MatX>(horizon))
      , B(std::vector<MatX>(horizon))
      , gamma(std::vector<VecX>(horizon))
      , Qxx((std::vector<MatX>(horizon)))
      , Qxu(std::vector<MatX>(horizon))
      , Quu((std::vector<MatX>(horizon)))
      , qx(std::vector<VecX>(horizon))
      , qu(std::vector<VecX>(horizon))
      , D((std::vector<MatX>(horizon)))
      , E((std::vector<MatX>(horizon)))
    {
    }
};

struct Output
{
    MatX              x, u;
    std::vector<MatX> P, K;
    std::vector<VecX> p, k;
};

Output lqr(const Input &input, const double rho = 0.1, const double alpha = 1.6,
           const double sigma = 10e-6)
{
    // Initialization
    Output output {
        MatX(input.nx, input.horizon + 1),    MatX(input.nu, input.horizon),
        std::vector<MatX>(input.horizon + 1), std::vector<MatX>(input.horizon),
        std::vector<VecX>(input.horizon + 1), std::vector<VecX>(input.horizon)
    };

    // Add constraints to cost
    std::vector<MatX> Qxx = input.Qxx;
    std::vector<MatX> Qxu = input.Qxu;
    std::vector<MatX> Quu = input.Quu;

    std::vector<VecX> qx = input.qx;
    std::vector<VecX> qu = input.qu;

    MatX Qt;
    VecX qt;

    for (int k = 0; k < input.horizon; ++k)
    {
        Qxx[k] += rho / 2 * input.D[k].transpose() * input.D[k] +
                  sigma / 2 * MatX::Identity(input.nx, input.nx);
        Qxu[k] = rho / 2 * input.D[k].transpose() * input.E[k];
        Quu[k] = rho / 2 * input.E[k].transpose() * input.E[k] +
                 sigma / 2 * MatX::Identity(input.nu, input.nu);

        VecX z  = -input.z.col(k) + 1 / rho * input.y.col(k);
        qx[k]  += rho * input.D[k].transpose() * z - sigma * input.x.col(k);
        qu[k]  += rho * input.E[k].transpose() * z - sigma * input.u.col(k);
    }

    Qt = input.Qt + rho / 2 * input.Dt.transpose() * input.Dt;
    qt = input.qt +
         rho * input.Dt.transpose() *
           (-input.z.col(input.horizon) + 1 / rho * input.y.col(input.horizon));

    output.P[input.horizon] = input.Qt;
    output.p[input.horizon] = input.qt;

    VecX r = VecX::Zero(input.nu);

    // Backward pass
    for (int k = input.horizon - 1; k >= 0; --k)
    {
        VecX h = r + input.B[k].transpose() *
                       (output.p[k + 1] + output.P[k + 1] * input.gamma[k]);

        MatX G = Qxu[k].transpose() +
                 input.B[k].transpose() * output.P[k + 1] * input.A[k];

        MatX H = Quu[k] + input.B[k].transpose() * output.P[k + 1] * input.B[k];

        // TODO: is LLT appropriate?
        if (input.nu > 1)
        {
            Eigen::LLT H_llt = H.llt();
            output.K[k]      = -H_llt.solve(G);
            output.k[k]      = -H_llt.solve(h);
        }
        else
        {
            double coeff = 1 / H(0, 0);
            output.K[k]  = -coeff * G;
            output.k[k]  = -coeff * h;
        }

        output.P[k] = Qxx[k] +
                      input.A[k].transpose() * output.P[k + 1] * input.A[k] -
                      output.K[k].transpose() * H * output.K[k];

        output.p[k] = qx[k] + output.K[k].transpose() * qu[k] +
                      (input.A[k] + input.B[k] * output.K[k]).transpose() *
                        (output.p[k + 1] + output.P[k + 1] * input.gamma[k]);
    }

    // Forward pass
    output.x.col(0) = VecX::Zero(input.nx);

    for (uint k = 0; k < input.horizon; ++k)
    {
        output.x.col(k + 1) =
          (input.A[k] + input.B[k] * output.K[k]) * output.x.col(k) +
          input.B[k] * output.k[k] + input.gamma[k];
        output.u.col(k) = output.K[k] * output.x.col(k) + output.k[k];
    }

    return output;
}
}
}
}