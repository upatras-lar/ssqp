#include "solvers/lqr.hpp"

namespace ssqp
{
namespace solvers
{
namespace lqr
{
Output lqr(const QPData &qp, const admm_qp::Solution &current_sol,
           const std::vector<DMatX> &rho, const double sigma)
{
    // Initialization
    Output output {
        MatX(qp.nx, qp.horizon + 1),       MatX(qp.nu, qp.horizon),
        std::vector<MatX>(qp.horizon + 1), std::vector<MatX>(qp.horizon),
        std::vector<VecX>(qp.horizon + 1), std::vector<VecX>(qp.horizon)
    };

    // Add constraints to cost
    std::vector<MatX> Qxx(qp.horizon);
    std::vector<MatX> Qxu(qp.horizon);
    std::vector<MatX> Quu(qp.horizon);

    std::vector<VecX> qx(qp.horizon);
    std::vector<VecX> qu(qp.horizon);

    MatX Qt = qp.Qxx_t;
    VecX qt = qp.qx_t;

    for (int k = 0; k < qp.horizon; ++k)
    {
        Qxx[k] = qp.stages[k].Qxx +
                 qp.stages[k].D.transpose() * rho[k] * qp.stages[k].D +
                 sigma * MatX::Identity(qp.nx, qp.nx);
        Qxu[k] = qp.stages[k].Qxu +
                 qp.stages[k].D.transpose() * rho[k] * qp.stages[k].E;
        Quu[k] = qp.stages[k].Quu +
                 qp.stages[k].E.transpose() * rho[k] * qp.stages[k].E +
                 sigma * MatX::Identity(qp.nu, qp.nu);

        qx[k] = qp.stages[k].qx +
                qp.stages[k].D.transpose() * current_sol.y.col(k) -
                qp.stages[k].D.transpose() * rho[k] * current_sol.z.col(k) -
                sigma * current_sol.dX.col(k);

        qu[k] = qp.stages[k].qu +
                qp.stages[k].E.transpose() * current_sol.y.col(k) -
                qp.stages[k].E.transpose() * rho[k] * current_sol.z.col(k) -
                sigma * current_sol.dU.col(k);
    }

    Qt += qp.Dt.transpose() * rho[qp.horizon] * qp.Dt +
          sigma * MatX::Identity(qp.nx, qp.nx);
    qt += qp.Dt.transpose() * current_sol.y.col(qp.horizon) -
          qp.Dt.transpose() * rho[qp.horizon] * current_sol.z.col(qp.horizon) +
          sigma * current_sol.dX.col(qp.horizon);

    output.P[qp.horizon] = Qt;
    output.p[qp.horizon] = qt;

    // Backward pass
    for (int k = qp.horizon - 1; k >= 0; --k)
    {

        VecX h =
          qu[k] + qp.stages[k].B.transpose() *
                    (output.p[k + 1] + output.P[k + 1] * qp.stages[k].gamma);

        MatX G = Qxu[k].transpose() +
                 qp.stages[k].B.transpose() * output.P[k + 1] * qp.stages[k].A;

        MatX H = Quu[k] +
                 qp.stages[k].B.transpose() * output.P[k + 1] * qp.stages[k].B;

        // TODO: is LLT appropriate?
        if (qp.nu > 1)
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

        output.P[k] =
          Qxx[k] +
          qp.stages[k].A.transpose() * output.P[k + 1] * qp.stages[k].A -
          output.K[k].transpose() * H * output.K[k];

        output.p[k] =
          qx[k] + output.K[k].transpose() * qu[k] +
          (qp.stages[k].A + qp.stages[k].B * output.K[k]).transpose() *
            (output.p[k + 1] + output.P[k + 1] * qp.stages[k].gamma);
    }

    // Forward pass
    output.x.col(0) = VecX::Zero(qp.nx);

    for (uint k = 0; k < qp.horizon; ++k)
    {
        output.x.col(k + 1) =
          (qp.stages[k].A + qp.stages[k].B * output.K[k]) * output.x.col(k) +
          qp.stages[k].B * output.k[k] + qp.stages[k].gamma;
        output.u.col(k) = output.K[k] * output.x.col(k) + output.k[k];
    }

    return output;
}
}
}
}