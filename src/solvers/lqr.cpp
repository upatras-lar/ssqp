#include "solvers/lqr.hpp"

namespace ssqp
{
namespace solvers
{
namespace lqr
{
Output lqr(const QPData &qp, const admm_qp::Solution &current_sol,
           const FlatVec &rho, const double sigma, bool constrained)
{
    // Initialization
    Output output { FlatVec(current_sol.dx, false),
                    FlatVec(current_sol.du, false),
                    std::vector<MatX>(qp.dims.horizon + 1),
                    std::vector<MatX>(qp.dims.horizon),
                    std::vector<VecX>(qp.dims.horizon + 1),
                    std::vector<VecX>(qp.dims.horizon) };

    // Add constraints to cost
    std::vector<MatX> Qxx(qp.dims.horizon);
    std::vector<MatX> Qxu(qp.dims.horizon);
    std::vector<MatX> Quu(qp.dims.horizon);

    std::vector<VecX> qx(qp.dims.horizon);
    std::vector<VecX> qu(qp.dims.horizon);

    MatX Qt = qp.Qxx_t;
    VecX qt = qp.qx_t;

    if (constrained)
    {
        for (uint k = 0; k < qp.dims.horizon; ++k)
        {
            Qxx[k] = qp.stages[k].Qxx +
                     qp.stages[k].D.transpose() * rho.stage(k).asDiagonal() *
                       qp.stages[k].D +
                     sigma * MatX::Identity(qp.dims.stage_state[k],
                                            qp.dims.stage_state[k]);
            Qxu[k] = qp.stages[k].Qxu + qp.stages[k].D.transpose() *
                                          rho.stage(k).asDiagonal() *
                                          qp.stages[k].E;
            Quu[k] = qp.stages[k].Quu +
                     qp.stages[k].E.transpose() * rho.stage(k).asDiagonal() *
                       qp.stages[k].E +
                     sigma * MatX::Identity(qp.dims.stage_control[k],
                                            qp.dims.stage_control[k]);

            qx[k] = qp.stages[k].qx +
                    qp.stages[k].D.transpose() * current_sol.y.stage(k) -
                    qp.stages[k].D.transpose() * rho.stage(k).asDiagonal() *
                      current_sol.z.stage(k) -
                    sigma * current_sol.dx.stage(k);

            qu[k] = qp.stages[k].qu +
                    qp.stages[k].E.transpose() * current_sol.y.stage(k) -
                    qp.stages[k].E.transpose() * rho.stage(k).asDiagonal() *
                      current_sol.z.stage(k) -
                    sigma * current_sol.du.stage(k);
        }

        Qt +=
          qp.Dt.transpose() * rho.stage(qp.dims.horizon).asDiagonal() * qp.Dt +
          sigma * MatX::Identity(qp.dims.stage_state[qp.dims.horizon],
                                 qp.dims.stage_state[qp.dims.horizon]);
        qt += qp.Dt.transpose() * current_sol.y.stage(qp.dims.horizon) -
              qp.Dt.transpose() * rho.stage(qp.dims.horizon).asDiagonal() *
                current_sol.z.stage(qp.dims.horizon) +
              sigma * current_sol.dx.stage(qp.dims.horizon);
    }
    else
    {
        for (uint k = 0; k < qp.dims.horizon; ++k)
        {
            Qxx[k] = qp.stages[k].Qxx;
            Qxu[k] = qp.stages[k].Qxu;
            Quu[k] = qp.stages[k].Quu;
            qx[k]  = qp.stages[k].qx;
            qu[k]  = qp.stages[k].qu;
        }
    }

    output.P[qp.dims.horizon] = Qt;
    output.p[qp.dims.horizon] = qt;

    // Backward pass
    for (int k = qp.dims.horizon - 1; k >= 0; --k)
    {
        VecX h =
          qu[k] + qp.stages[k].B.transpose() *
                    (output.p[k + 1] + output.P[k + 1] * qp.stages[k].gamma);

        MatX G = Qxu[k].transpose() +
                 qp.stages[k].B.transpose() * output.P[k + 1] * qp.stages[k].A;

        MatX H = Quu[k] +
                 qp.stages[k].B.transpose() * output.P[k + 1] * qp.stages[k].B;

        // TODO: is LLT appropriate?
        if (qp.dims.stage_control[k] > 1)
        {
            // std::cout << "H[" << k << "]:\n" << H << std::endl;
            // std::cout << "G[" << k << "]:\n" << G << std::endl;
            // std::cout << "h[" << k << "]: " << h.transpose() << std::endl;
            Eigen::LLT H_llt = H.llt();
            output.K[k]      = -H_llt.solve(G);
            output.k[k]      = -H_llt.solve(h);

            // std::cout << "K[" << k << "]: " << std::endl;
            // std::cout << output.K[k] << std::endl;

            // std::cout << "k[" << k << "]: " << std::endl;
            // std::cout << output.k[k] << std::endl;
        }
        else
        {
            double coeff = 1. / H(0, 0);
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
    output.x.stage(0) = VecX::Zero(qp.dims.stage_state[0]);

    for (uint k = 0; k < qp.dims.horizon; ++k)
    {
        output.x.stage(k + 1) =
          (qp.stages[k].A + qp.stages[k].B * output.K[k]) * output.x.stage(k) +
          qp.stages[k].B * output.k[k] + qp.stages[k].gamma;
        // std::cout << "output.x.stage(" << k + 1
        //   << "): " << output.x.stage(k + 1).transpose() << std::endl;
        // std::cout << "Because: " << std::endl;
        // std::cout << "First term: "
        //   << ((qp.stages[k].A) * output.x.stage(k)).transpose()
        //   << std::endl;
        // std::cout << "A: " << std::endl << qp.stages[k].A << std::endl;
        // std::cout << "B: " << std::endl << qp.stages[k].B << std::endl;
        // std::cout << "K: " << std::endl << output.K[k] << std::endl;
        // std::cout << "Prev: " << output.x.stage(k).transpose() << std::endl;
        output.u.stage(k) = output.K[k] * output.x.stage(k) + output.k[k];
    }

    return output;
}
}
}
}