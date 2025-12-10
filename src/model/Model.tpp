#include "model/Model.hpp"

namespace ssqp
{
namespace model
{
template<typename L, typename... Args>
void Model::addCost(const uint k, Args... args)
{
    static_assert(std::is_base_of_v<stagewise::StageCost, L>, "Invalid cost");

    std::shared_ptr<stagewise::StageCost> cost = std::make_shared<L>(args...);

    addCost(k, cost);
}

template<typename C, typename... Args>
void Model::addConstraints(const uint k, Args... args)
{
    static_assert(std::is_base_of_v<stagewise::StageConstraints, C>,
                  "Invalid constraint");

    std::shared_ptr<stagewise::StageConstraints> constraints =
      std::make_shared<C>(args...);

    std::vector<Force> forces(constraints->required_forces());
    for (uint i = 0; i < constraints->required_forces(); ++i)
    {
        forces[i] = addForce(k, constraints->force_frame(i));
    }

    constraints->set_forces(forces);

    addConstraints(k, constraints);
}

template<bool WithForces, IntegrationScheme IS>
std::shared_ptr<stagewise::StageDynamics> Model::_makeStageDynamics(
  const _StageDynamicsContext &ctx) const
{
    auto right_plus = [this](const VecX &x, const VecX &dx)
    {
        VecX out(_dims.nq[0] + _dims.nv[0]);

        out.head(_dims.nq[0]) =
          pinocchio_wrapper::integrate(_impl,
                                       x.head(_dims.nq[0]),
                                       dx.head(_dims.nv[0]));
        out.tail(_dims.nv[0]) = x.tail(_dims.nv[0]) + dx.tail(_dims.nv[0]);

        return out;
    };

    auto right_minus = [this](const VecX &x, const VecX &y)
    {
        VecX out(2 * _dims.nv[0]);

        out.head(_dims.nv[0]) =
          pinocchio_wrapper::difference(_impl,
                                        y.head(_dims.nq[0]),
                                        x.head(_dims.nq[0]));
        out.tail(_dims.nv[0]) = x.tail(_dims.nv[0]) - y.tail(_dims.nv[0]);

        return out;
    };

    ManifoldOperators<decltype(right_plus), decltype(right_minus)> ops = {
        right_plus,
        right_minus
    };

    auto dyn_eval = [this, ctx](const VecX &x, const VecX &u)
    {
        VecX tau = ctx.S * u.head(_dims.nv[0]);
        VecX ddq;

        if constexpr (WithForces)
        {
            pinocchio_wrapper::computeJointJacobians(_impl,
                                                     x.head(_dims.nq[0]));
            std::vector<Vec6> joint_forces(ctx.total_joints, Vec6::Zero());

            for (uint i = 0; i < ctx.nf; ++i)
            {
                Vec3 f = u.segment(_dims.nv[0] + 3 * i, 3);

                uint joint_id;
                Vec6 f_res = pinocchio_wrapper::framejMfAct(_impl,
                                                            ctx.forces[i].frame,
                                                            f,
                                                            joint_id);

                joint_forces[joint_id] += f_res;
            }

            ddq = pinocchio_wrapper::aba(_impl,
                                         x.head(_dims.nq[0]),
                                         x.tail(_dims.nv[0]),
                                         tau,
                                         joint_forces);
        }
        else
        {
            ddq = pinocchio_wrapper::aba(_impl,
                                         x.head(_dims.nq[0]),
                                         x.tail(_dims.nv[0]),
                                         tau);
        }

        VecX out(_dims.nq[0] + _dims.nv[0]);

        if constexpr (IS == IntegrationScheme::ExplicitEuler)
        {
            out.head(_dims.nq[0]) =
              pinocchio_wrapper::integrate(_impl,
                                           x.head(_dims.nq[0]),
                                           x.tail(_dims.nv[0]) * _dt);
            out.tail(_dims.nv[0]) = x.tail(_dims.nv[0]) + ddq * _dt;
        }
        else
        {
            out.tail(_dims.nv[0]) = x.tail(_dims.nv[0]) + ddq * _dt;
            out.head(_dims.nq[0]) =
              pinocchio_wrapper::integrate(_impl,
                                           x.head(_dims.nq[0]),
                                           out.tail(_dims.nv[0]) * _dt);
        }

        return out;
    };

    auto dyn_grad =
      [this, ctx](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
    {
        VecX q   = x.head(_dims.nq[0]);
        VecX v   = x.tail(_dims.nv[0]);
        VecX tau = ctx.S * u.head(_dims.nv[0]);

        MatX ddq_dq, ddq_dv, Minv;
        VecX vn;

        MatX Jf_full;

        if constexpr (WithForces)
        {
            pinocchio_wrapper::computeJointJacobians(_impl, q);

            Jf_full.setZero(3 * ctx.nf, _dims.nv[0]);
            std::vector<Vec6> joint_forces(ctx.total_joints, Vec6::Zero());
            for (uint i = 0; i < ctx.nf; ++i)
            {
                MatX Jf = frameJacobian(ctx.forces[i].frame, true).topRows<3>();
                Jf_full.block(3 * i, 0, 3, _dims.nv[0]) = Jf;

                Vec3 f = u.segment(_dims.nv[0] + 3 * i, 3);

                uint joint_id;
                Vec6 f_res = pinocchio_wrapper::framejMfAct(_impl,
                                                            ctx.forces[i].frame,
                                                            f,
                                                            joint_id);

                joint_forces[joint_id] += f_res;
            }

            if constexpr (IS == IntegrationScheme::ExplicitEuler)
            {
                vn = v;
            }
            else
            {
                VecX ddq =
                  pinocchio_wrapper::aba(_impl, q, v, tau, joint_forces);
                vn = v + ddq * _dt;
            }

            pinocchio_wrapper::computeABADerivatives(_impl,
                                                     q,
                                                     v,
                                                     tau,
                                                     joint_forces,
                                                     ddq_dq,
                                                     ddq_dv,
                                                     Minv);
        }
        else
        {
            if constexpr (IS == IntegrationScheme::ExplicitEuler)
            {
                vn = v;
            }
            else
            {
                VecX ddq = pinocchio_wrapper::aba(_impl, q, v, tau);
                vn       = v + ddq * _dt;
            }

            pinocchio_wrapper::computeABADerivatives(_impl,
                                                     q,
                                                     v,
                                                     tau,
                                                     ddq_dq,
                                                     ddq_dv,
                                                     Minv);
        }

        MatX Jint_q = MatX::Zero(_dims.nv[0], _dims.nv[0]);
        MatX Jint_v = MatX::Zero(_dims.nv[0], _dims.nv[0]);

        pinocchio_wrapper::dIntegrate(_impl, q, vn * _dt, Jint_q, 0);
        pinocchio_wrapper::dIntegrate(_impl, q, vn * _dt, Jint_v, 1);

        if constexpr (IS == IntegrationScheme::ExplicitEuler)
        {
            dX = MatX::Zero(2 * _dims.nv[0], 2 * _dims.nv[0]);
            dX.topLeftCorner(_dims.nv[0], _dims.nv[0])    = Jint_q;
            dX.topRightCorner(_dims.nv[0], _dims.nv[0])   = Jint_v * _dt;
            dX.bottomLeftCorner(_dims.nv[0], _dims.nv[0]) = ddq_dq * _dt;
            dX.bottomRightCorner(_dims.nv[0], _dims.nv[0]) =
              MatX::Identity(_dims.nv[0], _dims.nv[0]) + ddq_dv * _dt;

            if constexpr (WithForces)
            {
                dU = MatX::Zero(2 * _dims.nv[0], _dims.nv[0] + 3 * ctx.nf);
                dU.bottomLeftCorner(_dims.nv[0], _dims.nv[0]) =
                  Minv * ctx.S * _dt;
                dU.bottomRightCorner(_dims.nv[0], 3 * ctx.nf) =
                  Minv * Jf_full.transpose() * _dt;
            }
            else
            {
                dU = MatX::Zero(2 * _dims.nv[0], _dims.nv[0]);
                dU.bottomRows(_dims.nv[0]) = Minv * ctx.S * _dt;
            }
        }
        else
        {
            dX = MatX::Zero(2 * _dims.nv[0], 2 * _dims.nv[0]);
            dX.topLeftCorner(_dims.nv[0], _dims.nv[0]) =
              Jint_q + Jint_v * _dt * _dt * ddq_dq;
            dX.topRightCorner(_dims.nv[0], _dims.nv[0]) =
              Jint_v * _dt *
              (MatX::Identity(_dims.nv[0], _dims.nv[0]) + ddq_dv * _dt);
            dX.bottomLeftCorner(_dims.nv[0], _dims.nv[0]) = ddq_dq * _dt;
            dX.bottomRightCorner(_dims.nv[0], _dims.nv[0]) =
              MatX::Identity(_dims.nv[0], _dims.nv[0]) + ddq_dv * _dt;

            if constexpr (WithForces)
            {
                dU = MatX::Zero(2 * _dims.nv[0], _dims.nv[0] + 3 * ctx.nf);
                dU.topLeftCorner(_dims.nv[0], _dims.nv[0]) =
                  Jint_v * _dt * _dt * Minv * ctx.S;
                dU.topRightCorner(_dims.nv[0], 3 * ctx.nf) =
                  Jint_v * _dt * _dt * Minv * Jf_full.transpose();
                dU.bottomLeftCorner(_dims.nv[0], _dims.nv[0]) =
                  Minv * ctx.S * _dt;
                dU.bottomRightCorner(_dims.nv[0], 3 * ctx.nf) =
                  Minv * Jf_full.transpose() * _dt;
            }
            else
            {
                dU = MatX::Zero(2 * _dims.nv[0], _dims.nv[0]);
                dU.topRows(_dims.nv[0])    = Jint_v * _dt * _dt * Minv * ctx.S;
                dU.bottomRows(_dims.nv[0]) = Minv * ctx.S * _dt;
            }
        }
    };

    auto dyn_hessprod_xx =
      [this](const VecX &x, const VecX &u, const VecX &lambda, MatX &Pxx)
    { Pxx = MatX::Zero(2 * _dims.nv[0], 2 * _dims.nv[0]); };

    auto dyn_hessprod_xu =
      [this, ctx](const VecX &x, const VecX &u, const VecX &lambda, MatX &Pxu)
    {
        if constexpr (WithForces)
        {
            Pxu = MatX::Zero(2 * _dims.nv[0], _dims.nv[0] + 3 * ctx.nf);
        }
        else
        {
            Pxu = MatX::Zero(2 * _dims.nv[0], _dims.nv[0]);
        }
    };

    auto dyn_hessprod_uu =
      [this, ctx](const VecX &x, const VecX &u, const VecX &lambda, MatX &Puu)
    {
        if constexpr (WithForces)
        {
            Puu =
              MatX::Zero(_dims.nv[0] + 3 * ctx.nf, _dims.nv[0] + 3 * ctx.nf);
        }
        else
        {
            Puu = MatX::Zero(_dims.nv[0], _dims.nv[0]);
        }
    };

    return stagewise::factory::make_dynamics(_dims.nq[0] + _dims.nv[0],
                                             _dims.nv[0] + 3 * ctx.nf,
                                             _dims.nq[0] + _dims.nv[0],
                                             2 * _dims.nv[0],
                                             ctx.init_state,
                                             ctx.init_control,
                                             ops,
                                             dyn_eval,
                                             dyn_grad,
                                             dyn_hessprod_xx,
                                             dyn_hessprod_xu,
                                             dyn_hessprod_uu);
}
}
}