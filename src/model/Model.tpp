#include "model/Model.hpp"

namespace ssqp
{
namespace model
{
template<typename L, typename... Args>
std::shared_ptr<stagewise::StageCost> Model::addCost(const uint k, Args... args)
{
    static_assert(std::is_base_of_v<stagewise::StageCost, L>, "Invalid cost");

    std::shared_ptr<stagewise::StageCost> cost = std::make_shared<L>(args...);

    addCost(k, cost);

    return cost;
}

template<typename C, typename... Args>
std::shared_ptr<stagewise::StageConstraints> Model::addConstraints(const uint k,
                                                                   Args... args)
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

    return constraints;
}

template<bool FloatingBase, bool WithForces, DynamicsMode DM,
         IntegrationScheme IS>
std::shared_ptr<stagewise::StageDynamics> Model::_makeStageDynamics(
  const _StageDynamicsContext &ctx) const
{
    auto right_plus = [this, ctx](const VecX &x, const VecX &dx)
    {
        VecX out(ctx.nq + ctx.nv);

        out.head(ctx.nq) =
          pinocchio_wrapper::integrate(_impl, x.head(ctx.nq), dx.head(ctx.nv));
        out.tail(ctx.nv) = x.tail(ctx.nv) + dx.tail(ctx.nv);

        return out;
    };

    auto right_minus = [this, ctx](const VecX &x, const VecX &y)
    {
        VecX out(2 * ctx.nv);

        out.head(ctx.nv) =
          pinocchio_wrapper::difference(_impl, y.head(ctx.nq), x.head(ctx.nq));
        out.tail(ctx.nv) = x.tail(ctx.nv) - y.tail(ctx.nv);

        return out;
    };

    ManifoldOperators<decltype(right_plus), decltype(right_minus)> ops = {
        right_plus,
        right_minus
    };

    auto dyn_eval = [this, ctx](const VecX &x, const VecX &u)
    {
        VecX q    = x.head(ctx.nq);
        VecX v    = x.tail(ctx.nv);
        VecX ctrl = u.head(ctx.nu);
        VecX f    = u.tail(3 * ctx.nf);
        if constexpr (DM == DynamicsMode::Forward)
        {
            ctrl = ctx.S * ctrl;
        }

        VecX ddq;

        if constexpr (DM == DynamicsMode::Forward)
        {
            if constexpr (WithForces)
            {
                std::vector<Vec6> joint_forces =
                  _calculateJointForces(q, f, ctx.force_data);

                ddq = pinocchio_wrapper::aba(_impl, q, v, ctrl, joint_forces);
            }
            else
            {
                ddq = pinocchio_wrapper::aba(_impl, q, v, ctrl);
            }
        }
        else if constexpr (DM == DynamicsMode::Inverse)
        {
            ddq = ctrl;
        }
        else if constexpr (DM == DynamicsMode::ReducedInverse)
        {
            ddq = _calculateReducedInverseFullAcceleration<FloatingBase,
                                                           WithForces>(
              q,
              v,
              ctrl,
              f,
              ctx.force_data,
              ctx.actuated_joints,
              ctx.unactuated_joints);
        }

        VecX out(ctx.nq + ctx.nv);

        if constexpr (IS == IntegrationScheme::ExplicitEuler)
        {
            out.head(ctx.nq) = pinocchio_wrapper::integrate(_impl, q, v * _dt);
            out.tail(ctx.nv) = v + ddq * _dt;
        }
        else if constexpr (IS == IntegrationScheme::SemiImplicitEuler)
        {
            out.tail(ctx.nv) = v + ddq * _dt;
            out.head(ctx.nq) =
              pinocchio_wrapper::integrate(_impl, q, out.tail(ctx.nv) * _dt);
        }

        return out;
    };

    auto dyn_grad =
      [this, ctx](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
    {
        VecX q    = x.head(ctx.nq);
        VecX v    = x.tail(ctx.nv);
        VecX ctrl = u.head(ctx.nu);
        VecX f    = u.tail(3 * ctx.nf);
        if constexpr (DM == DynamicsMode::Forward)
        {
            ctrl = ctx.S * ctrl;
        }

        MatX ddq_dq, ddq_dv, Minv;
        VecX vn;

        std::vector<Vec6> joint_forces;
        MatX              Jf_full;

        VecX              ddq;
        MatX              Muu, Muj;
        Eigen::LDLT<MatX> Muu_ldlt;

        if constexpr (DM == DynamicsMode::Forward)
        {
            if constexpr (WithForces)
            {
                std::tie(joint_forces, Jf_full) =
                  _calculateJointForcesAndJacobian(q, f, ctx.force_data);

                if constexpr (IS == IntegrationScheme::ExplicitEuler)
                {
                    vn = v;
                }
                else if constexpr (IS == IntegrationScheme::SemiImplicitEuler)
                {
                    ddq =
                      pinocchio_wrapper::aba(_impl, q, v, ctrl, joint_forces);
                    vn = v + ddq * _dt;
                }

                pinocchio_wrapper::computeABADerivatives(_impl,
                                                         q,
                                                         v,
                                                         ctrl,
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
                else if constexpr (IS == IntegrationScheme::SemiImplicitEuler)
                {
                    VecX ddq = pinocchio_wrapper::aba(_impl, q, v, ctrl);
                    vn       = v + ddq * _dt;
                }

                pinocchio_wrapper::computeABADerivatives(_impl,
                                                         q,
                                                         v,
                                                         ctrl,
                                                         ddq_dq,
                                                         ddq_dv,
                                                         Minv);
            }
        }
        else if constexpr (DM == DynamicsMode::Inverse)
        {
            if constexpr (IS == IntegrationScheme::ExplicitEuler)
            {
                vn = v;
            }
            else if constexpr (IS == IntegrationScheme::SemiImplicitEuler)
            {
                vn = v + ctrl * _dt;
            }
        }
        else if constexpr (DM == DynamicsMode::ReducedInverse)
        {
            if constexpr (IS == IntegrationScheme::ExplicitEuler)
            {
                vn = v;
            }
            else if constexpr (IS == IntegrationScheme::SemiImplicitEuler)
            {
                ddq = _calculateReducedInverseFullAcceleration<FloatingBase,
                                                               WithForces>(
                  q,
                  v,
                  ctrl,
                  f,
                  ctx.force_data,
                  ctx.actuated_joints,
                  ctx.unactuated_joints,
                  Muu,
                  Muj,
                  Muu_ldlt,
                  joint_forces,
                  Jf_full);

                vn = v + ddq * _dt;
            }
        }

        MatX Jint_q = MatX::Zero(ctx.nv, ctx.nv);
        MatX Jint_v = MatX::Zero(ctx.nv, ctx.nv);

        pinocchio_wrapper::dIntegrate(_impl, q, vn * _dt, Jint_q, 0);
        pinocchio_wrapper::dIntegrate(_impl, q, vn * _dt, Jint_v, 1);

        dX = MatX::Zero(2 * ctx.nv, 2 * ctx.nv);
        dU = MatX::Zero(2 * ctx.nv, ctx.nu + 3 * ctx.nf);

        if constexpr (IS == IntegrationScheme::ExplicitEuler)
        {
            dX.topLeftCorner(ctx.nv, ctx.nv)  = Jint_q;
            dX.topRightCorner(ctx.nv, ctx.nv) = Jint_v * _dt;

            if constexpr (DM == DynamicsMode::Forward)
            {
                dX.bottomLeftCorner(ctx.nv, ctx.nv) = ddq_dq * _dt;
                dX.bottomRightCorner(ctx.nv, ctx.nv) =
                  MatX::Identity(ctx.nv, ctx.nv) + ddq_dv * _dt;

                dU.bottomLeftCorner(ctx.nv, ctx.nu) = Minv * ctx.S * _dt;

                if constexpr (WithForces)
                {
                    dU.bottomRightCorner(ctx.nv, 3 * ctx.nf) =
                      Minv * Jf_full.transpose() * _dt;
                }
            }
            else if constexpr (DM == DynamicsMode::Inverse)
            {
                dX.bottomRightCorner(ctx.nv, ctx.nv).setIdentity();
                dU.bottomLeftCorner(ctx.nv, ctx.nu).diagonal().setConstant(_dt);
            }
            else if constexpr (DM == DynamicsMode::ReducedInverse)
            {
                ddq = _calculateReducedInverseFullAcceleration<FloatingBase,
                                                               WithForces>(
                  q,
                  v,
                  ctrl,
                  f,
                  ctx.force_data,
                  ctx.actuated_joints,
                  ctx.unactuated_joints,
                  Muu,
                  Muj,
                  Muu_ldlt,
                  joint_forces,
                  Jf_full);

                MatX dtau_dq, dtau_dv, dtau_da, dtau_daj;
                MatX da_dq, da_dv, da_daj, da_df;

                _calculateReducedInverseFullAccelerationDerivatives<
                  FloatingBase,
                  WithForces>(q,
                              v,
                              ctrl,
                              f,
                              ctx.force_data,
                              ctx.actuated_joints,
                              ctx.unactuated_joints,
                              ddq,
                              Muu,
                              Muj,
                              Muu_ldlt,
                              joint_forces,
                              Jf_full,
                              dtau_dq,
                              dtau_dv,
                              dtau_da,
                              dtau_daj,
                              da_dq,
                              da_dv,
                              da_daj,
                              da_df);

                dX.bottomLeftCorner(ctx.nv, ctx.nv) = da_dq * _dt;
                dX.bottomRightCorner(ctx.nv, ctx.nv) =
                  MatX::Identity(ctx.nv, ctx.nv) + da_dv * _dt;

                dU.bottomLeftCorner(ctx.nv, ctx.nu) = da_daj * _dt;

                if constexpr (WithForces)
                {
                    dU.bottomRightCorner(ctx.nv, 3 * ctx.nf) = da_df * _dt;
                }
            }
        }
        else if constexpr (IS == IntegrationScheme::SemiImplicitEuler)
        {
            if constexpr (DM == DynamicsMode::Forward)
            {
                dX.topLeftCorner(ctx.nv, ctx.nv) =
                  Jint_q + Jint_v * _dt * _dt * ddq_dq;
                dX.topRightCorner(ctx.nv, ctx.nv) =
                  Jint_v * _dt *
                  (MatX::Identity(ctx.nv, ctx.nv) + ddq_dv * _dt);
                dX.bottomLeftCorner(ctx.nv, ctx.nv) = ddq_dq * _dt;
                dX.bottomRightCorner(ctx.nv, ctx.nv) =
                  MatX::Identity(ctx.nv, ctx.nv) + ddq_dv * _dt;

                dU.topLeftCorner(ctx.nv, ctx.nv) =
                  Jint_v * _dt * _dt * Minv * ctx.S;
                dU.bottomLeftCorner(ctx.nv, ctx.nv) = Minv * ctx.S * _dt;

                if constexpr (WithForces)
                {
                    dU.topRightCorner(ctx.nv, 3 * ctx.nf) =
                      Jint_v * _dt * _dt * Minv * Jf_full.transpose();

                    dU.bottomRightCorner(ctx.nv, 3 * ctx.nf) =
                      Minv * Jf_full.transpose() * _dt;
                }
            }
            else if constexpr (DM == DynamicsMode::Inverse)
            {
                dX.topLeftCorner(ctx.nv, ctx.nv)  = Jint_q;
                dX.topRightCorner(ctx.nv, ctx.nv) = Jint_v * _dt;
                dX.bottomRightCorner(ctx.nv, ctx.nv) =
                  MatX::Identity(ctx.nv, ctx.nv);

                dU.topLeftCorner(ctx.nv, ctx.nv) = Jint_v * _dt * _dt;
                dU.bottomLeftCorner(ctx.nv, ctx.nv).diagonal().setConstant(_dt);
            }
            else if constexpr (DM == DynamicsMode::ReducedInverse)
            {
                MatX dtau_dq, dtau_dv, dtau_da, dtau_daj;
                MatX da_dq, da_dv, da_daj, da_df;

                _calculateReducedInverseFullAccelerationDerivatives<
                  FloatingBase,
                  WithForces>(q,
                              v,
                              ctrl,
                              f,
                              ctx.force_data,
                              ctx.actuated_joints,
                              ctx.unactuated_joints,
                              ddq,
                              Muu,
                              Muj,
                              Muu_ldlt,
                              joint_forces,
                              Jf_full,
                              dtau_dq,
                              dtau_dv,
                              dtau_da,
                              dtau_daj,
                              da_dq,
                              da_dv,
                              da_daj,
                              da_df);

                dX.bottomLeftCorner(ctx.nv, ctx.nv) = da_dq * _dt;
                dX.bottomRightCorner(ctx.nv, ctx.nv) =
                  MatX::Identity(ctx.nv, ctx.nv) + da_dv * _dt;

                dX.topLeftCorner(ctx.nv, ctx.nv) =
                  Jint_q * (MatX::Identity(ctx.nv, ctx.nv) +
                            dX.bottomLeftCorner(ctx.nv, ctx.nv) * _dt);
                dX.topRightCorner(ctx.nv, ctx.nv) =
                  Jint_v * dX.bottomRightCorner(ctx.nv, ctx.nv) * _dt;

                dU.bottomLeftCorner(ctx.nv, ctx.nu) = da_daj * _dt;
                dU.topLeftCorner(ctx.nv, ctx.nu) =
                  Jint_v * dU.bottomLeftCorner(ctx.nv, ctx.nu) * _dt;

                if constexpr (WithForces)
                {
                    dU.bottomRightCorner(ctx.nv, 3 * ctx.nf) = da_df * _dt;
                    dU.topRightCorner(ctx.nv, 3 * ctx.nf) =
                      Jint_v * dU.bottomRightCorner(ctx.nv, 3 * ctx.nf) * _dt;
                }
            }
        }
    };

    auto dyn_hessprod_xx =
      [this, ctx](const VecX &x, const VecX &u, const VecX &lambda, MatX &Pxx)
    { Pxx = MatX::Zero(2 * ctx.nv, 2 * ctx.nv); };

    auto dyn_hessprod_xu =
      [this, ctx](const VecX &x, const VecX &u, const VecX &lambda, MatX &Pxu)
    { Pxu = MatX::Zero(2 * ctx.nv, ctx.nu + 3 * ctx.nf); };

    auto dyn_hessprod_uu =
      [this, ctx](const VecX &x, const VecX &u, const VecX &lambda, MatX &Puu)
    { Puu = MatX::Zero(ctx.nu + 3 * ctx.nf, ctx.nu + 3 * ctx.nf); };

    return stagewise::factory::make_dynamics(ctx.nq + ctx.nv,
                                             ctx.nu + 3 * ctx.nf,
                                             ctx.nq + ctx.nv,
                                             2 * ctx.nv,
                                             ctx.init_state,
                                             ctx.init_control,
                                             ops,
                                             dyn_eval,
                                             dyn_grad,
                                             dyn_hessprod_xx,
                                             dyn_hessprod_xu,
                                             dyn_hessprod_uu);
}

template<bool FloatingBase, bool IsTerminal, DynamicsMode DM>
std::shared_ptr<stagewise::StageConstraints> Model::_makeStageInnateConstraints(
  const _StageInnateConstraintsContext &ctx) const
{
    VecX lb, ub;

    if constexpr (DM == DynamicsMode::Forward)
    {
        std::tie(lb, ub) = pinocchio_wrapper::modelBounds(_impl, _free_flyer);
    }
    else if constexpr (DM == DynamicsMode::Inverse ||
                       DM == DynamicsMode::ReducedInverse)
    {
        std::tie(lb, ub) =
          pinocchio_wrapper::modelBounds(_impl, _free_flyer, true);
    }

    auto innate_constraint_eval = [this, ctx](const VecX &x, const VecX &u)
    {
        VecX c;

        if constexpr (IsTerminal || DM == DynamicsMode::Inverse ||
                      DM == DynamicsMode::ReducedInverse)
        {
            c.resize(2 * ctx.njoints);
        }
        else if constexpr (DM == DynamicsMode::Forward)
        {
            c.resize(3 * ctx.njoints);
        }

        if constexpr (FloatingBase)
        {
            c.head(ctx.njoints)                 = x.segment(7, ctx.njoints);
            c.segment(ctx.njoints, ctx.njoints) = x.tail(ctx.njoints);

            if constexpr (!IsTerminal && DM == DynamicsMode::Forward)
            {
                c.tail(ctx.njoints) = u.segment(6, ctx.njoints);
            }
        }
        else
        {
            c.head(2 * ctx.njoints) = x;

            if constexpr (!IsTerminal && DM == DynamicsMode::Forward)
            {
                c.tail(ctx.njoints) = u.head(ctx.njoints);
            }
        }

        return c;
    };

    auto innate_constraint_grad =
      [this, ctx](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
    {
        if constexpr (IsTerminal)
        {
            dX = MatX::Zero(2 * ctx.njoints, 2 * ctx.nv);
            dU = MatX::Zero(2 * ctx.njoints, 1);
        }
        else if constexpr (DM == DynamicsMode::Forward)
        {
            dX = MatX::Zero(3 * ctx.njoints, 2 * ctx.nv);
            dU = MatX::Zero(3 * ctx.njoints, ctx.nu + 3 * ctx.nf);
        }
        else if constexpr (DM == DynamicsMode::Inverse ||
                           DM == DynamicsMode::ReducedInverse)
        {
            dX = MatX::Zero(2 * ctx.njoints, 2 * ctx.nv);
            dU = MatX::Zero(2 * ctx.njoints, ctx.nu + 3 * ctx.nf);
        }

        if constexpr (FloatingBase)
        {
            dX.block(0, 6, ctx.njoints, ctx.njoints) =
              MatX::Identity(ctx.njoints, ctx.njoints);
            dX.block(ctx.njoints, ctx.nv + 6, ctx.njoints, ctx.njoints) =
              MatX::Identity(ctx.njoints, ctx.njoints);

            if constexpr (!IsTerminal && DM == DynamicsMode::Forward)
            {
                dU.block(2 * ctx.njoints, 6, ctx.njoints, ctx.njoints) =
                  MatX::Identity(ctx.njoints, ctx.njoints);
            }
        }
        else
        {
            dX.topRows(2 * ctx.njoints) =
              MatX::Identity(2 * ctx.njoints, 2 * ctx.njoints);

            if constexpr (!IsTerminal && DM == DynamicsMode::Forward)
            {
                dU.bottomLeftCorner(ctx.njoints, ctx.njoints) =
                  MatX::Identity(ctx.njoints, ctx.njoints);
            }
        }
    };

    auto innate_constraint_hessprod_xx =
      [this, ctx](const VecX &x, const VecX &u, const VecX &lambda, MatX &Pxx)
    { Pxx = MatX::Zero(2 * ctx.nv, 2 * ctx.nv); };

    auto innate_constraint_hessprod_xu =
      [this, ctx](const VecX &x, const VecX &u, const VecX &lambda, MatX &Pxu)
    { Pxu = MatX::Zero(2 * ctx.nv, ctx.nu + 3 * ctx.nf); };

    auto innate_constraint_hessprod_uu =
      [this, ctx](const VecX &x, const VecX &u, const VecX &lambda, MatX &Puu)
    { Puu = MatX::Zero(ctx.nu + 3 * ctx.nf, ctx.nu + 3 * ctx.nf); };

    if constexpr (IsTerminal)
    {
        return stagewise::factory::make_constraints(
          2 * ctx.njoints,
          2 * ctx.njoints,
          lb.head(2 * ctx.njoints),
          ub.head(2 * ctx.njoints),
          innate_constraint_eval,
          innate_constraint_grad,
          DefaultRightMinus,
          innate_constraint_hessprod_xx,
          innate_constraint_hessprod_xu,
          innate_constraint_hessprod_uu);
    }
    else if constexpr (DM == DynamicsMode::Forward)
    {
        return stagewise::factory::make_constraints(
          3 * ctx.njoints,
          3 * ctx.njoints,
          lb,
          ub,
          innate_constraint_eval,
          innate_constraint_grad,
          DefaultRightMinus,
          innate_constraint_hessprod_xx,
          innate_constraint_hessprod_xu,
          innate_constraint_hessprod_uu);
    }
    else if constexpr (DM == DynamicsMode::Inverse ||
                       DM == DynamicsMode::ReducedInverse)
    {
        return stagewise::factory::make_constraints(
          2 * ctx.njoints,
          2 * ctx.njoints,
          lb,
          ub,
          innate_constraint_eval,
          innate_constraint_grad,
          DefaultRightMinus,
          innate_constraint_hessprod_xx,
          innate_constraint_hessprod_xu,
          innate_constraint_hessprod_uu);
    }
}

template<bool FloatingBase, bool ReducedInverse, bool WithForces>
std::shared_ptr<stagewise::StageConstraints>
Model::_makeStageIDTorqueConstraints(
  const _StageInnateConstraintsContext &ctx) const
{
    VecX lb, ub;
    std::tie(lb, ub) = pinocchio_wrapper::modelEffortLimits(_impl);

    if constexpr (ReducedInverse)
    {
        VecX lb_n(ctx.nu);
        VecX ub_n(ctx.nu);

        for (uint i = 0; i < ctx.actuated_joints.size(); ++i)
        {
            if constexpr (FloatingBase)
            {
                lb_n[i] = lb[6 + ctx.actuated_joints[i]];
                ub_n[i] = ub[6 + ctx.actuated_joints[i]];
            }
            else
            {
                lb_n[i] = lb[ctx.actuated_joints[i]];
                ub_n[i] = ub[ctx.actuated_joints[i]];
            }
        }

        lb = lb_n;
        ub = ub_n;
    }
    else
    {
        uint offset = 0;
        if constexpr (FloatingBase)
        {
            lb.head<6>() = Eigen::Vector<double, 6>::Zero();
            ub.head<6>() = Eigen::Vector<double, 6>::Zero();
            offset       = 6;
        }

        for (uint i = 0; i < ctx.unactuated_joints.size(); ++i)
        {
            lb[offset + ctx.unactuated_joints[i]] = 0.0;
            ub[offset + ctx.unactuated_joints[i]] = 0.0;
        }
    }

    auto innate_constraint_eval = [this, ctx](const VecX &x, const VecX &u)
    {
        VecX c(ctx.nu);

        VecX q    = x.head(ctx.nq);
        VecX v    = x.tail(ctx.nv);
        VecX ctrl = u.head(ctx.nu);
        VecX f    = u.tail(3 * ctx.nf);

        VecX ddq;

        if constexpr (ReducedInverse)
        {
            ddq = _calculateReducedInverseFullAcceleration<FloatingBase,
                                                           WithForces>(
              q,
              v,
              ctrl,
              f,
              ctx.force_data,
              ctx.actuated_joints,
              ctx.unactuated_joints);
        }
        else
        {
            ddq = ctrl;
        }

        VecX tau;

        if constexpr (WithForces)
        {
            std::vector<Vec6> joint_forces =
              _calculateJointForces(q, f, ctx.force_data);

            tau = inverseDynamics(q, v, ddq, joint_forces);
        }
        else
        {
            tau = inverseDynamics(q, v, ddq);
        }

        if constexpr (ReducedInverse)
        {
            uint offset = 0;
            if constexpr (FloatingBase)
            {
                offset = 6;
            }

            for (uint i = 0; i < ctx.nu; ++i)
            {
                c[i] = tau[offset + ctx.actuated_joints[i]];
            }
        }
        else
        {
            c = tau;
        }

        return c;
    };

    auto innate_constraint_grad =
      [this, ctx](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
    {
        VecX q    = x.head(ctx.nq);
        VecX v    = x.tail(ctx.nv);
        VecX ctrl = u.head(ctx.nu);
        VecX f    = u.tail(3 * ctx.nf);

        dX = MatX::Zero(ctx.nu, 2 * ctx.nv);
        dU = MatX::Zero(ctx.nu, ctx.nu + 3 * ctx.nf);

        MatX Jf_full;

        std::vector<Vec6> joint_forces;

        if constexpr (ReducedInverse)
        {
            MatX              Muu, Muj;
            Eigen::LDLT<MatX> Muu_ldlt;

            VecX ddq = _calculateReducedInverseFullAcceleration<FloatingBase,
                                                                WithForces>(
              q,
              v,
              ctrl,
              f,
              ctx.force_data,
              ctx.actuated_joints,
              ctx.unactuated_joints,
              Muu,
              Muj,
              Muu_ldlt,
              joint_forces,
              Jf_full);

            MatX dtau_dq, dtau_dv, dtau_da, dtau_daj;
            MatX da_dq, da_dv, da_daj, da_df;

            _calculateReducedInverseFullAccelerationDerivatives<FloatingBase,
                                                                WithForces>(
              q,
              v,
              ctrl,
              f,
              ctx.force_data,
              ctx.actuated_joints,
              ctx.unactuated_joints,
              ddq,
              Muu,
              Muj,
              Muu_ldlt,
              joint_forces,
              Jf_full,
              dtau_dq,
              dtau_dv,
              dtau_da,
              dtau_daj,
              da_dq,
              da_dv,
              da_daj,
              da_df);

            MatX dX_full(ctx.nv, 2 * ctx.nv);
            MatX dU_full(ctx.nv, ctx.nu + 3 * ctx.nf);

            dX_full.leftCols(ctx.nv)  = dtau_dq + dtau_da * da_dq;
            dX_full.rightCols(ctx.nv) = dtau_dv + dtau_da * da_dv;
            dU_full.leftCols(ctx.nu)  = dtau_da * da_daj;

            if constexpr (WithForces)
            {
                dU_full.rightCols(3 * ctx.nf) =
                  -Jf_full.transpose() + dtau_da * da_df;
            }

            uint offset = 0;
            if constexpr (FloatingBase)
            {
                offset = 6;
            }

            for (uint i = 0; i < ctx.nu; ++i)
            {
                uint true_i = offset + ctx.actuated_joints[i];
                dX.row(i)   = dX_full.row(true_i);
                dU.row(i)   = dU_full.row(true_i);
            }
        }
        else
        {
            MatX dtau_dq, dtau_dv, M;

            if constexpr (WithForces)
            {
                std::tie(joint_forces, Jf_full) =
                  _calculateJointForcesAndJacobian(q, f, ctx.force_data);

                pinocchio_wrapper::computeRNEADerivatives(_impl,
                                                          q,
                                                          v,
                                                          ctrl,
                                                          joint_forces,
                                                          dtau_dq,
                                                          dtau_dv,
                                                          M);
            }
            else
            {
                pinocchio_wrapper::computeRNEADerivatives(_impl,
                                                          q,
                                                          v,
                                                          ctrl,
                                                          dtau_dq,
                                                          dtau_dv,
                                                          M);
            }

            dX.leftCols(ctx.nv)  = dtau_dq;
            dX.rightCols(ctx.nv) = dtau_dv;
            dU.leftCols(ctx.nu)  = M;

            if constexpr (WithForces)
            {
                dU.rightCols(3 * ctx.nf) = -Jf_full.transpose();
            }
        }
    };

    auto innate_constraint_hessprod_xx =
      [this, ctx](const VecX &x, const VecX &u, const VecX &lambda, MatX &Pxx)
    { Pxx = MatX::Zero(2 * ctx.nv, 2 * ctx.nv); };

    auto innate_constraint_hessprod_xu =
      [this, ctx](const VecX &x, const VecX &u, const VecX &lambda, MatX &Pxu)
    { Pxu = MatX::Zero(2 * ctx.nv, ctx.nu + 3 * ctx.nf); };

    auto innate_constraint_hessprod_uu =
      [this, ctx](const VecX &x, const VecX &u, const VecX &lambda, MatX &Puu)
    { Puu = MatX::Zero(ctx.nu + 3 * ctx.nf, ctx.nu + 3 * ctx.nf); };

    return stagewise::factory::make_constraints(ctx.nu,
                                                ctx.nu,
                                                lb,
                                                ub,
                                                innate_constraint_eval,
                                                innate_constraint_grad,
                                                DefaultRightMinus,
                                                innate_constraint_hessprod_xx,
                                                innate_constraint_hessprod_xu,
                                                innate_constraint_hessprod_uu);
}

template<bool FloatingBase, bool WithForces>
VecX Model::_calculateReducedInverseFullAcceleration(
  const VecX &q, const VecX &v, const VecX &ctrl, const VecX &f,
  const std::vector<Force> &force_data,
  const std::vector<uint>  &actuated_joints,
  const std::vector<uint> &unactuated_joints, MatX &Muu, MatX &Muj,
  Eigen::LDLT<MatX> &Muu_ldlt, std::vector<Vec6> &joint_forces,
  MatX &Jf_full) const
{
    const uint nv = v.size();
    const uint nu = ctrl.size();

    VecX ddq = VecX::Zero(nv);

    MatX M = pinocchio_wrapper::crba(_impl, q, true);

    VecX bias;

    if constexpr (WithForces)
    {
        std::tie(joint_forces, Jf_full) =
          _calculateJointForcesAndJacobian(q, f, force_data);

        bias =
          pinocchio_wrapper::rnea(_impl, q, v, VecX::Zero(nv), joint_forces);
    }
    else
    {
        bias = pinocchio_wrapper::rnea(_impl, q, v, VecX::Zero(nv));
    }

    const uint n_unact = nv - nu;

    Muu.resize(n_unact, n_unact);
    Muj.resize(n_unact, nu);
    VecX bias_unact(n_unact);

    uint offset = 0;
    if constexpr (FloatingBase)
    {
        Muu.topRows<6>()     = M.topLeftCorner(6, n_unact);
        Muj.topRows<6>()     = M.topRightCorner(6, nu);
        bias_unact.head<6>() = bias.head<6>();
        offset               = 6;
    }

    for (uint i = offset; i < n_unact; ++i)
    {
        uint true_i = offset + unactuated_joints[i - offset];

        if constexpr (FloatingBase)
        {
            Muu.block<1, 6>(i, 0) = M.block<1, 6>(true_i, 0);
            Muu.block<6, 1>(0, i) = M.block<6, 1>(0, true_i);
        }

        Muu(i, i) = M(true_i, true_i);

        for (uint j = offset; j < i; ++j)
        {
            Muu(i, j) = M(true_i, offset + unactuated_joints[j - offset]);
            Muu(j, i) = Muu(i, j);
        }

        for (uint j = 0; j < nu; ++j)
        {
            Muj(i, j) = M(true_i, offset + actuated_joints[j]);
        }

        bias_unact[i] = bias[true_i];
    }

    VecX rhs = Muj * ctrl + bias_unact;

    Muu_ldlt.compute(Muu);

    VecX unact_ddq = -Muu_ldlt.solve(rhs);

    if constexpr (FloatingBase)
    {
        ddq.head<6>() = unact_ddq.head<6>();
    }

    for (uint i = 0; i < actuated_joints.size(); ++i)
    {
        ddq[offset + actuated_joints[i]] = ctrl[i];
    }

    for (uint i = 0; i < unactuated_joints.size(); ++i)
    {
        ddq[offset + unactuated_joints[i]] = unact_ddq[offset + i];
    }

    return ddq;
}

template<bool FloatingBase, bool WithForces>
VecX Model::_calculateReducedInverseFullAcceleration(
  const VecX &q, const VecX &v, const VecX &ctrl, const VecX &f,
  const std::vector<Force> &force_data,
  const std::vector<uint>  &actuated_joints,
  const std::vector<uint>  &unactuated_joints) const
{
    MatX              Muu, Muj;
    Eigen::LDLT<MatX> Muu_ldlt;
    std::vector<Vec6> joint_forces;
    MatX              Jf_full;
    return _calculateReducedInverseFullAcceleration<FloatingBase, WithForces>(
      q,
      v,
      ctrl,
      f,
      force_data,
      actuated_joints,
      unactuated_joints,
      Muu,
      Muj,
      Muu_ldlt,
      joint_forces,
      Jf_full);
}

template<bool FloatingBase, bool WithForces>
void Model::_calculateReducedInverseFullAccelerationDerivatives(
  const VecX &q, const VecX &v, const VecX &ctrl, const VecX &f,
  const std::vector<Force> &force_data,
  const std::vector<uint>  &actuated_joints,
  const std::vector<uint> &unactuated_joints, const VecX &ddq, const MatX &Muu,
  const MatX &Muj, const Eigen::LDLT<MatX> &Muu_ldlt,
  const std::vector<Vec6> &joint_forces, const MatX &Jf_full, MatX &dtau_dq,
  MatX &dtau_dv, MatX &dtau_da, MatX &dtau_daj, MatX &da_dq, MatX &da_dv,
  MatX &da_daj, MatX &da_df) const
{
    const uint nv = v.size();
    const uint nu = ctrl.size();
    const uint nf = f.size() / 3;

    uint n_unact = nv - nu;
    MatX Ju(3 * nf, n_unact);

    if constexpr (WithForces)
    {
        pinocchio_wrapper::computeRNEADerivatives(_impl,
                                                  q,
                                                  v,
                                                  ddq,
                                                  joint_forces,
                                                  dtau_dq,
                                                  dtau_dv,
                                                  dtau_da);

        uint offset = 0;
        if constexpr (FloatingBase)
        {
            Ju.leftCols<6>() = Jf_full.leftCols<6>();
            offset           = 6;
        }

        for (uint i = 0; i < unactuated_joints.size(); ++i)
        {
            Ju.col(offset + i) = Jf_full.col(offset + unactuated_joints[i]);
        }
    }
    else
    {
        pinocchio_wrapper::computeRNEADerivatives(_impl,
                                                  q,
                                                  v,
                                                  ddq,
                                                  dtau_dq,
                                                  dtau_dv,
                                                  dtau_da);
    }

    MatX dtau_dq_unact(n_unact, nv);
    MatX dtau_dv_unact(n_unact, nv);

    uint offset = 0;
    if constexpr (FloatingBase)
    {
        dtau_dq_unact.topRows<6>() = dtau_dq.topRows<6>();
        dtau_dv_unact.topRows<6>() = dtau_dv.topRows<6>();
        offset                     = 6;
    }

    for (uint i = 0; i < unactuated_joints.size(); ++i)
    {
        dtau_dq_unact.row(offset + i) =
          dtau_dq.row(offset + unactuated_joints[i]);
        dtau_dv_unact.row(offset + i) =
          dtau_dv.row(offset + unactuated_joints[i]);
    }

    MatX da_dq_reordered  = MatX::Zero(nv, nv);
    MatX da_dv_reordered  = MatX::Zero(nv, nv);
    MatX da_daj_reordered = MatX::Zero(nv, nu);
    MatX da_df_reordered;

    da_dq_reordered.topLeftCorner(n_unact, nv) = -Muu_ldlt.solve(dtau_dq_unact);
    da_dv_reordered.topLeftCorner(n_unact, nv) = -Muu_ldlt.solve(dtau_dv_unact);
    da_daj_reordered.topRows(n_unact)          = -Muu_ldlt.solve(Muj);
    da_daj_reordered.bottomRows(nu).setIdentity();

    if constexpr (WithForces)
    {
        da_df_reordered = MatX::Zero(nv, 3 * nf);

        da_df_reordered.topRows(n_unact) = Muu_ldlt.solve(Ju.transpose());
    }

    da_dq  = MatX::Zero(nv, nv);
    da_dv  = MatX::Zero(nv, nv);
    da_daj = MatX::Zero(nv, nu);

    if constexpr (WithForces)
    {
        da_df = MatX::Zero(nv, 3 * nf);
    }

    if constexpr (FloatingBase)
    {
        da_dq.topRows<6>()  = da_dq_reordered.topRows<6>();
        da_dv.topRows<6>()  = da_dv_reordered.topRows<6>();
        da_daj.topRows<6>() = da_daj_reordered.topRows<6>();

        if constexpr (WithForces)
        {
            da_df.topRows<6>() = da_df_reordered.topRows<6>();
        }
    }

    for (uint i = 0; i < unactuated_joints.size(); ++i)
    {
        da_dq.row(offset + unactuated_joints[i]) =
          da_dq_reordered.row(offset + i);
        da_dv.row(offset + unactuated_joints[i]) =
          da_dv_reordered.row(offset + i);
        da_daj.row(offset + unactuated_joints[i]) =
          da_daj_reordered.row(offset + i);

        if constexpr (WithForces)
        {
            da_df.row(offset + unactuated_joints[i]) =
              da_df_reordered.row(offset + i);
        }
    }

    dtau_daj.resize(nv, nu);

    for (uint i = 0; i < actuated_joints.size(); ++i)
    {
        da_dq.row(offset + actuated_joints[i]) =
          da_dq_reordered.row(n_unact + i);
        da_dv.row(offset + actuated_joints[i]) =
          da_dv_reordered.row(n_unact + i);
        da_daj.row(offset + actuated_joints[i]) =
          da_daj_reordered.row(n_unact + i);

        if constexpr (WithForces)
        {
            da_df.row(offset + actuated_joints[i]) =
              da_df_reordered.row(n_unact + i);
        }

        dtau_daj.col(i) = dtau_da.col(offset + actuated_joints[i]);
    }
}
}
}