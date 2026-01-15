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

template<bool WithForces, DynamicsMode DM, IntegrationScheme IS>
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
        VecX ddq;

        if constexpr (DM == DynamicsMode::Forward)
        {
            VecX tau = ctx.S * u.head(_dims.nv[0]);

            if constexpr (WithForces)
            {
                std::vector<Vec6> joint_forces =
                  _calculateJointForces(x.head(_dims.nq[0]),
                                        u.tail(u.size() - _dims.nv[0]),
                                        ctx.force_data);

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
        }
        else if constexpr (DM == DynamicsMode::Inverse)
        {
            ddq = u.head(_dims.nv[0]);
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
        else if constexpr (IS == IntegrationScheme::SemiImplicitEuler)
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

        if constexpr (DM == DynamicsMode::Forward)
        {
            if constexpr (WithForces)
            {
                std::vector<Vec6> joint_forces;
                std::tie(joint_forces, Jf_full) =
                  _calculateJointForcesAndJacobian(
                    q,
                    u.tail(u.size() - _dims.nv[0]),
                    ctx.force_data);

                if constexpr (IS == IntegrationScheme::ExplicitEuler)
                {
                    vn = v;
                }
                else if constexpr (IS == IntegrationScheme::SemiImplicitEuler)
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
                else if constexpr (IS == IntegrationScheme::SemiImplicitEuler)
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
        }
        else if constexpr (DM == DynamicsMode::Inverse)
        {
            if constexpr (IS == IntegrationScheme::ExplicitEuler)
            {
                vn = v;
            }
            else if constexpr (IS == IntegrationScheme::SemiImplicitEuler)
            {
                vn = v + u.head(_dims.nv[0]) * _dt;
            }
        }

        MatX Jint_q = MatX::Zero(_dims.nv[0], _dims.nv[0]);
        MatX Jint_v = MatX::Zero(_dims.nv[0], _dims.nv[0]);

        pinocchio_wrapper::dIntegrate(_impl, q, vn * _dt, Jint_q, 0);
        pinocchio_wrapper::dIntegrate(_impl, q, vn * _dt, Jint_v, 1);

        if constexpr (IS == IntegrationScheme::ExplicitEuler)
        {
            dX = MatX::Zero(2 * _dims.nv[0], 2 * _dims.nv[0]);
            dX.topLeftCorner(_dims.nv[0], _dims.nv[0])  = Jint_q;
            dX.topRightCorner(_dims.nv[0], _dims.nv[0]) = Jint_v * _dt;

            if constexpr (DM == DynamicsMode::Forward)
            {
                dX.bottomLeftCorner(_dims.nv[0], _dims.nv[0]) = ddq_dq * _dt;
                dX.bottomRightCorner(_dims.nv[0], _dims.nv[0]) =
                  MatX::Identity(_dims.nv[0], _dims.nv[0]) + ddq_dv * _dt;

                if constexpr (WithForces)
                {
                    dU = MatX::Zero(2 * _dims.nv[0],
                                    _dims.nv[0] + 3 * ctx.force_data.size());
                    dU.bottomLeftCorner(_dims.nv[0], _dims.nv[0]) =
                      Minv * ctx.S * _dt;
                    dU.bottomRightCorner(_dims.nv[0],
                                         3 * ctx.force_data.size()) =
                      Minv * Jf_full.transpose() * _dt;
                }
                else
                {
                    dU = MatX::Zero(2 * _dims.nv[0], _dims.nv[0]);
                    dU.bottomRows(_dims.nv[0]) = Minv * ctx.S * _dt;
                }
            }
            else if constexpr (DM == DynamicsMode::Inverse)
            {
                dX.bottomRightCorner(_dims.nv[0], _dims.nv[0]) =
                  MatX::Identity(_dims.nv[0], _dims.nv[0]);

                if constexpr (WithForces)
                {
                    dU = MatX::Zero(2 * _dims.nv[0],
                                    _dims.nv[0] + 3 * ctx.force_data.size());
                }
                else
                {
                    dU = MatX::Zero(2 * _dims.nv[0], _dims.nv[0]);
                }
                dU.bottomLeftCorner(_dims.nv[0], _dims.nv[0])
                  .diagonal()
                  .setConstant(_dt);
            }
        }
        else if constexpr (IS == IntegrationScheme::SemiImplicitEuler)
        {
            dX = MatX::Zero(2 * _dims.nv[0], 2 * _dims.nv[0]);

            if constexpr (DM == DynamicsMode::Forward)
            {
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
                    dU = MatX::Zero(2 * _dims.nv[0],
                                    _dims.nv[0] + 3 * ctx.force_data.size());
                    dU.topLeftCorner(_dims.nv[0], _dims.nv[0]) =
                      Jint_v * _dt * _dt * Minv * ctx.S;
                    dU.topRightCorner(_dims.nv[0], 3 * ctx.force_data.size()) =
                      Jint_v * _dt * _dt * Minv * Jf_full.transpose();
                    dU.bottomLeftCorner(_dims.nv[0], _dims.nv[0]) =
                      Minv * ctx.S * _dt;
                    dU.bottomRightCorner(_dims.nv[0],
                                         3 * ctx.force_data.size()) =
                      Minv * Jf_full.transpose() * _dt;
                }
                else
                {
                    dU = MatX::Zero(2 * _dims.nv[0], _dims.nv[0]);
                    dU.topRows(_dims.nv[0]) = Jint_v * _dt * _dt * Minv * ctx.S;
                    dU.bottomRows(_dims.nv[0]) = Minv * ctx.S * _dt;
                }
            }
            else if constexpr (DM == DynamicsMode::Inverse)
            {
                dX.topLeftCorner(_dims.nv[0], _dims.nv[0])  = Jint_q;
                dX.topRightCorner(_dims.nv[0], _dims.nv[0]) = Jint_v * _dt;
                dX.bottomRightCorner(_dims.nv[0], _dims.nv[0]) =
                  MatX::Identity(_dims.nv[0], _dims.nv[0]);

                if constexpr (WithForces)
                {
                    dU = MatX::Zero(2 * _dims.nv[0],
                                    _dims.nv[0] + 3 * ctx.force_data.size());
                }
                else
                {
                    dU = MatX::Zero(2 * _dims.nv[0], _dims.nv[0]);
                }
                dU.topLeftCorner(_dims.nv[0], _dims.nv[0]) = Jint_v * _dt * _dt;
                dU.bottomLeftCorner(_dims.nv[0], _dims.nv[0])
                  .diagonal()
                  .setConstant(_dt);
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
            Pxu = MatX::Zero(2 * _dims.nv[0],
                             _dims.nv[0] + 3 * ctx.force_data.size());
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
            Puu = MatX::Zero(_dims.nv[0] + 3 * ctx.force_data.size(),
                             _dims.nv[0] + 3 * ctx.force_data.size());
        }
        else
        {
            Puu = MatX::Zero(_dims.nv[0], _dims.nv[0]);
        }
    };

    return stagewise::factory::make_dynamics(_dims.nq[0] + _dims.nv[0],
                                             _dims.nv[0] +
                                               3 * ctx.force_data.size(),
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

template<bool FloatingBase, bool IsTerminal, DynamicsMode DM, bool WithForces>
std::shared_ptr<stagewise::StageConstraints> Model::_makeStageInnateConstraints(
  const _StageInnateConstraintsContext &ctx) const
{
    VecX lb, ub;

    if constexpr (DM == DynamicsMode::Forward)
    {
        std::tie(lb, ub) = pinocchio_wrapper::modelBounds(_impl, _free_flyer);
    }
    else if constexpr (DM == DynamicsMode::Inverse)
    {
        std::tie(lb, ub) =
          pinocchio_wrapper::modelBounds(_impl, _free_flyer, true);
    }

    auto innate_constraint_eval = [this, ctx](const VecX &x, const VecX &u)
    {
        VecX c;

        if constexpr (IsTerminal)
        {
            c.resize(2 * _dims.njoints[0]);
        }
        else if constexpr (DM == DynamicsMode::Forward)
        {
            c.resize(3 * _dims.njoints[0]);
        }
        else if constexpr (DM == DynamicsMode::Inverse)
        {
            c.resize(2 * _dims.njoints[0]);
        }

        if constexpr (FloatingBase)
        {
            c.head(_dims.njoints[0]) = x.segment(7, _dims.njoints[0]);
            c.segment(_dims.njoints[0], _dims.njoints[0]) =
              x.tail(_dims.njoints[0]);

            if constexpr (!IsTerminal && DM == DynamicsMode::Forward)
            {
                c.tail(_dims.njoints[0]) = u.segment(6, _dims.njoints[0]);
            }
        }
        else
        {
            c.head(2 * _dims.njoints[0]) = x;

            if constexpr (!IsTerminal && DM == DynamicsMode::Forward)
            {
                c.tail(_dims.njoints[0]) = u.head(_dims.njoints[0]);
            }
        }

        return c;
    };

    auto innate_constraint_grad =
      [this, ctx](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
    {
        if constexpr (IsTerminal)
        {
            dX = MatX::Zero(2 * _dims.njoints[0], 2 * _dims.nv[0]);
            dU = MatX::Zero(2 * _dims.njoints[0], 1);
        }
        else if (DM == DynamicsMode::Forward)
        {
            dX = MatX::Zero(3 * _dims.njoints[0], 2 * _dims.nv[0]);
            dU = MatX::Zero(3 * _dims.njoints[0], u.size());
        }
        else if (DM == DynamicsMode::Inverse)
        {
            dX = MatX::Zero(2 * _dims.njoints[0], 2 * _dims.nv[0]);
            dU = MatX::Zero(2 * _dims.njoints[0], u.size());
        }

        if constexpr (FloatingBase)
        {
            dX.block(0, 6, _dims.njoints[0], _dims.njoints[0]) =
              MatX::Identity(_dims.njoints[0], _dims.njoints[0]);
            dX.block(_dims.njoints[0],
                     _dims.nv[0] + 6,
                     _dims.njoints[0],
                     _dims.njoints[0]) =
              MatX::Identity(_dims.njoints[0], _dims.njoints[0]);

            if constexpr (!IsTerminal && DM == DynamicsMode::Forward)
            {
                dU.block(2 * _dims.njoints[0],
                         6,
                         _dims.njoints[0],
                         _dims.njoints[0]) =
                  MatX::Identity(_dims.njoints[0], _dims.njoints[0]);
            }
        }
        else
        {
            dX.block(0, 0, 2 * _dims.njoints[0], 2 * _dims.njoints[0]) =
              MatX::Identity(2 * _dims.njoints[0], 2 * _dims.njoints[0]);

            if constexpr (!IsTerminal && DM == DynamicsMode::Forward)
            {
                dU.block(2 * _dims.njoints[0],
                         0,
                         _dims.njoints[0],
                         _dims.njoints[0]) =
                  MatX::Identity(_dims.njoints[0], _dims.njoints[0]);
            }
        }
    };

    auto innate_constraint_hessprod_xx =
      [this](const VecX &x, const VecX &u, const VecX &lambda, MatX &Pxx)
    { Pxx = MatX::Zero(2 * _dims.nv[0], 2 * _dims.nv[0]); };

    auto innate_constraint_hessprod_xu =
      [this](const VecX &x, const VecX &u, const VecX &lambda, MatX &Pxu)
    { Pxu = MatX::Zero(2 * _dims.nv[0], u.size()); };

    auto innate_constraint_hessprod_uu =
      [this](const VecX &x, const VecX &u, const VecX &lambda, MatX &Puu)
    { Puu = MatX::Zero(u.size(), u.size()); };

    if constexpr (IsTerminal)
    {
        return stagewise::factory::make_constraints(
          2 * _dims.njoints[0],
          2 * _dims.njoints[0],
          lb.head(2 * _dims.njoints[0]),
          ub.head(2 * _dims.njoints[0]),
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
          3 * _dims.njoints[0],
          3 * _dims.njoints[0],
          lb,
          ub,
          innate_constraint_eval,
          innate_constraint_grad,
          DefaultRightMinus,
          innate_constraint_hessprod_xx,
          innate_constraint_hessprod_xu,
          innate_constraint_hessprod_uu);
    }
    else if constexpr (DM == DynamicsMode::Inverse)
    {
        return stagewise::factory::make_constraints(
          2 * _dims.njoints[0],
          2 * _dims.njoints[0],
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

template<bool FloatingBase, bool WithForces>
std::shared_ptr<stagewise::StageConstraints>
Model::_makeStageIDTorqueConstraints(
  const _StageInnateConstraintsContext &ctx) const
{
    VecX lb, ub;
    std::tie(lb, ub) = pinocchio_wrapper::modelEffortLimits(_impl);

    if (!_actuated_joints.empty())
    {
        Eigen::VectorXd lb_n = Eigen::VectorXd::Zero(_dims.njoints[0]);
        Eigen::VectorXd ub_n = Eigen::VectorXd::Zero(_dims.njoints[0]);

        for (uint i = 0; i < _actuated_joints.size(); ++i)
        {
            uint constraint_idx;
            if constexpr (FloatingBase)
            {
                constraint_idx =
                  _dims.nq[0] + _dims.nv[0] + 6 + _actuated_joints[i];
            }
            else
            {
                constraint_idx =
                  _dims.nq[0] + _dims.nv[0] + _actuated_joints[i];
            }
            lb_n[_actuated_joints[i]] = lb[constraint_idx];
            ub_n[_actuated_joints[i]] = ub[constraint_idx];
        }

        lb.tail(_dims.njoints[0]) = lb_n;
        ub.tail(_dims.njoints[0]) = ub_n;
    }

    auto innate_constraint_eval = [this, ctx](const VecX &x, const VecX &u)
    {
        VecX c(_dims.nv[0]);

        if constexpr (WithForces)
        {
            std::vector<Vec6> joint_forces =
              _calculateJointForces(x.head(_dims.nq[0]),
                                    u.tail(u.size() - _dims.nv[0]),
                                    ctx.force_data);

            c = inverseDynamics(x.head(_dims.nq[0]),
                                x.tail(_dims.nv[0]),
                                u.head(_dims.nv[0]),
                                joint_forces);
        }
        else
        {
            c = inverseDynamics(x.head(_dims.nq[0]),
                                x.tail(_dims.nv[0]),
                                u.head(_dims.nv[0]));
        }

        return c;
    };

    auto innate_constraint_grad =
      [this, ctx](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
    {
        dX = MatX::Zero(_dims.nv[0], 2 * _dims.nv[0]);
        dU = MatX::Zero(_dims.nv[0], u.size());

        MatX dtau_dq, dtau_dv, M;
        MatX Jf_full;

        if constexpr (WithForces)
        {
            std::vector<Vec6> joint_forces;
            std::tie(joint_forces, Jf_full) =
              _calculateJointForcesAndJacobian(x.head(_dims.nq[0]),
                                               u.tail(u.size() - _dims.nv[0]),
                                               ctx.force_data);

            pinocchio_wrapper::computeRNEADerivatives(_impl,
                                                      x.head(_dims.nq[0]),
                                                      x.tail(_dims.nv[0]),
                                                      u.head(_dims.nv[0]),
                                                      joint_forces,
                                                      dtau_dq,
                                                      dtau_dv,
                                                      M);
        }
        else
        {
            pinocchio_wrapper::computeRNEADerivatives(_impl,
                                                      x.head(_dims.nq[0]),
                                                      x.tail(_dims.nv[0]),
                                                      u.head(_dims.nv[0]),
                                                      dtau_dq,
                                                      dtau_dv,
                                                      M);
        }

        M = M.selfadjointView<Eigen::Upper>();

        dX.block(0, 0, _dims.nv[0], _dims.nv[0])           = dtau_dq;
        dX.block(0, _dims.nv[0], _dims.nv[0], _dims.nv[0]) = dtau_dv;
        dU.block(0, 0, _dims.nv[0], _dims.nv[0])           = M;

        if constexpr (WithForces)
        {
            dU.block(0, _dims.nv[0], _dims.nv[0], 3 * ctx.force_data.size()) =
              -Jf_full.transpose();
        }
    };

    auto innate_constraint_hessprod_xx =
      [this](const VecX &x, const VecX &u, const VecX &lambda, MatX &Pxx)
    { Pxx = MatX::Zero(2 * _dims.nv[0], 2 * _dims.nv[0]); };

    auto innate_constraint_hessprod_xu =
      [this](const VecX &x, const VecX &u, const VecX &lambda, MatX &Pxu)
    { Pxu = MatX::Zero(2 * _dims.nv[0], u.size()); };

    auto innate_constraint_hessprod_uu =
      [this](const VecX &x, const VecX &u, const VecX &lambda, MatX &Puu)
    { Puu = MatX::Zero(u.size(), u.size()); };

    return stagewise::factory::make_constraints(_dims.nv[0],
                                                _dims.nv[0],
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
}