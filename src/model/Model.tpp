#include "model/Model.hpp"

namespace ssqp
{
namespace model
{
template<typename L, typename... Args>
std::shared_ptr<stagewise::StageCost> Model::addCost(const uint k,
                                                     Args &&...args)
{
    static_assert(std::is_base_of_v<stagewise::StageCost, L>, "Invalid cost");

    std::shared_ptr<stagewise::StageCost> cost =
      std::make_shared<L>(std::forward<Args>(args)...);

    addCost(k, cost);

    return cost;
}

template<typename L, typename... Args>
std::shared_ptr<stagewise::StageCost> Model::addRunningCost(Args &&...args)
{
    static_assert(std::is_base_of_v<stagewise::StageCost, L>, "Invalid cost");

    std::shared_ptr<stagewise::StageCost> cost =
      std::make_shared<L>(std::forward<Args>(args)...);

    addRunningCost(cost);

    return cost;
}

template<typename C, typename... Args>
std::shared_ptr<stagewise::StageConstraints> Model::addConstraints(
  const uint k, Args &&...args)
{
    static_assert(std::is_base_of_v<stagewise::StageConstraints, C>,
                  "Invalid constraint");

    std::shared_ptr<stagewise::StageConstraints> constraints =
      std::make_shared<C>(std::forward<Args>(args)...);

    uint nf = constraints->required_forces();

    std::vector<std::shared_ptr<Force>> forces(nf);
    for (uint i = 0; i < nf; ++i)
    {
        forces[i] = addForce(k, constraints->force_frame(i));
    }

    constraints->set_forces(forces);

    addConstraints(k, constraints);

    return constraints;
}

template<bool FloatingBase, bool WithForces, SolverMode SM,
         IntegrationScheme IS>
std::shared_ptr<stagewise::StageDynamics> Model::_makeStageDynamics(
  const std::shared_ptr<const _StageDynamicsContext> &ctx) const
{
    auto right_plus = [this, ctx](const VecX &x, const VecX &dx)
    {
        VecX out;

        if constexpr (SM == SolverMode::Kinematics)
        {
            out.resize(ctx->nq);
        }
        else if constexpr (SM == SolverMode::ForwardDynamics ||
                           SM == SolverMode::InverseDynamics ||
                           SM == SolverMode::ReducedInverseDynamics)
        {
            out.resize(ctx->nq + ctx->nv);
        }

        out.head(ctx->nq) = pinocchio_wrapper::integrate(_impl,
                                                         x.head(ctx->nq),
                                                         dx.head(ctx->nv));

        if constexpr (SM == SolverMode::ForwardDynamics ||
                      SM == SolverMode::InverseDynamics ||
                      SM == SolverMode::ReducedInverseDynamics)
        {
            out.tail(ctx->nv) = x.tail(ctx->nv) + dx.tail(ctx->nv);
        }

        return out;
    };

    auto right_minus = [this, ctx](const VecX &x, const VecX &y)
    {
        VecX out;
        if constexpr (SM == SolverMode::Kinematics)
        {
            out.resize(ctx->nv);
        }
        else if constexpr (SM == SolverMode::ForwardDynamics ||
                           SM == SolverMode::InverseDynamics ||
                           SM == SolverMode::ReducedInverseDynamics)
        {
            out.resize(2 * ctx->nv);
        }

        out.head(ctx->nv) = pinocchio_wrapper::difference(_impl,
                                                          y.head(ctx->nq),
                                                          x.head(ctx->nq));

        if constexpr (SM == SolverMode::ForwardDynamics ||
                      SM == SolverMode::InverseDynamics ||
                      SM == SolverMode::ReducedInverseDynamics)
        {
            out.tail(ctx->nv) = x.tail(ctx->nv) - y.tail(ctx->nv);
        }

        return out;
    };

    ManifoldOperators<decltype(right_plus), decltype(right_minus)> ops = {
        right_plus,
        right_minus
    };

    auto dyn_eval = [this, ctx](const VecX &x, const VecX &u)
    {
        VecX q    = x.head(ctx->nq);
        VecX ctrl = u.head(ctx->nu);

        VecX v;

        if constexpr (SM == SolverMode::ForwardDynamics ||
                      SM == SolverMode::InverseDynamics ||
                      SM == SolverMode::ReducedInverseDynamics)
        {
            v = x.tail(ctx->nv);
        }

        VecX out;

        // TODO: forces in kinematic mode? (maybe free variables?)
        if constexpr (SM == SolverMode::Kinematics)
        {
            out.resize(ctx->nq);
            out = pinocchio_wrapper::integrate(_impl, q, ctrl * _dt);
        }
        else if constexpr (SM == SolverMode::ForwardDynamics ||
                           SM == SolverMode::InverseDynamics ||
                           SM == SolverMode::ReducedInverseDynamics)
        {
            VecX ddq;

            if constexpr (SM == SolverMode::InverseDynamics)
            {
                ddq = ctrl;
            }
            else
            {
                DynamicsCache &cache =
                  _dynamicsCache(x, u, ctx->force_data, ctx->wrench_data);
                _ensureAcceleration<FloatingBase, WithForces, SM>(cache);
                ddq = cache.ddq;
            }

            out.resize(ctx->nq + ctx->nv);

            if constexpr (IS == IntegrationScheme::ExplicitEuler)
            {
                out.head(ctx->nq) =
                  pinocchio_wrapper::integrate(_impl, q, v * _dt);
                out.tail(ctx->nv) = v + ddq * _dt;
            }
            else if constexpr (IS == IntegrationScheme::SemiImplicitEuler)
            {
                out.tail(ctx->nv) = v + ddq * _dt;
                out.head(ctx->nq) =
                  pinocchio_wrapper::integrate(_impl,
                                               q,
                                               out.tail(ctx->nv) * _dt);
            }
        }

        return out;
    };

    auto dyn_grad =
      [this, ctx](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
    {
        VecX q    = x.head(ctx->nq);
        VecX ctrl = u.head(ctx->nu);

        VecX v;
        if constexpr (SM == SolverMode::ForwardDynamics ||
                      SM == SolverMode::InverseDynamics ||
                      SM == SolverMode::ReducedInverseDynamics)
        {
            v = x.tail(ctx->nv);
        }

        if constexpr (SM == SolverMode::Kinematics)
        {
            dX.setZero(ctx->nv, ctx->nv);
            dU.setZero(ctx->nv, ctx->nu + ctx->force_width);

            MatX Jint_v = MatX::Zero(ctx->nv, ctx->nv);

            pinocchio_wrapper::dIntegrate(_impl, q, ctrl * _dt, dX, 0);
            pinocchio_wrapper::dIntegrate(_impl, q, ctrl * _dt, Jint_v, 1);

            dU.leftCols(ctx->nv) = Jint_v * _dt;
        }
        else if constexpr (SM == SolverMode::ForwardDynamics ||
                           SM == SolverMode::InverseDynamics ||
                           SM == SolverMode::ReducedInverseDynamics)
        {
            DynamicsCache &cache =
              _dynamicsCache(x, u, ctx->force_data, ctx->wrench_data);

            VecX vn;

            if constexpr (SM == SolverMode::ForwardDynamics ||
                          SM == SolverMode::ReducedInverseDynamics)
            {
                _ensureAccelerationDerivatives<FloatingBase, WithForces, SM>(
                  cache);

                if constexpr (IS == IntegrationScheme::ExplicitEuler)
                {
                    vn = v;
                }
                else if constexpr (IS == IntegrationScheme::SemiImplicitEuler)
                {
                    _ensureAcceleration<FloatingBase, WithForces, SM>(cache);
                    vn = v + cache.ddq * _dt;
                }
            }
            else if constexpr (SM == SolverMode::InverseDynamics)
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

            const MatX &ddq_dq = cache.ddq_dq;
            const MatX &ddq_dv = cache.ddq_dv;
            const MatX &Minv   = cache.Minv;

            MatX Jint_q = MatX::Zero(ctx->nv, ctx->nv);
            MatX Jint_v = MatX::Zero(ctx->nv, ctx->nv);

            pinocchio_wrapper::dIntegrate(_impl, q, vn * _dt, Jint_q, 0);
            pinocchio_wrapper::dIntegrate(_impl, q, vn * _dt, Jint_v, 1);

            dX.setZero(2 * ctx->nv, 2 * ctx->nv);
            dU.setZero(2 * ctx->nv, ctx->nu + ctx->force_width);

            if constexpr (IS == IntegrationScheme::ExplicitEuler)
            {
                dX.topLeftCorner(ctx->nv, ctx->nv)  = Jint_q;
                dX.topRightCorner(ctx->nv, ctx->nv) = Jint_v * _dt;

                if constexpr (SM == SolverMode::ForwardDynamics)
                {
                    dX.bottomLeftCorner(ctx->nv, ctx->nv) = ddq_dq * _dt;
                    dX.bottomRightCorner(ctx->nv, ctx->nv) =
                      MatX::Identity(ctx->nv, ctx->nv) + ddq_dv * _dt;

                    auto dU_ctrl      = dU.bottomLeftCorner(ctx->nv, ctx->nu);
                    dU_ctrl.noalias() = Minv * _dt;

                    uint offset = 0;
                    if constexpr (FloatingBase)
                    {
                        dU_ctrl.leftCols<6>().setZero();
                        offset = 6;
                    }

                    for (const uint j : ctx->unactuated_joints)
                    {
                        dU_ctrl.col(offset + j).setZero();
                    }

                    if constexpr (WithForces)
                    {
                        dU.bottomRightCorner(ctx->nv, ctx->force_width) =
                          Minv * cache.Jf.transpose() * _dt;
                    }
                }
                else if constexpr (SM == SolverMode::InverseDynamics)
                {
                    dX.bottomRightCorner(ctx->nv, ctx->nv).setIdentity();
                    dU.bottomLeftCorner(ctx->nv, ctx->nu)
                      .diagonal()
                      .setConstant(_dt);
                }
                else if constexpr (SM == SolverMode::ReducedInverseDynamics)
                {
                    dX.bottomLeftCorner(ctx->nv, ctx->nv) = cache.da_dq * _dt;
                    dX.bottomRightCorner(ctx->nv, ctx->nv) =
                      MatX::Identity(ctx->nv, ctx->nv) + cache.da_dv * _dt;

                    dU.bottomLeftCorner(ctx->nv, ctx->nu) = cache.da_daj * _dt;

                    if constexpr (WithForces)
                    {
                        dU.bottomRightCorner(ctx->nv, ctx->force_width) =
                          cache.da_df * _dt;
                    }
                }
            }
            else if constexpr (IS == IntegrationScheme::SemiImplicitEuler)
            {
                if constexpr (SM == SolverMode::ForwardDynamics)
                {
                    dX.bottomLeftCorner(ctx->nv, ctx->nv) = ddq_dq * _dt;
                    dX.bottomRightCorner(ctx->nv, ctx->nv) =
                      MatX::Identity(ctx->nv, ctx->nv) + ddq_dv * _dt;

                    dX.topLeftCorner(ctx->nv, ctx->nv) =
                      Jint_q +
                      Jint_v * dX.bottomLeftCorner(ctx->nv, ctx->nv) * _dt;
                    dX.topRightCorner(ctx->nv, ctx->nv) =
                      Jint_v * dX.bottomRightCorner(ctx->nv, ctx->nv) * _dt;

                    MatX Gtau = Minv;

                    uint offset = 0;
                    if constexpr (FloatingBase)
                    {
                        Gtau.leftCols<6>().setZero();
                        offset = 6;
                    }

                    for (const uint j : ctx->unactuated_joints)
                    {
                        Gtau.col(offset + j).setZero();
                    }

                    dU.bottomLeftCorner(ctx->nv, ctx->nv) = Gtau * _dt;
                    dU.topLeftCorner(ctx->nv, ctx->nv) =
                      Jint_v * dU.bottomLeftCorner(ctx->nv, ctx->nv) * _dt;

                    if constexpr (WithForces)
                    {
                        MatX Gforce = Minv * cache.Jf.transpose();

                        dU.bottomRightCorner(ctx->nv, ctx->force_width) =
                          Gforce * _dt;
                        dU.topRightCorner(ctx->nv, ctx->force_width) =
                          Jint_v *
                          dU.bottomRightCorner(ctx->nv, ctx->force_width) * _dt;
                    }
                }
                else if constexpr (SM == SolverMode::InverseDynamics)
                {
                    dX.topLeftCorner(ctx->nv, ctx->nv)  = Jint_q;
                    dX.topRightCorner(ctx->nv, ctx->nv) = Jint_v * _dt;
                    dX.bottomRightCorner(ctx->nv, ctx->nv) =
                      MatX::Identity(ctx->nv, ctx->nv);

                    dU.topLeftCorner(ctx->nv, ctx->nv) = Jint_v * _dt * _dt;
                    dU.bottomLeftCorner(ctx->nv, ctx->nv)
                      .diagonal()
                      .setConstant(_dt);
                }
                else if constexpr (SM == SolverMode::ReducedInverseDynamics)
                {
                    dX.bottomLeftCorner(ctx->nv, ctx->nv) = cache.da_dq * _dt;
                    dX.bottomRightCorner(ctx->nv, ctx->nv) =
                      MatX::Identity(ctx->nv, ctx->nv) + cache.da_dv * _dt;

                    dX.topLeftCorner(ctx->nv, ctx->nv) =
                      Jint_q +
                      Jint_v * dX.bottomLeftCorner(ctx->nv, ctx->nv) * _dt;
                    dX.topRightCorner(ctx->nv, ctx->nv) =
                      Jint_v * dX.bottomRightCorner(ctx->nv, ctx->nv) * _dt;

                    dU.bottomLeftCorner(ctx->nv, ctx->nu) = cache.da_daj * _dt;
                    dU.topLeftCorner(ctx->nv, ctx->nu) =
                      Jint_v * dU.bottomLeftCorner(ctx->nv, ctx->nu) * _dt;

                    if constexpr (WithForces)
                    {
                        dU.bottomRightCorner(ctx->nv, ctx->force_width) =
                          cache.da_df * _dt;
                        dU.topRightCorner(ctx->nv, ctx->force_width) =
                          Jint_v *
                          dU.bottomRightCorner(ctx->nv, ctx->force_width) * _dt;
                    }
                }
            }
        }
    };

    auto dyn_hessprod_xx =
      [this, ctx](const VecX &x, const VecX &u, const VecX &lambda, MatX &Pxx)
    {
        if constexpr (SM == SolverMode::Kinematics)
        {
            Pxx.setZero(ctx->nv, ctx->nv);
        }
        else if constexpr (SM == SolverMode::ForwardDynamics ||
                           SM == SolverMode::InverseDynamics ||
                           SM == SolverMode::ReducedInverseDynamics)
        {
            Pxx.setZero(2 * ctx->nv, 2 * ctx->nv);
        }
    };

    auto dyn_hessprod_xu =
      [this, ctx](const VecX &x, const VecX &u, const VecX &lambda, MatX &Pxu)
    {
        if constexpr (SM == SolverMode::Kinematics)
        {
            Pxu.setZero(ctx->nv, ctx->nu + ctx->force_width);
        }
        else if constexpr (SM == SolverMode::ForwardDynamics ||
                           SM == SolverMode::InverseDynamics ||
                           SM == SolverMode::ReducedInverseDynamics)
        {
            Pxu.setZero(2 * ctx->nv, ctx->nu + ctx->force_width);
        }
    };

    auto dyn_hessprod_uu =
      [this, ctx](const VecX &x, const VecX &u, const VecX &lambda, MatX &Puu)
    { Puu.setZero(ctx->nu + ctx->force_width, ctx->nu + ctx->force_width); };

    if constexpr (SM == SolverMode::Kinematics)
    {
        return stagewise::factory::make_dynamics({ ctx->stage_dynamics_name },
                                                 ctx->nq,
                                                 ctx->nu + ctx->force_width,
                                                 ctx->nq,
                                                 ctx->nv,
                                                 ctx->init_state,
                                                 ctx->init_control,
                                                 ops,
                                                 dyn_eval,
                                                 dyn_grad,
                                                 dyn_hessprod_xx,
                                                 dyn_hessprod_xu,
                                                 dyn_hessprod_uu);
    }
    else if constexpr (SM == SolverMode::ForwardDynamics ||
                       SM == SolverMode::InverseDynamics ||
                       SM == SolverMode::ReducedInverseDynamics)
    {
        return stagewise::factory::make_dynamics({ ctx->stage_dynamics_name },
                                                 ctx->nq + ctx->nv,
                                                 ctx->nu + ctx->force_width,
                                                 ctx->nq + ctx->nv,
                                                 2 * ctx->nv,
                                                 ctx->init_state,
                                                 ctx->init_control,
                                                 ops,
                                                 dyn_eval,
                                                 dyn_grad,
                                                 dyn_hessprod_xx,
                                                 dyn_hessprod_xu,
                                                 dyn_hessprod_uu);
    }
}

template<bool FloatingBase, bool IsTerminal, SolverMode SM>
std::shared_ptr<stagewise::StageConstraints> Model::_makeStageInnateConstraints(
  const std::shared_ptr<const _StageInnateConstraintsContext> &ctx) const
{
    VecX lb, ub;

    if constexpr (SM == SolverMode::ForwardDynamics)
    {
        std::tie(lb, ub) =
          pinocchio_wrapper::modelBounds(_impl,
                                         _bounded_joint_indices,
                                         _bounded_joint_indices_to_config,
                                         _free_flyer);

        if (ctx->vel_bound.size())
        {
            lb.segment(_bounded_joint_indices.size(), _dims.njoints) =
              -ctx->vel_bound;
            ub.segment(_bounded_joint_indices.size(), _dims.njoints) =
              ctx->vel_bound;
        }

        if (ctx->tau_bound.size())
        {
            lb.segment(_bounded_joint_indices.size() + _dims.njoints,
                       _dims.nu) = -ctx->tau_bound;
            ub.segment(_bounded_joint_indices.size() + _dims.njoints,
                       _dims.nu) = ctx->tau_bound;
        }
    }
    else if constexpr (SM == SolverMode::InverseDynamics ||
                       SM == SolverMode::ReducedInverseDynamics)
    {
        std::tie(lb, ub) =
          pinocchio_wrapper::modelBounds(_impl,
                                         _bounded_joint_indices,
                                         _bounded_joint_indices_to_config,
                                         _free_flyer,
                                         false,
                                         true);

        if (ctx->vel_bound.size())
        {
            lb.segment(_bounded_joint_indices.size(), _dims.njoints) =
              -ctx->vel_bound;
            ub.segment(_bounded_joint_indices.size(), _dims.njoints) =
              ctx->vel_bound;
        }
    }
    else if constexpr (SM == SolverMode::Kinematics)
    {
        std::tie(lb, ub) =
          pinocchio_wrapper::modelBounds(_impl,
                                         _bounded_joint_indices,
                                         _bounded_joint_indices_to_config,
                                         _free_flyer,
                                         true,
                                         true);
    }

    if (ctx->lower_pos_bound.size())
    {
        lb.head(_bounded_joint_indices.size()) = ctx->lower_pos_bound;
    }

    if (ctx->upper_pos_bound.size())
    {
        ub.head(_bounded_joint_indices.size()) = ctx->upper_pos_bound;
    }

    uint nv_reduced = _dims.nv;
    if constexpr (FloatingBase)
    {
        nv_reduced -= 6;
    }

    auto innate_constraint_eval =
      [this, ctx, nv_reduced](const VecX &x, const VecX &u)
    {
        VecX c;

        if constexpr (SM == SolverMode::Kinematics)
        {
            c.resize(_bounded_joint_indices.size());
        }
        else if constexpr (IsTerminal || SM == SolverMode::InverseDynamics ||
                           SM == SolverMode::ReducedInverseDynamics)
        {
            c.resize(_bounded_joint_indices.size() + nv_reduced);
        }
        else if constexpr (SM == SolverMode::ForwardDynamics)
        {
            c.resize(_bounded_joint_indices.size() + 2 * nv_reduced);
        }

        for (uint i = 0; i < _bounded_joint_indices.size(); ++i)
        {
            c[i] =
              x[_bounded_joint_indices_to_config.at(_bounded_joint_indices[i])];
        }

        if constexpr (SM == SolverMode::ForwardDynamics ||
                      SM == SolverMode::InverseDynamics ||
                      SM == SolverMode::ReducedInverseDynamics)
        {
            c.segment(_bounded_joint_indices.size(), nv_reduced) =
              x.tail(nv_reduced);
        }

        if constexpr (!IsTerminal && SM == SolverMode::ForwardDynamics)
        {
            if constexpr (FloatingBase)
            {
                c.tail(nv_reduced) = u.segment(6, nv_reduced);
            }
            else
            {
                c.tail(nv_reduced) = u.head(nv_reduced);
            }
        }

        return c;
    };

    auto innate_constraint_grad =
      [this, ctx, nv_reduced](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
    {
        if constexpr (SM == SolverMode::Kinematics)
        {
            dX.setZero(_bounded_joint_indices.size(), ctx->nv);

            if constexpr (IsTerminal)
            {
                dU.setZero(_bounded_joint_indices.size(), 0);
            }
            else
            {
                dU.setZero(_bounded_joint_indices.size(),
                           ctx->nu + ctx->force_width);
            }
        }
        else if constexpr (IsTerminal)
        {
            dX.setZero(_bounded_joint_indices.size() + nv_reduced, 2 * ctx->nv);
            dU.setZero(_bounded_joint_indices.size() + nv_reduced, 0);
        }
        else if constexpr (SM == SolverMode::ForwardDynamics)
        {
            dX.setZero(_bounded_joint_indices.size() + 2 * nv_reduced,
                       2 * ctx->nv);
            dU.setZero(_bounded_joint_indices.size() + 2 * nv_reduced,
                       ctx->nu + ctx->force_width);
        }
        else if constexpr (SM == SolverMode::InverseDynamics ||
                           SM == SolverMode::ReducedInverseDynamics)
        {
            dX.setZero(_bounded_joint_indices.size() + nv_reduced, 2 * ctx->nv);
            dU.setZero(_bounded_joint_indices.size() + nv_reduced,
                       ctx->nu + ctx->force_width);
        }

        for (uint i = 0; i < _bounded_joint_indices.size(); ++i)
        {
            dX(i,
               _bounded_joint_indices_to_tangent.at(
                 _bounded_joint_indices[i])) = 1.0;
        }

        if constexpr (SM == SolverMode::ForwardDynamics ||
                      SM == SolverMode::InverseDynamics ||
                      SM == SolverMode::ReducedInverseDynamics)
        {
            if constexpr (FloatingBase)
            {
                dX.block(_bounded_joint_indices.size(),
                         ctx->nv + 6,
                         nv_reduced,
                         nv_reduced)
                  .setIdentity();
            }
            else
            {
                dX.block(_bounded_joint_indices.size(),
                         ctx->nv,
                         nv_reduced,
                         nv_reduced)
                  .setIdentity();
            }
        }

        if constexpr (!IsTerminal && SM == SolverMode::ForwardDynamics)
        {
            if constexpr (FloatingBase)
            {
                dU.block(_bounded_joint_indices.size() + nv_reduced,
                         6,
                         nv_reduced,
                         nv_reduced)
                  .setIdentity();
            }
            else
            {
                dU.block(_bounded_joint_indices.size() + nv_reduced,
                         0,
                         nv_reduced,
                         nv_reduced)
                  .setIdentity();
            }
        }
    };

    auto innate_constraint_hessprod_xx =
      [this, ctx](const VecX &x, const VecX &u, const VecX &lambda, MatX &Pxx)
    {
        if constexpr (SM == SolverMode::Kinematics)
        {
            Pxx.setZero(ctx->nv, ctx->nv);
        }
        else if constexpr (SM == SolverMode::ForwardDynamics ||
                           SM == SolverMode::InverseDynamics ||
                           SM == SolverMode::ReducedInverseDynamics)
        {
            Pxx.setZero(2 * ctx->nv, 2 * ctx->nv);
        }
    };

    auto innate_constraint_hessprod_xu =
      [this, ctx](const VecX &x, const VecX &u, const VecX &lambda, MatX &Pxu)
    {
        if constexpr (SM == SolverMode::Kinematics)
        {
            Pxu.setZero(ctx->nv, ctx->nu + ctx->force_width);
        }
        else if constexpr (SM == SolverMode::ForwardDynamics ||
                           SM == SolverMode::InverseDynamics ||
                           SM == SolverMode::ReducedInverseDynamics)
        {
            Pxu.setZero(2 * ctx->nv, ctx->nu + ctx->force_width);
        }
    };

    auto innate_constraint_hessprod_uu =
      [this, ctx](const VecX &x, const VecX &u, const VecX &lambda, MatX &Puu)
    { Puu.setZero(ctx->nu + ctx->force_width, ctx->nu + ctx->force_width); };

    if constexpr (SM == SolverMode::Kinematics)
    {
        return stagewise::factory::make_constraints(
          { ctx->stage_constraints_name },
          [this]() { return _bounded_joint_indices.size(); },
          [this]() { return _bounded_joint_indices.size(); },
          [lb]() { return lb; },
          [ub]() { return ub; },
          innate_constraint_eval,
          innate_constraint_grad,
          DefaultRightMinus,
          innate_constraint_hessprod_xx,
          innate_constraint_hessprod_xu,
          innate_constraint_hessprod_uu);
    }

    else if constexpr (IsTerminal)
    {
        return stagewise::factory::make_constraints(
          { ctx->stage_constraints_name },
          [this, nv_reduced]()
          { return _bounded_joint_indices.size() + nv_reduced; },
          [this, nv_reduced]()
          { return _bounded_joint_indices.size() + nv_reduced; },
          [this, lb, nv_reduced]()
          { return lb.head(_bounded_joint_indices.size() + nv_reduced); },
          [this, ub, nv_reduced]()
          { return ub.head(_bounded_joint_indices.size() + nv_reduced); },
          innate_constraint_eval,
          innate_constraint_grad,
          DefaultRightMinus,
          innate_constraint_hessprod_xx,
          innate_constraint_hessprod_xu,
          innate_constraint_hessprod_uu);
    }
    else if constexpr (SM == SolverMode::ForwardDynamics)
    {
        return stagewise::factory::make_constraints(
          { ctx->stage_constraints_name },
          [this, nv_reduced]()
          { return _bounded_joint_indices.size() + 2 * nv_reduced; },
          [this, nv_reduced]()
          { return _bounded_joint_indices.size() + 2 * nv_reduced; },
          [lb]() { return lb; },
          [ub]() { return ub; },
          innate_constraint_eval,
          innate_constraint_grad,
          DefaultRightMinus,
          innate_constraint_hessprod_xx,
          innate_constraint_hessprod_xu,
          innate_constraint_hessprod_uu);
    }
    else if constexpr (SM == SolverMode::InverseDynamics ||
                       SM == SolverMode::ReducedInverseDynamics)
    {
        return stagewise::factory::make_constraints(
          { ctx->stage_constraints_name },
          [this, nv_reduced]()
          { return _bounded_joint_indices.size() + nv_reduced; },
          [this, nv_reduced]()
          { return _bounded_joint_indices.size() + nv_reduced; },
          [lb]() { return lb; },
          [ub]() { return ub; },
          innate_constraint_eval,
          innate_constraint_grad,
          DefaultRightMinus,
          innate_constraint_hessprod_xx,
          innate_constraint_hessprod_xu,
          innate_constraint_hessprod_uu);
    }
}

template<bool FloatingBase, SolverMode SM, bool WithForces>
std::shared_ptr<stagewise::StageConstraints>
Model::_makeStageIDTorqueConstraints(
  const std::shared_ptr<const _StageInnateConstraintsContext> &ctx) const
{
    assert((SM == SolverMode::InverseDynamics ||
            SM == SolverMode::ReducedInverseDynamics) &&
           "ID torque constraints need an inverse dynamics solver mode!");

    VecX lb, ub;
    std::tie(lb, ub) = pinocchio_wrapper::modelEffortLimits(_impl);

    if (ctx->tau_bound.size())
    {
        lb.tail(_dims.nu) = -ctx->tau_bound;
        ub.tail(_dims.nu) = ctx->tau_bound;
    }

    if constexpr (SM == SolverMode::ReducedInverseDynamics)
    {
        VecX lb_n(ctx->nu);
        VecX ub_n(ctx->nu);

        for (uint i = 0; i < ctx->actuated_joints.size(); ++i)
        {
            if constexpr (FloatingBase)
            {
                lb_n[i] = lb[6 + ctx->actuated_joints[i]];
                ub_n[i] = ub[6 + ctx->actuated_joints[i]];
            }
            else
            {
                lb_n[i] = lb[ctx->actuated_joints[i]];
                ub_n[i] = ub[ctx->actuated_joints[i]];
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

        for (uint i = 0; i < ctx->unactuated_joints.size(); ++i)
        {
            lb[offset + ctx->unactuated_joints[i]] = 0.0;
            ub[offset + ctx->unactuated_joints[i]] = 0.0;
        }
    }

    auto innate_constraint_eval = [this, ctx](const VecX &x, const VecX &u)
    {
        VecX c(ctx->nu);

        DynamicsCache &cache =
          _dynamicsCache(x, u, ctx->force_data, ctx->wrench_data);
        _ensureTorque<FloatingBase, WithForces, SM>(cache);

        const VecX &tau = cache.tau;

        if constexpr (SM == SolverMode::ReducedInverseDynamics)
        {
            uint offset = 0;
            if constexpr (FloatingBase)
            {
                offset = 6;
            }

            for (uint i = 0; i < ctx->nu; ++i)
            {
                c[i] = tau[offset + ctx->actuated_joints[i]];
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
        dX.setZero(ctx->nu, 2 * ctx->nv);
        dU.setZero(ctx->nu, ctx->nu + ctx->force_width);

        DynamicsCache &cache =
          _dynamicsCache(x, u, ctx->force_data, ctx->wrench_data);

        if constexpr (SM == SolverMode::ReducedInverseDynamics)
        {
            _ensureAccelerationDerivatives<FloatingBase, WithForces, SM>(cache);

            const MatX &dtau_dq = cache.dtau_dq;
            const MatX &dtau_dv = cache.dtau_dv;
            const MatX &dtau_da = cache.dtau_da;
            const MatX &da_dq   = cache.da_dq;
            const MatX &da_dv   = cache.da_dv;
            const MatX &da_daj  = cache.da_daj;
            const MatX &da_df   = cache.da_df;

            MatX dX_full(ctx->nv, 2 * ctx->nv);
            MatX dU_full(ctx->nv, ctx->nu + ctx->force_width);

            dX_full.leftCols(ctx->nv)  = dtau_dq + dtau_da * da_dq;
            dX_full.rightCols(ctx->nv) = dtau_dv + dtau_da * da_dv;
            dU_full.leftCols(ctx->nu)  = dtau_da * da_daj;

            if constexpr (WithForces)
            {
                dU_full.rightCols(ctx->force_width) =
                  -cache.Jf.transpose() + dtau_da * da_df;
            }

            uint offset = 0;
            if constexpr (FloatingBase)
            {
                offset = 6;
            }

            for (uint i = 0; i < ctx->nu; ++i)
            {
                uint true_i = offset + ctx->actuated_joints[i];
                dX.row(i)   = dX_full.row(true_i);
                dU.row(i)   = dU_full.row(true_i);
            }
        }
        else
        {
            _ensureTorqueDerivatives<FloatingBase, WithForces, SM>(cache);

            dX.leftCols(ctx->nv)  = cache.dtau_dq;
            dX.rightCols(ctx->nv) = cache.dtau_dv;
            dU.leftCols(ctx->nu)  = cache.dtau_da;

            if constexpr (WithForces)
            {
                _ensureForceJacobian(cache);

                dU.rightCols(ctx->force_width) = -cache.Jf.transpose();
            }
        }
    };

    auto innate_constraint_hessprod_xx =
      [this, ctx](const VecX &x, const VecX &u, const VecX &lambda, MatX &Pxx)
    { Pxx.setZero(2 * ctx->nv, 2 * ctx->nv); };

    auto innate_constraint_hessprod_xu =
      [this, ctx](const VecX &x, const VecX &u, const VecX &lambda, MatX &Pxu)
    { Pxu.setZero(2 * ctx->nv, ctx->nu + ctx->force_width); };

    auto innate_constraint_hessprod_uu =
      [this, ctx](const VecX &x, const VecX &u, const VecX &lambda, MatX &Puu)
    { Puu.setZero(ctx->nu + ctx->force_width, ctx->nu + ctx->force_width); };

    return stagewise::factory::make_constraints(
      { ctx->stage_constraints_name },
      [ctx]() { return ctx->nu; },
      [ctx]() { return ctx->nu; },
      [lb]() { return lb; },
      [ub]() { return ub; },
      innate_constraint_eval,
      innate_constraint_grad,
      DefaultRightMinus,
      innate_constraint_hessprod_xx,
      innate_constraint_hessprod_xu,
      innate_constraint_hessprod_uu);
}

template<bool FloatingBase>
VecX Model::_forwardDynamicsControls(const VecX &u) const
{
    VecX ctrl = u.head(_dims.nu);

    uint offset = 0;

    if constexpr (FloatingBase)
    {
        ctrl.head<6>().setZero();
        offset = 6;
    }

    for (const uint j : _unactuated_joints)
    {
        ctrl[offset + j] = 0.0;
    }

    return ctrl;
}

template<bool FloatingBase, bool WithForces, SolverMode SM>
void Model::_ensureAcceleration(DynamicsCache &cache) const
{
    if (cache.acceleration)
    {
        return;
    }

    if constexpr (SM == SolverMode::ForwardDynamics)
    {
        const VecX q    = cache.x.head(_dims.nq);
        const VecX v    = cache.x.tail(_dims.nv);
        const VecX ctrl = _forwardDynamicsControls<FloatingBase>(cache.u);

        if constexpr (WithForces)
        {
            _ensureJointForces(cache);
            cache.ddq = pinocchio_wrapper::aba(_impl, q, v, ctrl, cache.fext);
        }
        else
        {
            cache.ddq = pinocchio_wrapper::aba(_impl, q, v, ctrl);
        }
    }
    else if constexpr (SM == SolverMode::InverseDynamics)
    {
        cache.ddq = cache.u.head(_dims.nu);
    }
    else if constexpr (SM == SolverMode::ReducedInverseDynamics)
    {
        _calculateReducedInverseFullAcceleration<FloatingBase, WithForces>(
          cache);
    }

    cache.acceleration = true;
}

template<bool FloatingBase, bool WithForces, SolverMode SM>
void Model::_ensureTorque(DynamicsCache &cache) const
{
    if (cache.torque)
    {
        return;
    }

    _ensureAcceleration<FloatingBase, WithForces, SM>(cache);

    if constexpr (SM == SolverMode::ReducedInverseDynamics)
    {
        // The acceleration already computed the mass matrix and the bias
        // torque, joint forces included
        cache.tau = cache.M * cache.ddq + cache.bias;
    }
    else
    {
        const VecX q = cache.x.head(_dims.nq);
        const VecX v = cache.x.tail(_dims.nv);

        if constexpr (WithForces)
        {
            _ensureJointForces(cache);
            cache.tau = inverseDynamics(q, v, cache.ddq, cache.fext);
        }
        else
        {
            cache.tau = inverseDynamics(q, v, cache.ddq);
        }
    }

    cache.torque = true;
}

template<bool FloatingBase, bool WithForces, SolverMode SM>
void Model::_ensureTorqueDerivatives(DynamicsCache &cache) const
{
    if (cache.torque_derivatives)
    {
        return;
    }

    _ensureAcceleration<FloatingBase, WithForces, SM>(cache);

    const VecX q = cache.x.head(_dims.nq);
    const VecX v = cache.x.tail(_dims.nv);

    if constexpr (WithForces)
    {
        _ensureJointForces(cache);
        pinocchio_wrapper::computeRNEADerivatives(_impl,
                                                  q,
                                                  v,
                                                  cache.ddq,
                                                  cache.fext,
                                                  cache.dtau_dq,
                                                  cache.dtau_dv,
                                                  cache.dtau_da);
    }
    else
    {
        pinocchio_wrapper::computeRNEADerivatives(_impl,
                                                  q,
                                                  v,
                                                  cache.ddq,
                                                  cache.dtau_dq,
                                                  cache.dtau_dv,
                                                  cache.dtau_da);
    }

    cache.torque_derivatives = true;
}

template<bool FloatingBase, bool WithForces, SolverMode SM>
void Model::_ensureAccelerationDerivatives(DynamicsCache &cache) const
{
    if (cache.acceleration_derivatives)
    {
        return;
    }

    if constexpr (SM == SolverMode::ForwardDynamics)
    {
        const VecX q    = cache.x.head(_dims.nq);
        const VecX v    = cache.x.tail(_dims.nv);
        const VecX ctrl = _forwardDynamicsControls<FloatingBase>(cache.u);

        if constexpr (WithForces)
        {
            _ensureForceJacobian(cache);
            pinocchio_wrapper::computeABADerivatives(_impl,
                                                     q,
                                                     v,
                                                     ctrl,
                                                     cache.fext,
                                                     cache.ddq_dq,
                                                     cache.ddq_dv,
                                                     cache.Minv);
        }
        else
        {
            pinocchio_wrapper::computeABADerivatives(_impl,
                                                     q,
                                                     v,
                                                     ctrl,
                                                     cache.ddq_dq,
                                                     cache.ddq_dv,
                                                     cache.Minv);
        }
    }
    else if constexpr (SM == SolverMode::ReducedInverseDynamics)
    {
        _calculateReducedInverseFullAccelerationDerivatives<FloatingBase,
                                                            WithForces>(cache);
    }

    cache.acceleration_derivatives = true;
}

template<bool FloatingBase, bool WithForces>
void Model::_calculateReducedInverseFullAcceleration(DynamicsCache &cache) const
{
    const uint nv = _dims.nv;
    const uint nu = _dims.nu;

    const VecX q    = cache.x.head(_dims.nq);
    const VecX v    = cache.x.tail(nv);
    const VecX ctrl = cache.u.head(nu);

    const std::vector<uint> &actuated_joints   = _actuated_joints;
    const std::vector<uint> &unactuated_joints = _unactuated_joints;

    VecX &ddq = cache.ddq;
    ddq       = VecX::Zero(nv);

    MatX              &M        = cache.M;
    MatX              &Muu      = cache.Muu;
    MatX              &Muj      = cache.Muj;
    Eigen::LDLT<MatX> &Muu_ldlt = cache.Muu_ldlt;

    pinocchio_wrapper::crba(_impl, q, M, true);

    VecX &bias = cache.bias;

    if constexpr (WithForces)
    {
        _ensureJointForces(cache);

        bias = pinocchio_wrapper::rnea(_impl, q, v, VecX::Zero(nv), cache.fext);
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
        Muu.topLeftCorner<6, 6>() = M.topLeftCorner(6, 6);

        for (uint i = 0; i < unactuated_joints.size(); ++i)
        {
            Muu.block<6, 1>(0, 6 + i) =
              M.block<6, 1>(0, 6 + unactuated_joints[i]);
        }

        for (uint i = 0; i < actuated_joints.size(); ++i)
        {
            Muj.block<6, 1>(0, i) = M.block<6, 1>(0, 6 + actuated_joints[i]);
        }

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
}

template<bool FloatingBase, bool WithForces>
void Model::_calculateReducedInverseFullAccelerationDerivatives(
  DynamicsCache &cache) const
{
    _ensureTorqueDerivatives<FloatingBase,
                             WithForces,
                             SolverMode::ReducedInverseDynamics>(cache);

    if constexpr (WithForces)
    {
        _ensureForceJacobian(cache);
    }

    const uint nv = _dims.nv;
    const uint nu = _dims.nu;
    const uint force_width =
      3 * cache.forces.size() + 6 * cache.wrenches.size();

    const std::vector<uint> &actuated_joints   = _actuated_joints;
    const std::vector<uint> &unactuated_joints = _unactuated_joints;

    const MatX              &Muu      = cache.Muu;
    const MatX              &Muj      = cache.Muj;
    const Eigen::LDLT<MatX> &Muu_ldlt = cache.Muu_ldlt;
    const MatX              &Jf_full  = cache.Jf;

    const MatX &dtau_dq  = cache.dtau_dq;
    const MatX &dtau_dv  = cache.dtau_dv;
    const MatX &dtau_da  = cache.dtau_da;
    MatX       &dtau_daj = cache.dtau_daj;
    MatX       &da_dq    = cache.da_dq;
    MatX       &da_dv    = cache.da_dv;
    MatX       &da_daj   = cache.da_daj;
    MatX       &da_df    = cache.da_df;

    MatX &Ju               = cache.Ju;
    MatX &dtau_dq_unact    = cache.dtau_dq_unact;
    MatX &dtau_dv_unact    = cache.dtau_dv_unact;
    MatX &da_dq_reordered  = cache.da_dq_reordered;
    MatX &da_dv_reordered  = cache.da_dv_reordered;
    MatX &da_daj_reordered = cache.da_daj_reordered;
    MatX &da_df_reordered  = cache.da_df_reordered;

    uint n_unact = nv - nu;
    Ju.resize(force_width, n_unact);

    if constexpr (WithForces)
    {
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

    dtau_dq_unact.resize(n_unact, nv);
    dtau_dv_unact.resize(n_unact, nv);

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

    da_dq_reordered  = MatX::Zero(nv, nv);
    da_dv_reordered  = MatX::Zero(nv, nv);
    da_daj_reordered = MatX::Zero(nv, nu);

    da_dq_reordered.topLeftCorner(n_unact, nv) = -Muu_ldlt.solve(dtau_dq_unact);
    da_dv_reordered.topLeftCorner(n_unact, nv) = -Muu_ldlt.solve(dtau_dv_unact);
    da_daj_reordered.topRows(n_unact)          = -Muu_ldlt.solve(Muj);
    da_daj_reordered.bottomRows(nu).setIdentity();

    if constexpr (WithForces)
    {
        da_df_reordered.setZero(nv, force_width);

        da_df_reordered.topRows(n_unact) = Muu_ldlt.solve(Ju.transpose());
    }

    da_dq.setZero(nv, nv);
    da_dv.setZero(nv, nv);
    da_daj.setZero(nv, nu);

    if constexpr (WithForces)
    {
        da_df.setZero(nv, force_width);
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

template<bool FloatingBase, bool WithForces>
void Model::_getActuatedTorqueVectorReducedDerivatives(
  const VecX &x, const VecX &u,
  const std::vector<std::shared_ptr<Force>> &forces,
  const std::vector<std::shared_ptr<Force>> &wrenches, MatX &dtau_dx,
  MatX &dtau_du) const
{
    const uint force_width = 3 * forces.size() + 6 * wrenches.size();

    DynamicsCache &cache = _dynamicsCache(x, u, forces, wrenches);
    _ensureAccelerationDerivatives<FloatingBase,
                                   WithForces,
                                   SolverMode::ReducedInverseDynamics>(cache);

    const MatX &Jf_full = cache.Jf;
    const MatX &dtau_dq = cache.dtau_dq;
    const MatX &dtau_dv = cache.dtau_dv;
    const MatX &dtau_da = cache.dtau_da;
    const MatX &da_dq   = cache.da_dq;
    const MatX &da_dv   = cache.da_dv;
    const MatX &da_daj  = cache.da_daj;

    MatX dX_full(_dims.nv, 2 * _dims.nv);
    MatX dU_full(_dims.nv, _dims.nu + force_width);

    dX_full.leftCols(_dims.nv)  = dtau_dq + dtau_da * da_dq;
    dX_full.rightCols(_dims.nv) = dtau_dv + dtau_da * da_dv;
    dU_full.leftCols(_dims.nu)  = dtau_da * da_daj;

    if constexpr (WithForces)
    {
        const MatX &da_df = cache.da_df;

        dU_full.rightCols(force_width) = -Jf_full.transpose() + dtau_da * da_df;
    }

    uint offset = FloatingBase ? 6 : 0;

    dtau_dx.resize(_dims.nu, 2 * _dims.nv);
    dtau_du.resize(_dims.nu, u.size());

    for (uint i = 0; i < _dims.nu; ++i)
    {
        uint true_i    = offset + _actuated_joints[i];
        dtau_dx.row(i) = dX_full.row(true_i);
        dtau_du.row(i) = dU_full.row(true_i);
    }
}
}
}
