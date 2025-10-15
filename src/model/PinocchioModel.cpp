#include "model/Model.hpp"

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/aba-derivatives.hpp>
#include <pinocchio/algorithm/jacobian.hpp>

namespace ssqp
{
namespace model
{
Model::Model(const std::string &urdf_path, const uint horizon, const double dt,
             const std::shared_ptr<const stagewise::StageCost> &cost,
             const std::shared_ptr<const stagewise::StageCost> &terminal_cost)
  : _costs(horizon + 1)
  , _dynamics(horizon + 1)
  , _constraints(horizon + 1)
{
    pinocchio::Model model;
    pinocchio::urdf::buildModel(urdf_path, model);

    ManifoldOperators ops;
    ops.right_plus = [model](const VecX &x, const VecX &dx)
    {
        VecX out(model.nq + model.nv);

        out.head(model.nq) =
          pinocchio::integrate(model, x.head(model.nq), dx.head(model.nv));
        out.tail(model.nv) = x.tail(model.nv) + dx.tail(model.nv);

        return out;
    };

    ops.right_minus = [model](const VecX &x, const VecX &y)
    {
        VecX out(2 * model.nv);

        out.head(model.nv) =
          pinocchio::difference(model, y.head(model.nq), x.head(model.nq));
        out.tail(model.nv) = x.tail(model.nv) - y.tail(model.nv);

        return out;
    };

    EvaluationFunction dyn_eval = [model, dt](const VecX &x, const VecX &u)
    {
        pinocchio::Data data(model);

        pinocchio::aba(model, data, x.head(model.nq), x.tail(model.nv), u);

        VecX out(model.nq + model.nv);
        out.head(model.nq) =
          pinocchio::integrate(model, x.head(model.nq), x.tail(model.nv) * dt);
        out.tail(model.nv) = x.tail(model.nv) + data.ddq * dt;

        return out;
    };

    multiGradientFunction dyn_grad =
      [model, dt](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
    {
        pinocchio::Data data(model);

        VecX q   = x.head(model.nq);
        VecX v   = x.tail(model.nv);
        VecX tau = u;

        pinocchio::computeABADerivatives(model, data, q, v, tau);

        MatX Jint_q = MatX::Zero(model.nv, model.nv);
        MatX Jint_v = MatX::Zero(model.nv, model.nv);

        pinocchio::dIntegrate(model,
                              q,
                              v * dt,
                              Jint_q,
                              pinocchio::ArgumentPosition::ARG0);

        pinocchio::dIntegrate(model,
                              q,
                              v * dt,
                              Jint_v,
                              pinocchio::ArgumentPosition::ARG1);

        dX = MatX::Zero(2 * model.nv, 2 * model.nv);
        dX.topLeftCorner(model.nv, model.nv)    = Jint_q;
        dX.topRightCorner(model.nv, model.nv)   = Jint_v * dt;
        dX.bottomLeftCorner(model.nv, model.nv) = data.ddq_dq * dt;
        dX.bottomRightCorner(model.nv, model.nv) =
          MatX::Identity(model.nv, model.nv) + data.ddq_dv * dt;

        dU                      = MatX::Zero(2 * model.nv, model.nv);
        dU.bottomRows(model.nv) = data.Minv * dt;
    };

    std::shared_ptr<const stagewise::StageDynamics> dynamics =
      stagewise::factory::make_dynamics(model.nq + model.nv,
                                        model.nv,
                                        model.nq + model.nv,
                                        2 * model.nv,
                                        ops,
                                        dyn_eval,
                                        dyn_grad);

    VecX lb(model.nq + 2 * model.nv), ub(model.nq + 2 * model.nv);
    lb << model.lowerPositionLimit, -model.velocityLimit, -model.effortLimit;
    ub << model.upperPositionLimit, model.velocityLimit, model.effortLimit;

    EvaluationFunction constraint_eval = [model](const VecX &x, const VecX &u)
    {
        VecX c(3 * model.nv);
        c.head(model.nv)              = x.head(model.nv);
        c.segment(model.nv, model.nv) = x.tail(model.nv);
        c.tail(model.nv)              = u;

        return c;
    };

    multiGradientFunction constraint_grad =
      [model](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
    {
        dX = MatX::Identity(3 * model.nv, 2 * model.nv);

        dU = MatX(3 * model.nv, model.nv);
        dU.block(0, 0, 2 * model.nv, model.nv).setZero();
        dU.block(2 * model.nv, 0, model.nv, model.nv) =
          MatX::Identity(model.nv, model.nv);
    };

    std::shared_ptr<const stagewise::StageConstraints> constraints =
      stagewise::factory::make_constraints(3 * model.nv,
                                           lb,
                                           ub,
                                           constraint_eval,
                                           constraint_grad);

    EvaluationFunction terminal_constraint_eval =
      [model](const VecX &x, const VecX &u)
    {
        VecX c(2 * model.nv);
        c.head(model.nv) = x.head(model.nv);
        c.tail(model.nv) = x.tail(model.nv);

        return c;
    };

    multiGradientFunction terminal_constraint_grad =
      [model](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
    {
        dX = MatX::Identity(2 * model.nv, 2 * model.nv);
        dU = MatX::Zero(1, 1);
    };

    std::shared_ptr<const stagewise::StageConstraints> terminal_constraints =
      stagewise::factory::make_constraints(2 * model.nv,
                                           lb.head(2 * model.nv),
                                           ub.head(2 * model.nv),
                                           terminal_constraint_eval,
                                           terminal_constraint_grad);

    for (uint k = 0; k <= horizon; ++k)
    {
        _costs[k]       = cost;
        _dynamics[k]    = dynamics;
        _constraints[k] = constraints;
    }

    _costs[horizon]       = terminal_cost;
    _constraints[horizon] = terminal_constraints;

    _setup();
}

}
}