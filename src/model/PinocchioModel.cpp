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
Model::Model(const std::string &urdf_path, const uint horizon, const double dt)
  : _costs(horizon + 1)
  , _dynamics(horizon + 1)
  , _constraints(horizon + 1)
{
    pinocchio::Model model;
    pinocchio::urdf::buildModel(urdf_path, model);
    // model.gravity.linear().setZero();

    EvaluationFunction cost_eval = [](const VecX &x, const VecX &u)
    {
        return (Eigen::Vector<double, 1>() << 0.1 * u.transpose() * u)
          .finished();
    };

    GradientFunction cost_grad =
      [model](const VecX &x, const VecX &u, VecX &dx, VecX &du)
    {
        dx = VecX::Zero(2 * model.nv);
        du = 0.2 * u;
    };

    HessianFunction cost_hess_uu = [](const VecX &x, const VecX &u, MatX &Huu)
    { Huu = 0.2 * MatX::Identity(u.rows(), u.rows()); };

    std::shared_ptr<const stagewise::StageCost> cost =
      stagewise::factory::make_cost(cost_eval,
                                    cost_grad,
                                    ZeroHessian_xx,
                                    ZeroHessian_xu,
                                    cost_hess_uu);

    VecX target(3);
    target << 1, 1, 1;
    EvaluationFunction term_cost_eval = [target](const VecX &x, const VecX &u)
    {
        return (Eigen::Vector<double, 1>()
                << 0.1 * u.transpose() * u +
                     100 * (x.head(3) - target).squaredNorm())
          .finished();
    };

    GradientFunction term_cost_grad =
      [model, target](const VecX &x, const VecX &u, VecX &dx, VecX &du)
    {
        dx         = VecX::Zero(2 * model.nv);
        dx.head(3) = 200 * (x.head(3) - target);

        du = 0.2 * u;
    };

    std::shared_ptr<const stagewise::StageCost> term_cost =
      stagewise::factory::make_cost(term_cost_eval, term_cost_grad);

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

    for (uint k = 0; k <= horizon; ++k)
    {
        _costs[k]    = cost;
        _dynamics[k] = dynamics;
    }

    //_costs[horizon] = term_cost;

    _setup();
}

}
}