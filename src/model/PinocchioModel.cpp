#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/jacobian.hpp>

#include "model/Model.hpp"

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
    pinocchio::Data data(model);

    EvaluationFunction cost_eval = [](const VecX &x, const VecX &u)
    {
        return (Eigen::Vector<double, 1>() << 0.1 * u.transpose() * u)
          .finished();
    };

    GradientFunction cost_grad =
      [](const VecX &x, const VecX &u, VecX &dx, VecX &du)
    {
        dx = VecX::Zero(u.rows());
        du = 0.2 * u;
    };

    HessianFunction cost_hess_xx = [](const VecX &x, const VecX &u, MatX &Hxx)
    { Hxx = MatX::Zero(u.rows(), u.rows()); };

    HessianFunction cost_hess_xu = [](const VecX &x, const VecX &u, MatX &Hxu)
    { Hxu = MatX::Zero(u.rows(), u.rows()); };

    HessianFunction cost_hess_uu = [](const VecX &x, const VecX &u, MatX &Huu)
    { Huu = 0.2 * MatX::Identity(u.rows(), u.rows()); };

    std::shared_ptr<const stagewise::StageCost> cost =
      stagewise::factory::make_cost(cost_eval,
                                    cost_grad,
                                    cost_hess_xx,
                                    cost_hess_xu,
                                    cost_hess_uu);

    VecX target(3);
    target << 1, 1, 1;
    EvaluationFunction term_cost_eval = [target](const VecX &x, const VecX &u)
    {
        return (Eigen::Vector<double, 1>()
                << 0.1 * u.transpose() * u + (x.head(3) - target).norm())
          .finished();
    };

    GradientFunction term_cost_grad =
      [](const VecX &x, const VecX &u, VecX &dx, VecX &du)
    {
        dx         = VecX::Zero(u.rows());
        dx.head(3) = VecX::Ones(3);

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
          pinocchio::difference(model, x.head(model.nq), y.head(model.nq));
        out.tail(model.nv) = x.tail(model.nv) - y.tail(model.nv);

        return out;
    };

    pinocchio::Data   *data_ptr = &data;
    EvaluationFunction dyn_eval =
      [model, data_ptr, dt](const VecX &x, const VecX &u)
    {
        pinocchio::aba(model,
                       *data_ptr,
                       x.head(model.nq),
                       x.tail(model.nv),
                       u.head(model.nv));

        VecX out(model.nq + model.nv);
        out.head(model.nq) =
          pinocchio::integrate(model, x.head(model.nq), data_ptr->ddq * dt);
        out.tail(model.nv) = x.tail(model.nv) + u.tail(model.nv) * dt;

        return out;
    };

    multiGradientFunction dyn_grad =
      [model, data, dt](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
    {
        pinocchio::dIntegrate(model,
                              x.head(model.nq),
                              x.tail(model.nv) * dt,
                              dX,
                              pinocchio::ArgumentPosition::ARG0);

        pinocchio::dIntegrate(model,
                              x.head(model.nq),
                              x.tail(model.nv) * dt,
                              dU,
                              pinocchio::ArgumentPosition::ARG1);
    };

    std::shared_ptr<const stagewise::StageDynamics> dynamics =
      stagewise::factory::make_dynamics(model.nq + model.nv,
                                        model.nv,
                                        model.nq + model.nv,
                                        model.nv,
                                        ops,
                                        dyn_eval,
                                        dyn_grad);

    for (uint k = 0; k <= horizon; ++k)
    {
        _costs[k]    = cost;
        _dynamics[k] = dynamics;
    }

    _costs[horizon] = term_cost;

    _setup();
}

}
}