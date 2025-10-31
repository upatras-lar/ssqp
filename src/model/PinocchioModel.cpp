#include "model/Model.hpp"

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/aba-derivatives.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>

namespace ssqp
{
namespace model
{
struct Model::ModelImpl
{
    pinocchio::Model model;
    pinocchio::Data  data;
};

Model::Model(
  const std::string &urdf_path, const uint horizon, const double dt,
  const std::shared_ptr<stagewise::StageCost> &cost,
  const std::shared_ptr<stagewise::StageCost> &terminal_cost,
  const std::vector<std::shared_ptr<stagewise::StageConstraints>> &constraints,
  const std::vector<VecX>                                         &init_states)
  : Model(urdf_path, horizon, dt, init_states)
{
    finalize(cost, terminal_cost, constraints);
}

Model::Model(
  const std::string &urdf_path, const uint horizon, const double dt,
  const std::vector<std::shared_ptr<stagewise::StageCost>> &costs,
  const std::shared_ptr<stagewise::StageCost>              &terminal_cost,
  const std::vector<std::shared_ptr<stagewise::StageConstraints>> &constraints,
  const std::vector<VecX>                                         &init_states)
  : Model(urdf_path, horizon, dt, init_states)
{
    finalize(costs, terminal_cost, constraints);
}

Model::Model(const std::string &urdf_path, const uint horizon, const double dt,
             const std::vector<VecX> &init_states)
  : _impl(std::make_shared<ModelImpl>())
  , _costs(horizon + 1)
  , _dynamics(horizon + 1)
  , _constraints(horizon + 1)
{
    pinocchio::urdf::buildModel(urdf_path, _impl->model);
    _impl->data = pinocchio::Data(_impl->model);

    auto *impl = _impl.get();

    ManifoldOperators ops;
    ops.right_plus = [impl](const VecX &x, const VecX &dx)
    {
        VecX out(impl->model.nq + impl->model.nv);

        out.head(impl->model.nq) =
          pinocchio::integrate(impl->model,
                               x.head(impl->model.nq),
                               dx.head(impl->model.nv));
        out.tail(impl->model.nv) =
          x.tail(impl->model.nv) + dx.tail(impl->model.nv);

        return out;
    };

    ops.right_minus = [impl](const VecX &x, const VecX &y)
    {
        VecX out(2 * impl->model.nv);

        out.head(impl->model.nv) =
          pinocchio::difference(impl->model,
                                y.head(impl->model.nq),
                                x.head(impl->model.nq));
        out.tail(impl->model.nv) =
          x.tail(impl->model.nv) - y.tail(impl->model.nv);

        return out;
    };

    auto dyn_eval = [impl, dt](const VecX &x, const VecX &u)
    {
        pinocchio::aba(impl->model,
                       impl->data,
                       x.head(impl->model.nq),
                       x.tail(impl->model.nv),
                       u);

        VecX out(impl->model.nq + impl->model.nv);
        out.head(impl->model.nq) =
          pinocchio::integrate(impl->model,
                               x.head(impl->model.nq),
                               x.tail(impl->model.nv) * dt);
        out.tail(impl->model.nv) = x.tail(impl->model.nv) + impl->data.ddq * dt;

        return out;
    };

    auto dyn_grad = [impl, dt](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
    {
        VecX q   = x.head(impl->model.nq);
        VecX v   = x.tail(impl->model.nv);
        VecX tau = u;

        pinocchio::computeABADerivatives(impl->model, impl->data, q, v, tau);

        MatX Jint_q = MatX::Zero(impl->model.nv, impl->model.nv);
        MatX Jint_v = MatX::Zero(impl->model.nv, impl->model.nv);

        pinocchio::dIntegrate(impl->model,
                              q,
                              v * dt,
                              Jint_q,
                              pinocchio::ArgumentPosition::ARG0);

        pinocchio::dIntegrate(impl->model,
                              q,
                              v * dt,
                              Jint_v,
                              pinocchio::ArgumentPosition::ARG1);

        dX = MatX::Zero(2 * impl->model.nv, 2 * impl->model.nv);
        dX.topLeftCorner(impl->model.nv, impl->model.nv)  = Jint_q;
        dX.topRightCorner(impl->model.nv, impl->model.nv) = Jint_v * dt;
        dX.bottomLeftCorner(impl->model.nv, impl->model.nv) =
          impl->data.ddq_dq * dt;
        dX.bottomRightCorner(impl->model.nv, impl->model.nv) =
          MatX::Identity(impl->model.nv, impl->model.nv) +
          impl->data.ddq_dv * dt;

        dU = MatX::Zero(2 * impl->model.nv, impl->model.nv);
        dU.bottomRows(impl->model.nv) = impl->data.Minv * dt;
    };

    std::shared_ptr<stagewise::StageDynamics> dynamics = nullptr;

    if (init_states.empty())
    {
        dynamics = stagewise::factory::make_dynamics(
          impl->model.nq + impl->model.nv,
          impl->model.nv,
          impl->model.nq + impl->model.nv,
          2 * impl->model.nv,
          VecX::Zero(impl->model.nq + impl->model.nv),
          ops,
          dyn_eval,
          dyn_grad);
    }
    else if (init_states.size() == 1)
    {
        dynamics =
          stagewise::factory::make_dynamics(impl->model.nq + impl->model.nv,
                                            impl->model.nv,
                                            impl->model.nq + impl->model.nv,
                                            2 * impl->model.nv,
                                            init_states[0],
                                            ops,
                                            dyn_eval,
                                            dyn_grad);
    }
    else
    {
        assert(init_states.size() == horizon &&
               "init_states.size() must match horizon");
    }

    // TODO: linearize bounds
    VecX lb(impl->model.nq + 2 * impl->model.nv),
      ub(impl->model.nq + 2 * impl->model.nv);
    lb << impl->model.lowerPositionLimit, -impl->model.velocityLimit,
      -impl->model.effortLimit;
    ub << impl->model.upperPositionLimit, impl->model.velocityLimit,
      impl->model.effortLimit;

    auto innate_constraint_eval = [impl](const VecX &x, const VecX &u)
    {
        VecX c(3 * impl->model.nv);
        c.head(impl->model.nv)                    = x.head(impl->model.nv);
        c.segment(impl->model.nv, impl->model.nv) = x.tail(impl->model.nv);
        c.tail(impl->model.nv)                    = u;

        return c;
    };

    auto innate_constraint_grad =
      [impl](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
    {
        dX = MatX::Identity(3 * impl->model.nv, 2 * impl->model.nv);

        dU = MatX(3 * impl->model.nv, impl->model.nv);
        dU.block(0, 0, 2 * impl->model.nv, impl->model.nv).setZero();
        dU.block(2 * impl->model.nv, 0, impl->model.nv, impl->model.nv) =
          MatX::Identity(impl->model.nv, impl->model.nv);
    };

    std::shared_ptr<stagewise::StageConstraints> innate_constraints =
      stagewise::factory::make_constraints(3 * impl->model.nv,
                                           lb,
                                           ub,
                                           innate_constraint_eval,
                                           innate_constraint_grad);

    auto terminal_innate_constraint_eval = [impl](const VecX &x, const VecX &u)
    {
        VecX c(2 * impl->model.nv);
        c.head(impl->model.nv) = x.head(impl->model.nv);
        c.tail(impl->model.nv) = x.tail(impl->model.nv);

        return c;
    };

    auto terminal_innate_constraint_grad =
      [impl](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
    {
        dX = MatX::Identity(2 * impl->model.nv, 2 * impl->model.nv);
        dU = MatX::Zero(1, 1);
    };

    std::shared_ptr<stagewise::StageConstraints> terminal_innate_constraints =
      stagewise::factory::make_constraints(2 * impl->model.nv,
                                           lb.head(2 * impl->model.nv),
                                           ub.head(2 * impl->model.nv),
                                           terminal_innate_constraint_eval,
                                           terminal_innate_constraint_grad);

    for (uint k = 0; k <= horizon; ++k)
    {
        _dynamics[k] =
          dynamics != nullptr
            ? dynamics
            : stagewise::factory::make_dynamics(impl->model.nq + impl->model.nv,
                                                impl->model.nv,
                                                impl->model.nq + impl->model.nv,
                                                2 * impl->model.nv,
                                                init_states[k],
                                                ops,
                                                dyn_eval,
                                                dyn_grad);
        _constraints[k] = innate_constraints;
    }

    _constraints[horizon] = terminal_innate_constraints;
}

const std::pair<VecX, MatX> Model::forwardKinematics(
  const VecX &x, const std::string &frame) const
{
    pinocchio::forwardKinematics(_impl->model, _impl->data, x);

    int frame_id = _impl->model.getFrameId(frame);
    pinocchio::updateFramePlacement(_impl->model, _impl->data, frame_id);

    return std::make_pair<VecX, MatX>(_impl->data.oMf[frame_id].translation(),
                                      _impl->data.oMf[frame_id].rotation());
}

/*
const VecX Model::inverseKinematics(const VecX        &desired_translation,
                                    const MatX        &desired_rotation,
                                    const std::string &joint) const
{
    static double eps      = 1e-4;
    static uint   MAX_ITER = 1000;
    static double DT       = 1e-1;
    static double damp     = 1e-12;

    pinocchio::Data data(model);

    int            joint_id = model.getJointId(joint);
    pinocchio::SE3 oMdes =
      pinocchio::SE3(static_cast<Eigen::Matrix3d>(desired_rotation),
                     static_cast<Eigen::Vector3d>(desired_translation));

    VecX q = pinocchio::neutral(model);

    bool success = false;
    for (uint i = 0; i < MAX_ITER; ++i)
    {
        pinocchio::forwardKinematics(model, data, q);
        pinocchio::updateFramePlacements(model, data);

        pinocchio::SE3 dMi = oMdes.actInv(data.oMi[joint_id]);

        VecX err = pinocchio::log6(dMi).toVector();

        if (err.norm() < eps)
        {
            success = true;
            break;
        }

        MatX J(6, model.nv);
        pinocchio::computeJointJacobian(model, data, q, joint_id, J);

        MatX JJt = J * J.transpose();
        VecX v   = -J.transpose() *
                 (JJt + damp * MatX::Identity(6, 6)).ldlt().solve(err);

        q = pinocchio::integrate(model, q, v * DT);

        for (int j = 0; j < model.nq; ++j)
        {
            q[j] = std::min(std::max(q[j], model.lowerPositionLimit[j]),
                            model.upperPositionLimit[j]);
        }
    }

    if (!success)
    {
        LOGLN("Warning: IK failed to converge");
        LOGLN("Desired translation: " << desired_translation.transpose());
        LOGLN("Actual: " << data.oMi[joint_id].translation().transpose());
    }

    return q;
}
*/

const MatX Model::frameJacobian(const VecX &x, const std::string &frame) const
{
    int frame_id = _impl->model.getFrameId(frame);

    pinocchio::forwardKinematics(_impl->model, _impl->data, x);
    pinocchio::updateFramePlacements(_impl->model, _impl->data);

    VecX p_ee = _impl->data.oMf[frame_id].translation();

    pinocchio::computeJointJacobians(_impl->model, _impl->data, x);

    MatX J6(6, _impl->model.nv);
    pinocchio::getFrameJacobian(_impl->model,
                                _impl->data,
                                frame_id,
                                pinocchio::ReferenceFrame::WORLD,
                                J6);

    return J6;
}

void Model::finalize(
  const std::shared_ptr<stagewise::StageCost> &cost,
  const std::shared_ptr<stagewise::StageCost> &terminal_cost,
  const std::vector<std::shared_ptr<stagewise::StageConstraints>> &constraints,
  const std::vector<VecX>                                         &init_states)
{
    finalize(
      std::vector<std::shared_ptr<stagewise::StageCost>>(_dynamics.size() - 1,
                                                         cost),
      terminal_cost,
      constraints,
      init_states);
}

void Model::finalize(
  const std::vector<std::shared_ptr<stagewise::StageCost>> &costs,
  const std::shared_ptr<stagewise::StageCost>              &terminal_cost,
  const std::vector<std::shared_ptr<stagewise::StageConstraints>> &constraints,
  const std::vector<VecX>                                         &init_states)
{
    int horizon = _dynamics.size() - 1;

    if (!init_states.empty())
    {
        if (init_states.size() == 1)
        {
            for (uint k = 0; k < horizon; ++k)
            {
                _dynamics[k]->set_init_state(init_states[0]);
            }
        }
        else
        {
            assert(init_states.size() == horizon &&
                   "init_states.size() must match horizon");
            for (uint k = 0; k < horizon; ++k)
            {
                _dynamics[k]->set_init_state(init_states[k]);
            }
        }
    }

    assert(costs.size() == horizon && "Costs size mismatch");

    bool extra_constraints = !constraints.empty();
    if (extra_constraints)
    {
        assert(constraints.size() == horizon + 1 &&
               "Constraints size mismatch");
    }

    if (extra_constraints)
    {
        for (uint k = 0; k < horizon; ++k)
        {
            _costs[k] = costs[k];
            if (constraints[k] != nullptr)
            {
                _constraints[k] =
                  stagewise::factory::merge_constraints(_constraints[k],
                                                        constraints[k]);
            }
        }
        _costs[horizon] = terminal_cost;
        if (constraints[horizon] != nullptr)
        {
            _constraints[horizon] =
              stagewise::factory::merge_constraints(_constraints[horizon],
                                                    constraints[horizon]);
        }
    }
    else
    {
        for (uint k = 0; k < horizon; ++k)
        {
            _costs[k] = costs[k];
        }
        _costs[horizon] = terminal_cost;
    }

    _setup();
}

}
}