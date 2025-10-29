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

    std::shared_ptr<stagewise::StageDynamics> dynamics = nullptr;

    if (init_states.empty())
    {
        dynamics =
          stagewise::factory::make_dynamics(model.nq + model.nv,
                                            model.nv,
                                            model.nq + model.nv,
                                            2 * model.nv,
                                            VecX::Zero(model.nq + model.nv),
                                            ops,
                                            dyn_eval,
                                            dyn_grad);
    }
    else if (init_states.size() == 1)
    {
        dynamics = stagewise::factory::make_dynamics(model.nq + model.nv,
                                                     model.nv,
                                                     model.nq + model.nv,
                                                     2 * model.nv,
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
    VecX lb(model.nq + 2 * model.nv), ub(model.nq + 2 * model.nv);
    lb << model.lowerPositionLimit, -model.velocityLimit, -model.effortLimit;
    ub << model.upperPositionLimit, model.velocityLimit, model.effortLimit;

    EvaluationFunction innate_constraint_eval =
      [model](const VecX &x, const VecX &u)
    {
        VecX c(3 * model.nv);
        c.head(model.nv)              = x.head(model.nv);
        c.segment(model.nv, model.nv) = x.tail(model.nv);
        c.tail(model.nv)              = u;

        return c;
    };

    multiGradientFunction innate_constraint_grad =
      [model](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
    {
        dX = MatX::Identity(3 * model.nv, 2 * model.nv);

        dU = MatX(3 * model.nv, model.nv);
        dU.block(0, 0, 2 * model.nv, model.nv).setZero();
        dU.block(2 * model.nv, 0, model.nv, model.nv) =
          MatX::Identity(model.nv, model.nv);
    };

    std::shared_ptr<stagewise::StageConstraints> innate_constraints =
      stagewise::factory::make_constraints(3 * model.nv,
                                           lb,
                                           ub,
                                           innate_constraint_eval,
                                           innate_constraint_grad);

    EvaluationFunction terminal_innate_constraint_eval =
      [model](const VecX &x, const VecX &u)
    {
        VecX c(2 * model.nv);
        c.head(model.nv) = x.head(model.nv);
        c.tail(model.nv) = x.tail(model.nv);

        return c;
    };

    multiGradientFunction terminal_innate_constraint_grad =
      [model](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
    {
        dX = MatX::Identity(2 * model.nv, 2 * model.nv);
        dU = MatX::Zero(1, 1);
    };

    std::shared_ptr<stagewise::StageConstraints> terminal_innate_constraints =
      stagewise::factory::make_constraints(2 * model.nv,
                                           lb.head(2 * model.nv),
                                           ub.head(2 * model.nv),
                                           terminal_innate_constraint_eval,
                                           terminal_innate_constraint_grad);

    for (uint k = 0; k <= horizon; ++k)
    {
        _dynamics[k] =
          dynamics != nullptr
            ? dynamics
            : stagewise::factory::make_dynamics(model.nq + model.nv,
                                                model.nv,
                                                model.nq + model.nv,
                                                2 * model.nv,
                                                init_states[k],
                                                ops,
                                                dyn_eval,
                                                dyn_grad);
        _constraints[k] = innate_constraints;
    }

    _constraints[horizon] = terminal_innate_constraints;

    _forwardKinematics = [model](const VecX &x, const std::string &frame)
    {
        pinocchio::Data data(model);

        pinocchio::forwardKinematics(model, data, x);

        int frame_id = model.getFrameId(frame);
        pinocchio::updateFramePlacement(model, data, frame_id);

        return std::make_pair<VecX, MatX>(data.oMf[frame_id].translation(),
                                          data.oMf[frame_id].rotation());
    };

    _inverseKinematics = [model](const VecX &desired_translation,
                                 const MatX &desired_rotation,
                                 const std::string &joint)
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
    };

    _frameJacobian = [model](const VecX &x, const std::string &frame)
    {
        pinocchio::Data data(model);

        int frame_id = model.getFrameId(frame);

        pinocchio::forwardKinematics(model, data, x);
        pinocchio::updateFramePlacements(model, data);

        VecX p_ee = data.oMf[frame_id].translation();

        pinocchio::computeJointJacobians(model, data, x);

        MatX J6(6, model.nv);
        getFrameJacobian(model,
                         data,
                         frame_id,
                         pinocchio::ReferenceFrame::WORLD,
                         J6);

        return J6;
    };
}

void Model::finalize(
  const std::shared_ptr<stagewise::StageCost> &cost,
  const std::shared_ptr<stagewise::StageCost> &terminal_cost,
  const std::vector<std::shared_ptr<stagewise::StageConstraints>> &constraints,
  const std::vector<VecX>                                         &init_states)
{
    finalize(
      std::vector<std::shared_ptr<stagewise::StageCost>>(_dynamics.size(),
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

    assert(costs.size() == horizon + 1 && "Costs size mismatch");

    bool extra_constraints = !constraints.empty();
    if (extra_constraints)
    {
        assert(constraints.size() == horizon + 1 &&
               "Constraints size mismatch");
    }

    if (extra_constraints)
    {
        for (uint k = 0; k <= horizon; ++k)
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
        for (uint k = 0; k <= horizon; ++k)
        {
            _costs[k] = costs[k];
        }
        _costs[horizon] = terminal_cost;
    }

    _setup();
}

}
}