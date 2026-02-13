#include "generators/gradients.hpp"

namespace ssqp
{
namespace tests
{
namespace generators
{
void setup_model(std::shared_ptr<model::Model> &model)
{
    model::Dimensions dims = model->dims();
    for (uint k = 0; k <= dims.horizon; ++k)
    {
        model->addCost<
          model::stagewise::predefined::costs::SquaredJointVelocityCost>(
          k,
          dims.nv[k],
          1,
          model->floatingBase());
    }
}

std::shared_ptr<Test> get_fwdyn_explicit_euler_without_forces_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.integration = model::IntegrationScheme::ExplicitEuler;
    model_structure.urdf_path   = "examples/assets/urdf/panda/panda.urdf";
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    setup_model(model);
    model->finalize();

    std::shared_ptr<FDGradDynamics> dyn_test = std::make_shared<FDGradDynamics>(
      "FwdynExplicitEulerWithoutForcesGradTest",
      model,
      0);

    return dyn_test;
}

std::shared_ptr<Test> get_fwdyn_explicit_euler_with_forces_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.integration = model::IntegrationScheme::ExplicitEuler;
    model_structure.urdf_path   = "examples/assets/urdf/a1/a1.urdf";
    model_structure.free_flyer  = true;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    setup_model(model);

    model->addForce(0, "FL_foot");
    model->addForce(0, "FR_foot");
    model->addForce(0, "RL_foot");
    model->addForce(0, "RR_foot");
    model->finalize();

    std::shared_ptr<FDGradDynamics> dyn_test =
      std::make_shared<FDGradDynamics>("FwdynExplicitEulerWithForcesGradTest",
                                       model,
                                       0);

    return dyn_test;
}

std::shared_ptr<Test> get_fwdyn_semi_implicit_euler_without_forces_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.integration = model::IntegrationScheme::SemiImplicitEuler;
    model_structure.urdf_path   = "examples/assets/urdf/panda/panda.urdf";
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    setup_model(model);
    model->finalize();

    std::shared_ptr<FDGradDynamics> dyn_test = std::make_shared<FDGradDynamics>(
      "FwdynExplicitEulerWithoutForcesGradTest",
      model,
      0);

    return dyn_test;
}

std::shared_ptr<Test> get_fwdyn_semi_implicit_euler_with_forces_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.integration = model::IntegrationScheme::SemiImplicitEuler;
    model_structure.urdf_path   = "examples/assets/urdf/a1/a1.urdf";
    model_structure.free_flyer  = true;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    setup_model(model);

    model->addForce(0, "FL_foot");
    model->addForce(0, "FR_foot");
    model->addForce(0, "RL_foot");
    model->addForce(0, "RR_foot");
    model->finalize();

    std::shared_ptr<FDGradDynamics> dyn_test = std::make_shared<FDGradDynamics>(
      "FwdynSemiImplicitEulerWithForcesGradTest",
      model,
      0);

    return dyn_test;
}

std::shared_ptr<Test> get_invdyn_explicit_euler_without_forces_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.integration   = model::IntegrationScheme::ExplicitEuler;
    model_structure.urdf_path     = "examples/assets/urdf/panda/panda.urdf";
    model_structure.dynamics_mode = model::DynamicsMode::Inverse;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    setup_model(model);
    model->finalize();

    std::shared_ptr<FDGradDynamics> dyn_test = std::make_shared<FDGradDynamics>(
      "InvdynExplicitEulerWithoutForcesGradTest",
      model,
      0);

    return dyn_test;
}

std::shared_ptr<Test> get_invdyn_explicit_euler_with_forces_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.integration   = model::IntegrationScheme::ExplicitEuler;
    model_structure.urdf_path     = "examples/assets/urdf/a1/a1.urdf";
    model_structure.free_flyer    = true;
    model_structure.dynamics_mode = model::DynamicsMode::Inverse;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    setup_model(model);

    model->addForce(0, "FL_foot");
    model->addForce(0, "FR_foot");
    model->addForce(0, "RL_foot");
    model->addForce(0, "RR_foot");
    model->finalize();

    std::shared_ptr<FDGradDynamics> dyn_test =
      std::make_shared<FDGradDynamics>("InvdynExplicitEulerWithForcesGradTest",
                                       model,
                                       0);

    return dyn_test;
}

std::shared_ptr<Test> get_invdyn_semi_implicit_euler_without_forces_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.integration   = model::IntegrationScheme::SemiImplicitEuler;
    model_structure.urdf_path     = "examples/assets/urdf/panda/panda.urdf";
    model_structure.dynamics_mode = model::DynamicsMode::Inverse;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    setup_model(model);
    model->finalize();

    std::shared_ptr<FDGradDynamics> dyn_test = std::make_shared<FDGradDynamics>(
      "InvdynExplicitEulerWithoutForcesGradTest",
      model,
      0);

    return dyn_test;
}

std::shared_ptr<Test> get_invdyn_semi_implicit_euler_with_forces_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.integration   = model::IntegrationScheme::SemiImplicitEuler;
    model_structure.urdf_path     = "examples/assets/urdf/a1/a1.urdf";
    model_structure.free_flyer    = true;
    model_structure.dynamics_mode = model::DynamicsMode::Inverse;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    setup_model(model);

    model->addForce(0, "FL_foot");
    model->addForce(0, "FR_foot");
    model->addForce(0, "RL_foot");
    model->addForce(0, "RR_foot");
    model->finalize();

    std::shared_ptr<FDGradDynamics> dyn_test = std::make_shared<FDGradDynamics>(
      "InvdynSemiImplicitEulerWithForcesGradTest",
      model,
      0);

    return dyn_test;
}

std::shared_ptr<Test> get_fwdyn_innate_constraint_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/assets/urdf/a1/a1.urdf";
    model_structure.free_flyer = true;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    setup_model(model);
    model->finalize();

    std::shared_ptr<FDGradConstraints> constraints_test =
      std::make_shared<FDGradConstraints>("FwdynInnateConstraintGradTest",
                                          model,
                                          0);

    return constraints_test;
}

std::shared_ptr<Test> get_invdyn_innate_constraint_without_forces_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.urdf_path     = "examples/assets/urdf/panda/panda.urdf";
    model_structure.dynamics_mode = model::DynamicsMode::Inverse;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    setup_model(model);
    model->finalize();

    std::shared_ptr<FDGradConstraints> constraints_test =
      std::make_shared<FDGradConstraints>(
        "InvdynInnateConstraintWithoutForcesGradTest",
        model,
        0);

    return constraints_test;
}

std::shared_ptr<Test> get_invdyn_innate_constraint_with_forces_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.urdf_path     = "examples/assets/urdf/a1/a1.urdf";
    model_structure.free_flyer    = true;
    model_structure.dynamics_mode = model::DynamicsMode::Inverse;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    model->addForce(0, "FL_foot");

    setup_model(model);
    model->finalize();

    std::shared_ptr<FDGradConstraints> constraints_test =
      std::make_shared<FDGradConstraints>(
        "InvdynInnateConstraintWithForcesGradTest",
        model,
        0);

    return constraints_test;
}

std::shared_ptr<Test> get_frame_position_bound_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.ignore_innate_constraints = true;
    model_structure.urdf_path  = "examples/assets/urdf/a1/a1.urdf";
    model_structure.free_flyer = true;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);
    setup_model(model);

    model->addConstraints<
      model::stagewise::predefined::constraints::FramePositionBound>(
      0,
      *model,
      "trunk",
      Vec3::Zero(),
      Vec3::Zero(),
      19,
      18);

    model->finalize();

    std::shared_ptr<FDGradConstraints> constraints_test =
      std::make_shared<FDGradConstraints>("FramePositionBoundGradTest",
                                          model,
                                          0);

    return constraints_test;
}

std::shared_ptr<Test> get_frame_velocity_bound_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.ignore_innate_constraints = true;
    model_structure.urdf_path  = "examples/assets/urdf/a1/a1.urdf";
    model_structure.free_flyer = true;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);
    setup_model(model);

    model->addConstraints<
      model::stagewise::predefined::constraints::FrameLinearVelocityBound>(
      0,
      *model,
      "trunk",
      Vec3::Zero(),
      Vec3::Zero(),
      19,
      18);

    model->finalize();

    std::shared_ptr<FDGradConstraints> constraints_test =
      std::make_shared<FDGradConstraints>("FrameLinearVelocityBoundGradTest",
                                          model,
                                          0);

    return constraints_test;
}

std::shared_ptr<Test> get_contact_constraint_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.ignore_innate_constraints = true;
    model_structure.urdf_path  = "examples/assets/urdf/a1/a1.urdf";
    model_structure.free_flyer = true;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);
    setup_model(model);

    model->addConstraints<
      model::stagewise::predefined::constraints::ContactConstraint>(0,
                                                                    *model,
                                                                    "FL_foot",
                                                                    19,
                                                                    18,
                                                                    0.6);

    model->finalize();

    std::shared_ptr<FDGradConstraints> constraints_test =
      std::make_shared<FDGradConstraints>("ContactConstraintGradTest",
                                          model,
                                          0);

    return constraints_test;
}

std::shared_ptr<Test> get_squared_joint_angle_cost_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.urdf_path = "examples/assets/urdf/panda/panda.urdf";
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    for (uint k = 0; k <= model->dims().horizon; ++k)
    {
        model
          ->addCost<model::stagewise::predefined::costs::SquaredJointAngleCost>(
            k,
            9,
            9,
            9.94);
    }

    model->finalize();

    std::shared_ptr<FDGradCost> cost_test =
      std::make_shared<FDGradCost>("SquaredJointAngleCostGradTest", model, 0);

    return cost_test;
}

std::shared_ptr<Test> get_squared_joint_angle_cost_floating_base_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/assets/urdf/a1/a1.urdf";
    model_structure.free_flyer = true;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    for (uint k = 0; k <= model->dims().horizon; ++k)
    {
        model
          ->addCost<model::stagewise::predefined::costs::SquaredJointAngleCost>(
            k,
            19,
            18,
            0.58,
            true);
    }

    model->finalize();

    std::shared_ptr<FDGradCost> cost_test =
      std::make_shared<FDGradCost>("SquaredJointAngleCostFloatingBaseGradTest",
                                   model,
                                   0);

    return cost_test;
}

std::shared_ptr<Test> get_squared_joint_velocity_cost_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.urdf_path = "examples/assets/urdf/panda/panda.urdf";
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    for (uint k = 0; k <= model->dims().horizon; ++k)
    {
        model->addCost<
          model::stagewise::predefined::costs::SquaredJointVelocityCost>(k,
                                                                         9,
                                                                         5.72);
    }

    model->finalize();

    std::shared_ptr<FDGradCost> cost_test =
      std::make_shared<FDGradCost>("SquaredJointVelocityCostGradTest",
                                   model,
                                   0);

    return cost_test;
}

std::shared_ptr<Test> get_squared_joint_velocity_cost_floating_base_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/assets/urdf/a1/a1.urdf";
    model_structure.free_flyer = true;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    for (uint k = 0; k <= model->dims().horizon; ++k)
    {
        model->addCost<
          model::stagewise::predefined::costs::SquaredJointVelocityCost>(k,
                                                                         18,
                                                                         9.43,
                                                                         true);
    }

    model->finalize();

    std::shared_ptr<FDGradCost> cost_test = std::make_shared<FDGradCost>(
      "SquaredJointVelocityCostFloatingBaseGradTest",
      model,
      0);

    return cost_test;
}

std::shared_ptr<Test> get_squared_control_cost_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/assets/urdf/a1/a1.urdf";
    model_structure.free_flyer = true;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    for (uint k = 0; k < model->dims().horizon; ++k)
    {
        model->addCost<model::stagewise::predefined::costs::SquaredControlCost>(
          k,
          18,
          0.47);
    }

    model
      ->addCost<model::stagewise::predefined::costs::SquaredJointVelocityCost>(
        2,
        1,
        1.0);

    model->finalize();

    std::shared_ptr<FDGradCost> cost_test =
      std::make_shared<FDGradCost>("SquaredControlCostGradTest", model, 0);

    return cost_test;
}

std::shared_ptr<Test> get_squared_force_cost_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/assets/urdf/a1/a1.urdf";
    model_structure.free_flyer = true;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    model->addForce(0, "FL_foot");

    for (uint k = 0; k < model->dims().horizon; ++k)
    {
        model->addCost<model::stagewise::predefined::costs::SquaredForceCost>(
          k,
          18,
          9.78);
    }

    model
      ->addCost<model::stagewise::predefined::costs::SquaredJointVelocityCost>(
        2,
        1,
        1.0);

    model->finalize();

    std::shared_ptr<FDGradCost> cost_test =
      std::make_shared<FDGradCost>("SquaredForceCostGradTest", model, 0);

    return cost_test;
}

std::shared_ptr<Test> get_squared_configuration_difference_cost_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.urdf_path = "examples/assets/urdf/panda/panda.urdf";
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    for (uint k = 0; k <= model->dims().horizon; ++k)
    {
        model->addCost<
          model::stagewise::predefined::costs::SquaredConfigurationErrorCost>(
          k,
          VecX::Zero(9),
          9,
          9,
          9.17,
          false);
    }

    model->finalize();

    std::shared_ptr<FDGradCost> cost_test =
      std::make_shared<FDGradCost>("SquaredConfigurationErrorCostGradTest",
                                   model,
                                   0);

    return cost_test;
}

std::shared_ptr<Test>
get_squared_configuration_difference_cost_floating_base_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/assets/urdf/a1/a1.urdf";
    model_structure.free_flyer = true;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    for (uint k = 0; k <= model->dims().horizon; ++k)
    {
        model->addCost<
          model::stagewise::predefined::costs::SquaredConfigurationErrorCost>(
          k,
          VecX::Zero(12),
          19,
          18,
          3.27,
          true);
    }

    model->finalize();

    std::shared_ptr<FDGradCost> cost_test = std::make_shared<FDGradCost>(
      "SquaredConfigurationErrorCostFloatingBaseGradTest",
      model,
      0);

    return cost_test;
}

std::shared_ptr<Test> get_frame_squared_position_error_cost_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/assets/urdf/a1/a1.urdf";
    model_structure.free_flyer = true;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    for (uint k = 0; k <= model->dims().horizon; ++k)
    {
        model->addCost<model::stagewise::predefined::costs::
                         FrameSquaredTranslationErrorCost>(k,
                                                           *model,
                                                           Vec3::Zero(),
                                                           "trunk",
                                                           19,
                                                           18,
                                                           3.58);
    }

    model->finalize();

    std::shared_ptr<FDGradCost> cost_test =
      std::make_shared<FDGradCost>("FrameSquaredTranslationErrorCostGradTest",
                                   model,
                                   0);

    return cost_test;
}

std::shared_ptr<Test> get_frame_squared_rotation_error_cost_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/assets/urdf/a1/a1.urdf";
    model_structure.free_flyer = true;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    for (uint k = 0; k <= model->dims().horizon; ++k)
    {
        model->addCost<
          model::stagewise::predefined::costs::FrameSquaredRotationErrorCost>(
          k,
          *model,
          Mat3::Identity(),
          "trunk",
          19,
          18,
          6.59);
    }

    model->finalize();

    std::shared_ptr<FDGradCost> cost_test =
      std::make_shared<FDGradCost>("FrameSquaredRotationErrorCostGradTest",
                                   model,
                                   0);

    return cost_test;
}

std::shared_ptr<Test> get_frame_squared_pose_error_cost_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/assets/urdf/a1/a1.urdf";
    model_structure.free_flyer = true;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    for (uint k = 0; k <= model->dims().horizon; ++k)
    {
        model->addCost<
          model::stagewise::predefined::costs::FrameSquaredPoseErrorCost>(
          k,
          *model,
          Vec3::Zero(),
          Mat3::Identity(),
          "trunk",
          19,
          18,
          9.41,
          0.86);
    }

    model->finalize();

    std::shared_ptr<FDGradCost> cost_test =
      std::make_shared<FDGradCost>("FrameSquaredPoseErrorCostGradTest",
                                   model,
                                   0);

    return cost_test;
}

std::shared_ptr<Test> get_frame_squared_velocity_error_cost_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/assets/urdf/a1/a1.urdf";
    model_structure.free_flyer = true;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    Vec6 vd(1.0, 1.0, 0.1, 0.5, 0.3, -0.1);

    for (uint k = 0; k <= model->dims().horizon; ++k)
    {
        model->addCost<
          model::stagewise::predefined::costs::FrameSquaredVelocityErrorCost>(
          k,
          *model,
          vd,
          "FR_foot",
          19,
          18,
          5.87);
    }

    model->finalize();

    std::shared_ptr<FDGradCost> cost_test =
      std::make_shared<FDGradCost>("FrameSquaredVelocityErrorCostGradTest",
                                   model,
                                   0);

    return cost_test;
}

std::shared_ptr<Test> get_frame_squared_velocity_cost_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/assets/urdf/a1/a1.urdf";
    model_structure.free_flyer = true;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    for (uint k = 0; k <= model->dims().horizon; ++k)
    {
        model->addCost<
          model::stagewise::predefined::costs::FrameSquaredVelocityCost>(
          k,
          *model,
          "FR_foot",
          19,
          18,
          2.61);
    }

    model->finalize();

    std::shared_ptr<FDGradCost> cost_test =
      std::make_shared<FDGradCost>("FrameSquaredVelocityCostGradTest",
                                   model,
                                   0);

    return cost_test;
}

std::shared_ptr<Test> get_lf_grad_test()
{
    const uint              horizon = 40;
    model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/assets/urdf/go2/go2.urdf";
    model_structure.free_flyer = true;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(horizon, 0.01, model_structure);

    for (uint k = 0; k < horizon; ++k)
    {
        model->addCost<
          model::stagewise::predefined::costs::SquaredJointVelocityCost>(k,
                                                                         18,
                                                                         2.98,
                                                                         true);
        model->addCost<model::stagewise::predefined::costs::SquaredControlCost>(
          k,
          18,
          7.07);
    }

    model
      ->addCost<model::stagewise::predefined::costs::SquaredJointVelocityCost>(
        horizon,
        18,
        6.33,
        true);

    VecX init_state(37);
    init_state[6] = 1.0;

    model::Model::Config config;
    config.init_states = { init_state };
    model->finalize(config);

    std::shared_ptr<FDGradFullTrajectoryCost> lf_test =
      std::make_shared<FDGradFullTrajectoryCost>("lfGradTest", model);

    return lf_test;
}

std::shared_ptr<Test> get_gammaf_grad_test()
{
    const uint              horizon = 40;
    model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/assets/urdf/go2/go2.urdf";
    model_structure.free_flyer = true;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(horizon, 0.01, model_structure);

    for (uint k = 0; k < horizon; ++k)
    {
        model->addCost<
          model::stagewise::predefined::costs::SquaredJointVelocityCost>(k,
                                                                         18,
                                                                         8.13,
                                                                         true);
        model->addCost<model::stagewise::predefined::costs::SquaredControlCost>(
          k,
          18,
          0.66);
    }

    model
      ->addCost<model::stagewise::predefined::costs::SquaredJointVelocityCost>(
        horizon,
        18,
        8.72,
        true);

    VecX init_state(37);
    init_state[6] = 1.0;

    model::Model::Config config;
    config.init_states = { init_state };
    model->finalize(config);

    std::shared_ptr<FDGradFullTrajectoryStageGap> gammaf_test =
      std::make_shared<FDGradFullTrajectoryStageGap>("gammafGradTest",
                                                     model,
                                                     10);

    return gammaf_test;
}

std::shared_ptr<Test> get_cf_grad_test()
{
    const uint              horizon = 40;
    model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/assets/urdf/go2/go2.urdf";
    model_structure.free_flyer = true;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(horizon, 0.01, model_structure);

    Vec3 lb(-std::numeric_limits<double>::infinity(), 0.1, -3.2);
    Vec3 ub(5.2, std::numeric_limits<double>::infinity(), -1.0);

    for (uint k = 0; k < horizon; ++k)
    {
        model->addCost<
          model::stagewise::predefined::costs::SquaredJointVelocityCost>(k,
                                                                         18,
                                                                         8.37,
                                                                         true);
        model->addCost<model::stagewise::predefined::costs::SquaredControlCost>(
          k,
          18,
          7.53);

        model->addConstraints<
          model::stagewise::predefined::constraints::FramePositionBound>(
          k,
          *model.get(),
          "FL_foot",
          lb,
          ub,
          19,
          18);
    }

    model
      ->addCost<model::stagewise::predefined::costs::SquaredJointVelocityCost>(
        horizon,
        18,
        2.16,
        true);

    model->addConstraints<
      model::stagewise::predefined::constraints::FramePositionBound>(
      horizon,
      *model.get(),
      "FL_foot",
      lb,
      ub,
      19,
      18);

    VecX init_state(37);
    init_state[6] = 1.0;

    model::Model::Config config;
    config.init_states = { init_state };
    model->finalize(config);

    std::shared_ptr<FDGradFullTrajectoryConstraintViolation> cf_test =
      std::make_shared<FDGradFullTrajectoryConstraintViolation>("cfGradTest",
                                                                model,
                                                                10);

    return cf_test;
}

}
}
}