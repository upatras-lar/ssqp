#include "generators/gradients.hpp"

namespace ssqp
{
namespace tests
{
namespace generators
{
void setup_model(std::shared_ptr<model::Model> &model)
{
    for (uint k = 0; k <= model->dims().horizon; ++k)
    {
        model->addCost<
          model::stagewise::predefined::costs::SquaredJointVelocityCost>(k,
                                                                         1,
                                                                         1);
    }
}

std::shared_ptr<Test> get_explicit_euler_without_forces_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.integration = model::IntegrationScheme::ExplicitEuler;
    model_structure.urdf_path   = "examples/urdf/panda/panda.urdf";
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    setup_model(model);
    model->finalize();

    std::shared_ptr<FDGradDynamics> dyn_test =
      std::make_shared<FDGradDynamics>("ExplicitEulerWithoutForcesGradTest",
                                       model,
                                       0);

    return dyn_test;
}

std::shared_ptr<Test> get_explicit_euler_with_forces_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.integration = model::IntegrationScheme::ExplicitEuler;
    model_structure.urdf_path   = "examples/urdf/a1/a1.urdf";
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
      std::make_shared<FDGradDynamics>("ExplicitEulerWithForcesGradTest",
                                       model,
                                       0);

    return dyn_test;
}

std::shared_ptr<Test> get_semi_implicit_euler_without_forces_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.integration = model::IntegrationScheme::SemiImplicitEuler;
    model_structure.urdf_path   = "examples/urdf/panda/panda.urdf";
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    setup_model(model);
    model->finalize();

    std::shared_ptr<FDGradDynamics> dyn_test =
      std::make_shared<FDGradDynamics>("ExplicitEulerWithoutForcesGradTest",
                                       model,
                                       0);

    return dyn_test;
}

std::shared_ptr<Test> get_semi_implicit_euler_with_forces_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.integration = model::IntegrationScheme::SemiImplicitEuler;
    model_structure.urdf_path   = "examples/urdf/a1/a1.urdf";
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
      std::make_shared<FDGradDynamics>("SemiImplicitEulerWithForcesGradTest",
                                       model,
                                       0);

    return dyn_test;
}

std::shared_ptr<Test> get_innate_constraint_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/urdf/a1/a1.urdf";
    model_structure.free_flyer = true;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    setup_model(model);
    model->finalize();

    std::shared_ptr<FDGradConstraints> constraints_test =
      std::make_shared<FDGradConstraints>("InnateConstraintGradTest", model, 0);

    return constraints_test;
}

std::shared_ptr<Test> get_frame_position_bound_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.ignore_innate_constraints = true;
    model_structure.urdf_path                 = "examples/urdf/a1/a1.urdf";
    model_structure.free_flyer                = true;
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
    model_structure.urdf_path                 = "examples/urdf/a1/a1.urdf";
    model_structure.free_flyer                = true;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);
    setup_model(model);

    model->addConstraints<
      model::stagewise::predefined::constraints::FrameVelocityBound>(
      0,
      *model,
      "trunk",
      Vec3::Zero(),
      Vec3::Zero(),
      19,
      18);

    model->finalize();

    std::shared_ptr<FDGradConstraints> constraints_test =
      std::make_shared<FDGradConstraints>("FrameVelocityBoundGradTest",
                                          model,
                                          0);

    return constraints_test;
}

std::shared_ptr<Test> get_contact_constraint_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.ignore_innate_constraints = true;
    model_structure.urdf_path                 = "examples/urdf/a1/a1.urdf";
    model_structure.free_flyer                = true;
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

std::shared_ptr<Test> get_squared_joint_velocity_cost_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/urdf/a1/a1.urdf";
    model_structure.free_flyer = true;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    for (uint k = 0; k <= model->dims().horizon; ++k)
    {
        model->addCost<
          model::stagewise::predefined::costs::SquaredJointVelocityCost>(k,
                                                                         18,
                                                                         1);
    }

    model->finalize();

    std::shared_ptr<FDGradCost> cost_test =
      std::make_shared<FDGradCost>("SquaredJointVelocityCostGradTest",
                                   model,
                                   0);

    return cost_test;
}

std::shared_ptr<Test> get_squared_control_cost_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/urdf/a1/a1.urdf";
    model_structure.free_flyer = true;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    for (uint k = 0; k < model->dims().horizon; ++k)
    {
        model->addCost<model::stagewise::predefined::costs::SquaredControlCost>(
          k,
          18,
          1);
    }

    model
      ->addCost<model::stagewise::predefined::costs::SquaredJointVelocityCost>(
        2,
        1,
        1);

    model->finalize();

    std::shared_ptr<FDGradCost> cost_test =
      std::make_shared<FDGradCost>("SquaredControlCostGradTest", model, 0);

    return cost_test;
}

std::shared_ptr<Test> get_squared_force_cost_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/urdf/a1/a1.urdf";
    model_structure.free_flyer = true;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    model->addForce(0, "Fl_foot");

    for (uint k = 0; k < model->dims().horizon; ++k)
    {
        model->addCost<model::stagewise::predefined::costs::SquaredForceCost>(
          k,
          18,
          1);
    }

    model
      ->addCost<model::stagewise::predefined::costs::SquaredJointVelocityCost>(
        2,
        1,
        1);

    model->finalize();

    std::shared_ptr<FDGradCost> cost_test =
      std::make_shared<FDGradCost>("SquaredForceCostGradTest", model, 0);

    return cost_test;
}

std::shared_ptr<Test> get_frame_squared_position_error_cost_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/urdf/a1/a1.urdf";
    model_structure.free_flyer = true;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    for (uint k = 0; k <= model->dims().horizon; ++k)
    {
        model->addCost<
          model::stagewise::predefined::costs::FrameSquaredPositionErrorCost>(
          k,
          *model,
          Vec3::Zero(),
          "trunk",
          19,
          18,
          1);
    }

    model->finalize();

    std::shared_ptr<FDGradCost> cost_test =
      std::make_shared<FDGradCost>("FrameSquaredPositionErrorCostGradTest",
                                   model,
                                   0);

    return cost_test;
}

std::shared_ptr<Test> get_frame_squared_rotation_error_cost_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/urdf/a1/a1.urdf";
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
          1);
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
    model_structure.urdf_path  = "examples/urdf/a1/a1.urdf";
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
          1,
          1);
    }

    model->finalize();

    std::shared_ptr<FDGradCost> cost_test =
      std::make_shared<FDGradCost>("FrameSquaredPoseErrorCostGradTest",
                                   model,
                                   0);

    return cost_test;
}

std::shared_ptr<Test> get_frame_squared_velocity_cost_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/urdf/a1/a1.urdf";
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
          1);
    }

    model->finalize();

    std::shared_ptr<FDGradCost> cost_test =
      std::make_shared<FDGradCost>("FrameSquaredVelocityCostGradTest",
                                   model,
                                   0);

    return cost_test;
}

}
}
}