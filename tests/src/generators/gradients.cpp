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
          *model.get(),
          1.0);
    }
}

void add_forces_to_model(std::shared_ptr<model::Model> &model,
                         std::mt19937                  &rng)
{
    const std::vector<std::string> ordered_body_frames =
      model->orderedBodyFrames();

    std::uniform_int_distribution<std::mt19937::result_type> dist_f(
      1,
      ordered_body_frames.size());

    uint total_forces = dist_f(rng);

    for (uint f = 0; f < total_forces; ++f)
    {
        model->addForce(0, ordered_body_frames[dist_f(rng) - 1]);
    }
}

void set_model_unactuated_joints(std::shared_ptr<model::Model> &model,
                                 std::mt19937                  &rng)
{
    std::uniform_int_distribution<std::mt19937::result_type> dist_bin(0, 1);

    const std::vector<std::string> ordered_actuated_joints =
      model->orderedActuatedJoints();

    std::vector<uint> actuated_joints;

    for (uint i = 0; i < ordered_actuated_joints.size(); ++i)
    {
        if (dist_bin(rng))
        {
            actuated_joints.push_back(i);
        }
    }

    model->setActuatedJoints(actuated_joints);
}

std::shared_ptr<Test> generate_dynamics_gradient_test(
  const std::string &urdf_path, const bool floating_base,
  const model::SolverMode &sm, const model::IntegrationScheme &is,
  const bool forces, const bool unactuated_joints, const uint seed)
{
    std::random_device dev;
    std::mt19937       rng(dev());
    rng.seed(seed);

    model::Model::Structure model_structure;
    model_structure.urdf_path   = urdf_path;
    model_structure.free_flyer  = floating_base;
    model_structure.solver_mode = sm;
    model_structure.integration = is;

    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    setup_model(model);

    if (forces)
    {
        add_forces_to_model(model, rng);
    }

    if (unactuated_joints)
    {
        set_model_unactuated_joints(model, rng);
    }

    model->finalize();

    std::stringstream test_name_ss;

    switch (sm)
    {
    default:
    case model::SolverMode::Kinematics:
        test_name_ss << "Kin";
        break;
    case model::SolverMode::ForwardDynamics:
        test_name_ss << "Fwdyn";
        break;
    case model::SolverMode::InverseDynamics:
        test_name_ss << "Invdyn";
        break;
    case model::SolverMode::ReducedInverseDynamics:
        test_name_ss << "ReducedInvdyn";
        break;
    }

    if (sm != model::SolverMode::Kinematics)
    {
        switch (is)
        {
        default:
        case model::IntegrationScheme::SemiImplicitEuler:
            test_name_ss << "SemiImplicitEuler";
            break;
        case model::IntegrationScheme::ExplicitEuler:
            test_name_ss << "ExplicitEuler";
            break;
        }
    }

    if (floating_base)
    {
        test_name_ss << "FloatingBase";
    }

    if (forces)
    {
        test_name_ss << "WithForces";
    }

    if (unactuated_joints)
    {
        test_name_ss << "WithUnactuatedJoints";
    }

    std::filesystem::path p(urdf_path);
    test_name_ss << "_" << p.stem().string() << "_";

    test_name_ss << "DynamicsGradTest";

    return std::make_shared<FDGradDynamics>(test_name_ss.str(),
                                            model,
                                            0,
                                            100,
                                            1e-4);
}

std::shared_ptr<Test> generate_innate_constraint_gradient_test(
  const std::string &urdf_path, const bool floating_base,
  const model::SolverMode &sm, const bool forces, const bool unactuated_joints,
  const uint seed)
{
    std::random_device dev;
    std::mt19937       rng(dev());
    rng.seed(seed);

    model::Model::Structure model_structure;
    model_structure.urdf_path   = urdf_path;
    model_structure.free_flyer  = floating_base;
    model_structure.solver_mode = sm;

    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    setup_model(model);

    if (forces)
    {
        add_forces_to_model(model, rng);
    }

    if (unactuated_joints)
    {
        set_model_unactuated_joints(model, rng);
    }

    model->finalize();

    std::stringstream test_name_ss;

    switch (sm)
    {
    default:
    case model::SolverMode::ForwardDynamics:
        test_name_ss << "Fwdyn";
        break;
    case model::SolverMode::InverseDynamics:
        test_name_ss << "Invdyn";
        break;
    case model::SolverMode::ReducedInverseDynamics:
        test_name_ss << "ReducedInvdyn";
        break;
    }

    if (floating_base)
    {
        test_name_ss << "FloatingBase";
    }

    if (forces)
    {
        test_name_ss << "WithForces";
    }

    if (unactuated_joints)
    {
        test_name_ss << "WithUnactuatedJoints";
    }

    std::filesystem::path p(urdf_path);
    test_name_ss << "_" << p.stem().string() << "_";

    test_name_ss << "InnateConstraintGradTest";

    return std::make_shared<FDGradConstraints>(test_name_ss.str(),
                                               model,
                                               0,
                                               100,
                                               1e-4);
}

std::shared_ptr<Test> get_double_pendulum_dynamics_grad_test(const uint seed)
{
    std::random_device dev;
    std::mt19937       rng(dev());
    rng.seed(seed);

    std::uniform_real_distribution<double> dist(0.1, 100.0);

    model::Model::Structure       model_structure;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01);

    std::shared_ptr<model::Model::CustomProperties> custom_properties =
      std::make_shared<model::Model::CustomProperties>();
    custom_properties->nq            = 2;
    custom_properties->nv            = 2;
    custom_properties->nu            = 2;
    custom_properties->njoints       = 2;
    custom_properties->floating_base = false;

    model->setCustomProperties(custom_properties);

    setup_model(model);

    Vec6 theta;

    for (uint i = 0; i < 6; ++i)
    {
        theta[i] = dist(rng);
    }

    std::shared_ptr<model::custom::DoublePendulumDynamics> dynamics =
      std::make_shared<model::custom::DoublePendulumDynamics>(theta);

    model::Model::Config config;
    config.dynamics = { dynamics, dynamics, dynamics };

    model->finalize(config);

    std::shared_ptr<FDGradDynamics> dynamics_test =
      std::make_shared<FDGradDynamics>("DoublePendulumDynamicsGradTest",
                                       model,
                                       0);

    return dynamics_test;
}

std::shared_ptr<Test> get_double_pendulum_dynamics_parameter_grad_test(
  const uint seed)
{
    model::Model::Structure       model_structure;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01);

    std::shared_ptr<model::Model::CustomProperties> custom_properties =
      std::make_shared<model::Model::CustomProperties>();
    custom_properties->nq            = 2;
    custom_properties->nv            = 2;
    custom_properties->nu            = 2;
    custom_properties->njoints       = 2;
    custom_properties->floating_base = false;

    model->setCustomProperties(custom_properties);

    setup_model(model);

    std::shared_ptr<model::custom::DoublePendulumDynamics> dynamics =
      std::make_shared<model::custom::DoublePendulumDynamics>(Vec6::Zero());

    model::Model::Config config;
    config.dynamics = { dynamics, dynamics, dynamics };

    model->finalize(config);

    std::shared_ptr<FDGradDynamicsParameters> dynamics_test =
      std::make_shared<FDGradDynamicsParameters>(
        "DoublePendulumDynamicsParameterGradTest",
        model,
        0,
        100,
        1e-6,
        seed);

    return dynamics_test;
}

std::shared_ptr<Test> get_joint_angle_constraint_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.ignore_innate_constraints = true;
    model_structure.urdf_path = "examples/assets/urdf/panda/panda.urdf";
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);
    setup_model(model);

    model->addConstraints<
      model::stagewise::predefined::constraints::JointAngleConstraint>(
      0,
      *model.get(),
      VecX::Zero(9),
      VecX::Zero(9));

    model->finalize();

    std::shared_ptr<FDGradConstraints> constraints_test =
      std::make_shared<FDGradConstraints>("JointAngleConstraintGradTest",
                                          model,
                                          0);

    return constraints_test;
}

std::shared_ptr<Test> get_joint_velocity_constraint_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.ignore_innate_constraints = true;
    model_structure.urdf_path = "examples/assets/urdf/panda/panda.urdf";
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);
    setup_model(model);

    model->addConstraints<
      model::stagewise::predefined::constraints::JointVelocityConstraint>(
      0,
      *model.get(),
      VecX::Zero(9),
      VecX::Zero(9));

    model->finalize();

    std::shared_ptr<FDGradConstraints> constraints_test =
      std::make_shared<FDGradConstraints>("JointVelocityConstraintGradTest",
                                          model,
                                          0);

    return constraints_test;
}
std::shared_ptr<Test> get_joint_control_constraint_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.ignore_innate_constraints = true;
    model_structure.urdf_path = "examples/assets/urdf/panda/panda.urdf";
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);
    setup_model(model);

    model->addConstraints<
      model::stagewise::predefined::constraints::JointControlConstraint>(
      0,
      *model.get(),
      VecX::Zero(9),
      VecX::Zero(9));

    model->finalize();

    std::shared_ptr<FDGradConstraints> constraints_test =
      std::make_shared<FDGradConstraints>("JointControlConstraintGradTest",
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
      *model.get(),
      "trunk",
      Vec3::Zero(),
      Vec3::Zero());

    model->finalize();

    std::shared_ptr<FDGradConstraints> constraints_test =
      std::make_shared<FDGradConstraints>("FramePositionBoundGradTest",
                                          model,
                                          0);

    return constraints_test;
}

std::shared_ptr<Test> get_frame_rotation_bound_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.ignore_innate_constraints = true;
    model_structure.urdf_path  = "examples/assets/urdf/a1/a1.urdf";
    model_structure.free_flyer = true;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);
    setup_model(model);

    model->addConstraints<
      model::stagewise::predefined::constraints::FrameRotationBound>(
      0,
      *model.get(),
      "trunk",
      MatX::Identity(3, 3),
      Vec3::Zero(),
      Vec3::Zero());

    model->finalize();

    std::shared_ptr<FDGradConstraints> constraints_test =
      std::make_shared<FDGradConstraints>("FrameRotationBoundGradTest",
                                          model,
                                          0,
                                          100,
                                          1e-5);

    return constraints_test;
}

std::shared_ptr<Test> get_frame_linear_velocity_bound_grad_test()
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
      *model.get(),
      "trunk",
      Vec3::Zero(),
      Vec3::Zero());

    model->finalize();

    std::shared_ptr<FDGradConstraints> constraints_test =
      std::make_shared<FDGradConstraints>("FrameLinearVelocityBoundGradTest",
                                          model,
                                          0);

    return constraints_test;
}

std::shared_ptr<Test> get_center_of_mass_bound_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.ignore_innate_constraints = true;
    model_structure.urdf_path  = "examples/assets/urdf/a1/a1.urdf";
    model_structure.free_flyer = true;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);
    setup_model(model);

    model->addConstraints<
      model::stagewise::predefined::constraints::CenterOfMassBound>(
      0,
      *model.get(),
      Vec3::Zero(),
      Vec3::Zero());

    model->finalize();

    std::shared_ptr<FDGradConstraints> constraints_test =
      std::make_shared<FDGradConstraints>("CenterOfMassBoundTest", model, 0);

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
      model::stagewise::predefined::constraints::ContactConstraint>(
      0,
      *model.get(),
      "FL_foot",
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
            *model.get(),
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
            *model.get(),
            0.58);
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
          model::stagewise::predefined::costs::SquaredJointVelocityCost>(
          k,
          *model.get(),
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
          model::stagewise::predefined::costs::SquaredJointVelocityCost>(
          k,
          *model.get(),
          9.43);
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
          *model.get(),
          0.47);
    }

    model
      ->addCost<model::stagewise::predefined::costs::SquaredJointVelocityCost>(
        2,
        *model.get(),
        1.0);

    model->finalize();

    std::shared_ptr<FDGradCost> cost_test =
      std::make_shared<FDGradCost>("SquaredControlCostGradTest", model, 0);

    return cost_test;
}

std::shared_ptr<Test> get_squared_ID_torque_cost_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.urdf_path   = "examples/assets/urdf/panda/panda.urdf";
    model_structure.free_flyer  = false;
    model_structure.solver_mode = model::SolverMode::InverseDynamics;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    for (uint k = 0; k < model->dims().horizon; ++k)
    {
        model
          ->addCost<model::stagewise::predefined::costs::SquaredIDTorqueCost>(
            k,
            *model.get(),
            0.74);
    }

    model
      ->addCost<model::stagewise::predefined::costs::SquaredJointVelocityCost>(
        2,
        *model.get(),
        1.0);

    model->finalize();

    std::shared_ptr<FDGradCost> cost_test =
      std::make_shared<FDGradCost>("SquaredIDTorqueCostGradTest",
                                   model,
                                   0,
                                   100,
                                   1e-5);

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
          *model.get(),
          9.78);
    }

    model
      ->addCost<model::stagewise::predefined::costs::SquaredJointVelocityCost>(
        2,
        *model.get(),
        1.0);

    model->finalize();

    std::shared_ptr<FDGradCost> cost_test =
      std::make_shared<FDGradCost>("SquaredForceCostGradTest", model, 0);

    return cost_test;
}

std::shared_ptr<Test> get_squared_configuration_error_cost_grad_test()
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
          *model.get(),
          VecX::Zero(9),
          9.17);
    }

    model->finalize();

    std::shared_ptr<FDGradCost> cost_test =
      std::make_shared<FDGradCost>("SquaredConfigurationErrorCostGradTest",
                                   model,
                                   0);

    return cost_test;
}

std::shared_ptr<Test>
get_squared_configuration_error_cost_floating_base_grad_test()
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
          *model.get(),
          VecX::Zero(12),
          3.27);
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
                                                           *model.get(),
                                                           "trunk",
                                                           Vec3::Zero(),
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
          *model.get(),
          "trunk",
          Mat3::Identity(),
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
          *model.get(),
          "trunk",
          Vec3::Zero(),
          Mat3::Identity(),
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
          *model.get(),
          "FR_foot",
          vd,
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
          *model.get(),
          "FR_foot",
          2.61);
    }

    model->finalize();

    std::shared_ptr<FDGradCost> cost_test =
      std::make_shared<FDGradCost>("FrameSquaredVelocityCostGradTest",
                                   model,
                                   0);

    return cost_test;
}

std::shared_ptr<Test> get_center_of_mass_tracking_cost_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/assets/urdf/a1/a1.urdf";
    model_structure.free_flyer = true;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    for (uint k = 0; k <= model->dims().horizon; ++k)
    {
        model->addCost<
          model::stagewise::predefined::costs::CenterOfMassTrackingCost>(
          k,
          *model.get(),
          (Eigen::Vector3d() << 2.1, 3.2, -1.3).finished(),
          4.13);
    }

    model->finalize();

    std::shared_ptr<FDGradCost> cost_test =
      std::make_shared<FDGradCost>("CenterOfMassTrackingCostGradTest",
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
          model::stagewise::predefined::costs::SquaredJointVelocityCost>(
          k,
          *model.get(),
          2.98);
        model->addCost<model::stagewise::predefined::costs::SquaredControlCost>(
          k,
          *model.get(),
          7.07);
    }

    model
      ->addCost<model::stagewise::predefined::costs::SquaredJointVelocityCost>(
        horizon,
        *model.get(),
        6.33);

    VecX init_state = VecX::Zero(37);
    init_state[6]   = 1.0;

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
          model::stagewise::predefined::costs::SquaredJointVelocityCost>(
          k,
          *model.get(),
          8.13);
        model->addCost<model::stagewise::predefined::costs::SquaredControlCost>(
          k,
          *model.get(),
          0.66);
    }

    model
      ->addCost<model::stagewise::predefined::costs::SquaredJointVelocityCost>(
        horizon,
        *model.get(),
        8.72);

    VecX init_state = VecX::Zero(37);
    init_state[6]   = 1.0;

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
          model::stagewise::predefined::costs::SquaredJointVelocityCost>(
          k,
          *model.get(),
          8.37);
        model->addCost<model::stagewise::predefined::costs::SquaredControlCost>(
          k,
          *model.get(),
          7.53);

        model->addConstraints<
          model::stagewise::predefined::constraints::FramePositionBound>(
          k,
          *model.get(),
          "FL_foot",
          lb,
          ub);
    }

    model
      ->addCost<model::stagewise::predefined::costs::SquaredJointVelocityCost>(
        horizon,
        *model.get(),
        2.16);

    model->addConstraints<
      model::stagewise::predefined::constraints::FramePositionBound>(
      horizon,
      *model.get(),
      "FL_foot",
      lb,
      ub);

    VecX init_state = VecX::Zero(37);
    init_state[6]   = 1.0;

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