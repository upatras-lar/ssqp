#ifndef SSQP_TESTS_GENERATORS_GRADIENTS_HPP
#define SSQP_TESTS_GENERATORS_GRADIENTS_HPP

#include <random>
#include <filesystem>

#include "Test.hpp"
#include "FDGrad.hpp"
#include "types.hpp"
#include "model/Model.hpp"
#include "model/stagewise/predefined/Costs.hpp"
#include "model/stagewise/predefined/Constraints.hpp"

#include "models/double_pendulum/DoublePendulumAmbient.hpp"

namespace ssqp
{
namespace tests
{
namespace generators
{
// Helper functions
void setup_model(std::shared_ptr<model::Model> &model);
void add_forces_to_model(std::shared_ptr<model::Model> &model,
                         std::mt19937                  &rng);
void set_model_unactuated_joints(std::shared_ptr<model::Model> &model,
                                 std::mt19937                  &rng);

// Dynamics gradients
std::shared_ptr<Test> generate_dynamics_gradient_test(
  const std::string &urdf_path, const bool floating_base,
  const model::SolverMode &sm, const model::IntegrationScheme &is,
  const bool forces, const bool unactuated_joints, const uint seed = 42);

// Innate Constraint gradients
std::shared_ptr<Test> generate_innate_constraint_gradient_test(
  const std::string &urdf_path, const bool floating_base,
  const model::SolverMode &sm, const bool forces, const bool unactuated_joints,
  const uint seed = 42);

// Dynamics gradient after reconfiguring the actuated joints more than once
std::shared_ptr<Test> generate_reconfigured_actuated_joints_dynamics_grad_test(
  const std::string &urdf_path, const bool floating_base, const uint seed = 42);

// Predefined dynamics model gradients
std::shared_ptr<Test> get_double_pendulum_ambient_dynamics_grad_test(
  const uint seed = 42);
std::shared_ptr<Test> get_pendubot_ambient_dynamics_grad_test(
  const uint seed = 42);
std::shared_ptr<Test> get_acrobot_ambient_dynamics_grad_test(
  const uint seed = 42);

// Predefined cost gradients for predefined dynamics models
std::shared_ptr<Test> get_double_pendulum_ambient_tracking_cost_grad_test(
  const uint seed = 42);

// Predefined Dynamics model parameter gradients
std::shared_ptr<Test> get_double_pendulum_ambient_dynamics_parameter_grad_test(
  const uint seed = 42);

// Predefined Constraint gradients
std::shared_ptr<Test> get_joint_angle_constraint_grad_test();
std::shared_ptr<Test> get_joint_velocity_constraint_grad_test();
std::shared_ptr<Test> get_joint_control_constraint_grad_test();
std::shared_ptr<Test> get_floating_base_position_bound_grad_test();
std::shared_ptr<Test> get_floating_base_rotation_bound_grad_test();
std::shared_ptr<Test> get_floating_base_pose_bound_grad_test();
std::shared_ptr<Test> get_floating_base_linear_velocity_bound_grad_test();
std::shared_ptr<Test> get_floating_base_angular_velocity_bound_grad_test();
std::shared_ptr<Test> get_frame_translation_bound_grad_test();
std::shared_ptr<Test> get_frame_rotation_bound_grad_test();
std::shared_ptr<Test> get_frame_pose_bound_grad_test();
std::shared_ptr<Test> get_frame_linear_velocity_bound_grad_test();
std::shared_ptr<Test> get_frame_angular_velocity_bound_grad_test();
std::shared_ptr<Test> get_center_of_mass_bound_grad_test();
std::shared_ptr<Test> get_contact_constraint_grad_test();

// Cost gradients
std::shared_ptr<Test> get_squared_joint_angle_cost_grad_test();
std::shared_ptr<Test> get_squared_joint_angle_cost_floating_base_grad_test();
std::shared_ptr<Test> get_squared_joint_velocity_cost_grad_test();
std::shared_ptr<Test> get_squared_joint_velocity_cost_floating_base_grad_test();
std::shared_ptr<Test> get_squared_control_cost_grad_test();
std::shared_ptr<Test> get_squared_ID_torque_cost_grad_test();
std::shared_ptr<Test> get_squared_ID_torque_cost_reduced_grad_test(
  const uint seed = 42);
std::shared_ptr<Test> get_squared_ID_torque_cost_floating_base_grad_test(
  const uint seed = 42);
std::shared_ptr<Test>
get_squared_ID_torque_cost_reduced_floating_base_grad_test(
  const uint seed = 42);
std::shared_ptr<Test> get_squared_force_cost_grad_test();
std::shared_ptr<Test> get_squared_wrench_cost_grad_test();
std::shared_ptr<Test> get_wrench_dynamics_grad_test();
std::shared_ptr<Test> get_squared_configuration_error_cost_grad_test();
std::shared_ptr<Test>
get_squared_configuration_error_cost_floating_base_grad_test();
std::shared_ptr<Test>
get_floating_base_squared_translation_error_cost_grad_test();
std::shared_ptr<Test> get_floating_base_squared_rotation_error_cost_grad_test();
std::shared_ptr<Test> get_floating_base_squared_pose_error_cost_grad_test();
std::shared_ptr<Test> get_floating_base_squared_velocity_error_cost_grad_test();
std::shared_ptr<Test> get_frame_squared_translation_error_cost_grad_test();
std::shared_ptr<Test> get_frame_squared_rotation_error_cost_grad_test();
std::shared_ptr<Test> get_frame_squared_pose_error_cost_grad_test();
std::shared_ptr<Test> get_frame_squared_velocity_error_cost_grad_test();
std::shared_ptr<Test> get_center_of_mass_tracking_cost_grad_test();
std::shared_ptr<Test> get_frames_inverse_squared_distance_cost_grad_test();

// Full trajectory cost gradients
std::shared_ptr<Test> get_lf_grad_test();
std::shared_ptr<Test> get_gammaf_grad_test();
std::shared_ptr<Test> get_cf_grad_test();
}
}
}

#endif