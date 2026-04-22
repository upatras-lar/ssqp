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
  const model::DynamicsMode &dm, const model::IntegrationScheme &is,
  const bool forces, const bool unactuated_joints, const uint seed = 42);

// Innate Constraint gradients
std::shared_ptr<Test> generate_innate_constraint_gradient_test(
  const std::string &urdf_path, const bool floating_base,
  const model::DynamicsMode &dm, const bool forces,
  const bool unactuated_joints, const uint seed = 42);

// Predefined Constraint gradients
std::shared_ptr<Test> get_joint_angle_constraint_grad_test();
std::shared_ptr<Test> get_joint_velocity_constraint_grad_test();
std::shared_ptr<Test> get_joint_control_constraint_grad_test();
std::shared_ptr<Test> get_frame_position_bound_grad_test();
std::shared_ptr<Test> get_frame_rotation_bound_grad_test();
std::shared_ptr<Test> get_frame_linear_velocity_bound_grad_test();
std::shared_ptr<Test> get_center_of_mass_bound_grad_test();
std::shared_ptr<Test> get_contact_constraint_grad_test();

// Cost gradients
std::shared_ptr<Test> get_squared_joint_angle_cost_grad_test();
std::shared_ptr<Test> get_squared_joint_angle_cost_floating_base_grad_test();
std::shared_ptr<Test> get_squared_joint_velocity_cost_grad_test();
std::shared_ptr<Test> get_squared_joint_velocity_cost_floating_base_grad_test();
std::shared_ptr<Test> get_squared_control_cost_grad_test();
std::shared_ptr<Test> get_squared_ID_torque_cost_grad_test();
std::shared_ptr<Test> get_squared_force_cost_grad_test();
std::shared_ptr<Test> get_squared_configuration_error_cost_grad_test();
std::shared_ptr<Test>
get_squared_configuration_error_cost_floating_base_grad_test();
std::shared_ptr<Test> get_frame_squared_position_error_cost_grad_test();
std::shared_ptr<Test> get_frame_squared_rotation_error_cost_grad_test();
std::shared_ptr<Test> get_frame_squared_pose_error_cost_grad_test();
std::shared_ptr<Test> get_frame_squared_velocity_error_cost_grad_test();
std::shared_ptr<Test> get_frame_squared_velocity_cost_grad_test();
std::shared_ptr<Test> get_center_of_mass_tracking_cost_grad_test();

// Full trajectory cost gradients
std::shared_ptr<Test> get_lf_grad_test();
std::shared_ptr<Test> get_gammaf_grad_test();
std::shared_ptr<Test> get_cf_grad_test();
}
}
}

#endif