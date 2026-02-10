#ifndef SSQP_TESTS_GENERATORS_GRADIENTS_HPP
#define SSQP_TESTS_GENERATORS_GRADIENTS_HPP

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
void setup_model(std::shared_ptr<model::Model> &model);

// Dynamics gradients
std::shared_ptr<Test> get_fwdyn_explicit_euler_without_forces_grad_test();
std::shared_ptr<Test> get_fwdyn_explicit_euler_with_forces_grad_test();
std::shared_ptr<Test> get_fwdyn_semi_implicit_euler_without_forces_grad_test();
std::shared_ptr<Test> get_fwdyn_semi_implicit_euler_with_forces_grad_test();
std::shared_ptr<Test> get_invdyn_explicit_euler_without_forces_grad_test();
std::shared_ptr<Test> get_invdyn_explicit_euler_with_forces_grad_test();
std::shared_ptr<Test> get_invdyn_semi_implicit_euler_without_forces_grad_test();
std::shared_ptr<Test> get_invdyn_semi_implicit_euler_with_forces_grad_test();

// Constraints gradients
std::shared_ptr<Test> get_fwdyn_innate_constraint_grad_test();
std::shared_ptr<Test> get_invdyn_innate_constraint_without_forces_grad_test();
std::shared_ptr<Test> get_invdyn_innate_constraint_with_forces_grad_test();
std::shared_ptr<Test> get_frame_position_bound_grad_test();
std::shared_ptr<Test> get_frame_velocity_bound_grad_test();
std::shared_ptr<Test> get_contact_constraint_grad_test();

// Cost gradients
std::shared_ptr<Test> get_squared_joint_angle_cost_grad_test();
std::shared_ptr<Test> get_squared_joint_angle_cost_floating_base_grad_test();
std::shared_ptr<Test> get_squared_joint_velocity_cost_grad_test();
std::shared_ptr<Test> get_squared_joint_velocity_cost_floating_base_grad_test();
std::shared_ptr<Test> get_squared_control_cost_grad_test();
std::shared_ptr<Test> get_squared_force_cost_grad_test();
std::shared_ptr<Test> get_squared_configuration_difference_cost_grad_test();
std::shared_ptr<Test>
get_squared_configuration_difference_cost_floating_base_grad_test();
std::shared_ptr<Test> get_frame_squared_position_error_cost_grad_test();
std::shared_ptr<Test> get_frame_squared_rotation_error_cost_grad_test();
std::shared_ptr<Test> get_frame_squared_pose_error_cost_grad_test();
std::shared_ptr<Test> get_frame_squared_velocity_error_cost_grad_test();
std::shared_ptr<Test> get_frame_squared_velocity_cost_grad_test();

// Full trajectory cost gradients
std::shared_ptr<Test> get_lf_grad_test();
std::shared_ptr<Test> get_gammaf_grad_test();
std::shared_ptr<Test> get_cf_grad_test();
}
}
}

#endif