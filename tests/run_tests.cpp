#include "Test.hpp"
#include "generators/gradients.hpp"

int main()
{
    ssqp::tests::TestContainer tests;

    tests.addTest(
      ssqp::tests::generators::get_explicit_euler_without_forces_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_explicit_euler_with_forces_grad_test());

    tests.addTest(ssqp::tests::generators::
                    get_semi_implicit_euler_without_forces_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_semi_implicit_euler_with_forces_grad_test());

    tests.addTest(ssqp::tests::generators::get_innate_constraint_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_frame_position_bound_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_frame_velocity_bound_grad_test());

    tests.addTest(ssqp::tests::generators::get_contact_constraint_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_squared_joint_velocity_cost_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_squared_control_cost_grad_test());

    tests.addTest(ssqp::tests::generators::get_squared_force_cost_grad_test());

    tests.addTest(ssqp::tests::generators::
                    get_frame_squared_position_error_cost_grad_test());

    tests.addTest(ssqp::tests::generators::
                    get_frame_squared_rotation_error_cost_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_frame_squared_pose_error_cost_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_frame_squared_velocity_cost_grad_test());

    tests.runAll();
}