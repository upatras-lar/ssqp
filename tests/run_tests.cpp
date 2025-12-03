#include "Test.hpp"
#include "generators/gradients.hpp"

int main()
{
    ssqp::tests::TestContainer tests;
    tests.addTest(
      ssqp::tests::generators::get_semi_implicit_euler_with_forces_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_explicit_euler_with_forces_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_frame_position_bound_grad_test());

    tests.runAll();
}