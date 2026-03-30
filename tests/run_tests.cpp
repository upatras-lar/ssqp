#include "Test.hpp"
#include "generators/gradients.hpp"
#include "generators/qp_solvers.hpp"

int main()
{
    ssqp::tests::TestContainer tests;

    const std::vector<std::pair<std::string, bool>> urdfs = {
        { "examples/assets/urdf/double_pendulum/double_pendulum.urdf", false },
        {        "examples/assets/urdf/double_pendulum/pendubot.urdf", false },
        {   "examples/assets/urdf/double_pendulum/pendubot_cont.urdf", false },
        {                     "examples/assets/urdf/panda/panda.urdf", false },
        {                         "examples/assets/urdf/go2/go2.urdf",  true },
        {       "examples/assets/urdf/cont_floater/cont_floater.urdf",  true }
    };

    const std::vector<ssqp::model::DynamicsMode> dms = {
        ssqp::model::DynamicsMode::Forward,
        ssqp::model::DynamicsMode::Inverse,
        ssqp::model::DynamicsMode::ReducedInverse
    };

    const std::vector<ssqp::model::IntegrationScheme> iss = {
        ssqp::model::IntegrationScheme::ExplicitEuler,
        ssqp::model::IntegrationScheme::SemiImplicitEuler
    };

    const std::vector<bool> opts = { false, true };

    for (const auto &urdf_pair : urdfs)
    {
        for (const auto &dm : dms)
        {
            for (const auto &is : iss)
            {
                for (const auto &forces : opts)
                {
                    for (const auto &unactuated_joints : opts)
                    {
                        tests.addTest(
                          ssqp::tests::generators::
                            generate_dynamics_gradient_test(urdf_pair.first,
                                                            urdf_pair.second,
                                                            dm,
                                                            is,
                                                            forces,
                                                            unactuated_joints));
                    }
                }
            }
        }
    }

    for (const auto &urdf_pair : urdfs)
    {
        for (const auto &dm : dms)
        {
            for (const auto &forces : opts)
            {
                for (const auto &unactuated_joints : opts)
                {
                    tests.addTest(ssqp::tests::generators::
                                    generate_innate_constraint_gradient_test(
                                      urdf_pair.first,
                                      urdf_pair.second,
                                      dm,
                                      forces,
                                      unactuated_joints));
                }
            }
        }
    }

    tests.addTest(
      ssqp::tests::generators::get_joint_angle_constraint_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_joint_velocity_constraint_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_joint_control_constraint_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_frame_position_bound_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_frame_rotation_bound_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_frame_linear_velocity_bound_grad_test());

    tests.addTest(ssqp::tests::generators::get_contact_constraint_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_squared_joint_angle_cost_grad_test());

    tests.addTest(ssqp::tests::generators::
                    get_squared_joint_angle_cost_floating_base_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_squared_joint_velocity_cost_grad_test());

    tests.addTest(ssqp::tests::generators::
                    get_squared_joint_velocity_cost_floating_base_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_squared_ID_torque_cost_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_squared_control_cost_grad_test());

    tests.addTest(ssqp::tests::generators::get_squared_force_cost_grad_test());

    tests.addTest(ssqp::tests::generators::
                    get_squared_configuration_error_cost_grad_test());

    tests.addTest(
      ssqp::tests::generators::
        get_squared_configuration_error_cost_floating_base_grad_test());

    tests.addTest(ssqp::tests::generators::
                    get_frame_squared_position_error_cost_grad_test());

    tests.addTest(ssqp::tests::generators::
                    get_frame_squared_rotation_error_cost_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_frame_squared_pose_error_cost_grad_test());

    tests.addTest(ssqp::tests::generators::
                    get_frame_squared_velocity_error_cost_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_frame_squared_velocity_cost_grad_test());

    tests.addTest(ssqp::tests::generators::get_lf_grad_test());

    tests.addTest(ssqp::tests::generators::get_gammaf_grad_test());

    tests.addTest(ssqp::tests::generators::get_cf_grad_test());

    // for (std::shared_ptr<ssqp::tests::Test> test :
    //      ssqp::tests::generators::get_hpipm_problems())
    // {
    //     tests.addTest(test);
    // }

    // for (std::shared_ptr<ssqp::tests::Test> test :
    //      ssqp::tests::generators::get_admm_problems())
    // {
    //     tests.addTest(test);
    // }

    // tests.addTest(ssqp::tests::generators::get_admm_infeasibility());

    tests.runAll();
}