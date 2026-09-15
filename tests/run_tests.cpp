#include "Test.hpp"
#include "MPC.hpp"
#include "Regression.hpp"
#include "generators/gradients.hpp"
#include "generators/qp_solvers.hpp"

int main(int argc, char **argv)
{
    std::string filter;
    uint32_t    seed = 42;
    bool        list = false;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--list") list = true;
        else if (arg == "--filter" && i + 1 < argc) filter = argv[++i];
        else if (arg == "--seed" && i + 1 < argc)
        {
            try
            {
                seed = std::stoul(argv[++i]);
            }
            catch (...)
            {
                std::cerr << "Invalid seed\n";
                return 2;
            }
        }
        else
        {
            std::cerr << "Usage: run_tests [--list] [--filter substring] "
                         "[--seed integer]\n";
            return 2;
        }
    }
    ssqp::set_silent_mode();
    ssqp::tests::TestContainer tests;

    const std::vector<std::pair<std::string, bool>> urdfs = {
        { "examples/assets/urdf/double_pendulum/double_pendulum.urdf", false },
        {        "examples/assets/urdf/double_pendulum/pendubot.urdf", false },
        {   "examples/assets/urdf/double_pendulum/pendubot_cont.urdf", false },
        {                     "examples/assets/urdf/panda/panda.urdf", false },
        {                         "examples/assets/urdf/go2/go2.urdf",  true },
        {       "examples/assets/urdf/cont_floater/cont_floater.urdf",  true }
    };

    const std::vector<ssqp::model::SolverMode> sms = {
        ssqp::model::SolverMode::Kinematics,
        ssqp::model::SolverMode::ForwardDynamics,
        ssqp::model::SolverMode::InverseDynamics,
        ssqp::model::SolverMode::ReducedInverseDynamics
    };

    const std::vector<ssqp::model::IntegrationScheme> iss = {
        ssqp::model::IntegrationScheme::ExplicitEuler,
        ssqp::model::IntegrationScheme::SemiImplicitEuler
    };

    const std::vector<bool> opts = { false, true };

    for (const auto &urdf_pair : urdfs)
    {
        for (const auto &sm : sms)
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
                                                            sm,
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
        for (const auto &sm : sms)
        {
            for (const auto &forces : opts)
            {
                for (const auto &unactuated_joints : opts)
                {
                    tests.addTest(ssqp::tests::generators::
                                    generate_innate_constraint_gradient_test(
                                      urdf_pair.first,
                                      urdf_pair.second,
                                      sm,
                                      forces,
                                      unactuated_joints));
                }
            }
        }
    }

    for (const auto &urdf_pair : urdfs)
    {
        tests.addTest(
          ssqp::tests::generators::
            generate_reconfigured_actuated_joints_dynamics_grad_test(
              urdf_pair.first,
              urdf_pair.second));
    }

    tests.addTest(ssqp::tests::generators::
                    get_double_pendulum_ambient_dynamics_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_pendubot_ambient_dynamics_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_acrobot_ambient_dynamics_grad_test());

    // tests.addTest(ssqp::tests::generators::
    //                 get_double_pendulum_ambient_dynamics_parameter_grad_test());

    tests.addTest(ssqp::tests::generators::
                    get_double_pendulum_ambient_tracking_cost_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_joint_angle_constraint_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_joint_velocity_constraint_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_joint_control_constraint_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_floating_base_position_bound_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_floating_base_rotation_bound_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_floating_base_pose_bound_grad_test());

    tests.addTest(ssqp::tests::generators::
                    get_floating_base_linear_velocity_bound_grad_test());

    tests.addTest(ssqp::tests::generators::
                    get_floating_base_angular_velocity_bound_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_frame_translation_bound_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_frame_rotation_bound_grad_test());

    tests.addTest(ssqp::tests::generators::get_frame_pose_bound_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_frame_linear_velocity_bound_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_frame_angular_velocity_bound_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_center_of_mass_bound_grad_test());

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

    tests.addTest(ssqp::tests::generators::
                    get_squared_ID_torque_cost_reduced_grad_test());

    tests.addTest(ssqp::tests::generators::
                    get_squared_ID_torque_cost_floating_base_grad_test());

    tests.addTest(
      ssqp::tests::generators::
        get_squared_ID_torque_cost_reduced_floating_base_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_squared_control_cost_grad_test());

    tests.addTest(ssqp::tests::generators::get_squared_force_cost_grad_test());

    tests.addTest(ssqp::tests::generators::
                    get_squared_configuration_error_cost_grad_test());

    tests.addTest(
      ssqp::tests::generators::
        get_squared_configuration_error_cost_floating_base_grad_test());

    tests.addTest(
      ssqp::tests::generators::
        get_floating_base_squared_translation_error_cost_grad_test());

    tests.addTest(ssqp::tests::generators::
                    get_floating_base_squared_rotation_error_cost_grad_test());

    tests.addTest(ssqp::tests::generators::
                    get_floating_base_squared_pose_error_cost_grad_test());

    tests.addTest(ssqp::tests::generators::
                    get_floating_base_squared_velocity_error_cost_grad_test());

    tests.addTest(ssqp::tests::generators::
                    get_frame_squared_translation_error_cost_grad_test());

    tests.addTest(ssqp::tests::generators::
                    get_frame_squared_rotation_error_cost_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_frame_squared_pose_error_cost_grad_test());

    tests.addTest(ssqp::tests::generators::
                    get_frame_squared_velocity_error_cost_grad_test());

    tests.addTest(
      ssqp::tests::generators::get_center_of_mass_tracking_cost_grad_test());

    tests.addTest(ssqp::tests::generators::
                    get_frames_inverse_squared_distance_cost_grad_test());

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

    tests.addTest(std::make_shared<ssqp::tests::MPCStepTest>(
      "MPCStepADMMTest",
      ssqp::solvers::ssqp::QPSolverType::ADMM));

#ifdef WITH_HPIPM
    tests.addTest(std::make_shared<ssqp::tests::MPCStepTest>(
      "MPCStepHPIPMTest",
      ssqp::solvers::ssqp::QPSolverType::HPIPM));
#endif

    ssqp::tests::addRegressionTests(tests);
    if (list)
    {
        tests.list();
        return 0;
    }
    return tests.runAll(filter, seed) ? 0 : 1;
}
