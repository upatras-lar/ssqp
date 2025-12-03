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
        model
          ->addCost<model::stagewise::predefined::costs::SquaredVelocityCost>(
            k,
            1,
            1);
    }
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
      std::make_shared<FDGradDynamics>("ExplicitEulerWithForcesTest", model, 0);

    return dyn_test;
}

std::shared_ptr<Test> get_semi_implicit_euler_with_forces_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/urdf/a1/a1.urdf";
    model_structure.free_flyer = true;
    std::shared_ptr<model::Model> model =
      std::make_shared<model::Model>(2, 0.01, model_structure);

    setup_model(model);

    model->addForce(0, "FL_foot");
    model->addForce(0, "FR_foot");
    model->addForce(0, "RL_foot");
    model->addForce(0, "RR_foot");
    model->finalize();

    std::shared_ptr<FDGradDynamics> dyn_test =
      std::make_shared<FDGradDynamics>("SemiImplicitEulerWithForcesTest",
                                       model,
                                       0);

    return dyn_test;
}

std::shared_ptr<Test> get_frame_position_bound_grad_test()
{
    model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/urdf/a1/a1.urdf";
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
      std::make_shared<FDGradConstraints>("FramePositionBoundTest", model, 0);

    return constraints_test;
}

}
}
}