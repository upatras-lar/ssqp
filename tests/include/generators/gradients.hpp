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

std::shared_ptr<Test> get_explicit_euler_with_forces_grad_test();
std::shared_ptr<Test> get_semi_implicit_euler_with_forces_grad_test();
std::shared_ptr<Test> get_frame_position_bound_grad_test();
}
}
}