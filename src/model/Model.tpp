#include "model/Model.hpp"

namespace ssqp
{
namespace model
{
template<typename L, typename... Args>
void Model::addCost(const uint k, Args... args)
{
    static_assert(std::is_base_of_v<stagewise::StageCost, L>, "Invalid cost");

    std::shared_ptr<stagewise::StageCost> cost = std::make_shared<L>(args...);

    addCost(k, cost);
}

template<typename C, typename... Args>
void Model::addConstraints(const uint k, Args... args)
{
    static_assert(std::is_base_of_v<stagewise::StageConstraints, C>,
                  "Invalid constraint");

    std::shared_ptr<stagewise::StageConstraints> constraints =
      std::make_shared<C>(args...);

    std::vector<Force> forces(constraints->required_forces());
    for (uint i = 0; i < constraints->required_forces(); ++i)
    {
        forces[i] = addForce(k, constraints->force_frame(i));
    }

    constraints->set_forces(forces);

    addConstraints(k, constraints);
}
}
}