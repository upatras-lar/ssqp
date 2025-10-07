#ifndef SSQP_MODEL_HPP
#define SSQP_MODEL_HPP

#include "types.hpp"
#include "model/Container.hpp"
#include "model/stagewise/StageCost.hpp"
#include "model/stagewise/StageDynamics.hpp"
#include "model/stagewise/StageConstraints.hpp"

namespace ssqp
{
namespace model
{
class Model
{
  public:
    Model(const std::vector<std::shared_ptr<const stagewise::StageCost>> &costs,
          const std::vector<std::shared_ptr<const stagewise::StageDynamics>>
            &dynamics,
          const std::vector<std::shared_ptr<const stagewise::StageConstraints>>
            &constraints = { nullptr });

    const uint              horizon() const;
    const uint              total_state_size() const;
    const uint              total_control_size() const;
    const uint              total_constraint_size() const;
    const std::vector<uint> state_dims() const;
    const std::vector<uint> control_dims() const;
    const std::vector<uint> constraint_dims() const;
    const Dimensions        dims() const;

    const Container<const stagewise::StageCost>        costs() const;
    const Container<const stagewise::StageDynamics>    dynamics() const;
    const Container<const stagewise::StageConstraints> constraints() const;

  protected:
    Dimensions                                   _dims;
    Container<const stagewise::StageCost>        _costs;
    Container<const stagewise::StageDynamics>    _dynamics;
    Container<const stagewise::StageConstraints> _constraints;
};
}
}

#endif