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

using CostsContainer       = Container<const stagewise::StageCost>;
using DynamicsContainer    = Container<const stagewise::StageDynamics>;
using ConstraintsContainer = Container<const stagewise::StageConstraints>;

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

    const Dimensions linear_dims() const;

    const CostsContainer       costs() const;
    const DynamicsContainer    dynamics() const;
    const ConstraintsContainer constraints() const;

  protected:
    Dimensions           _dims, _lin_dims;
    CostsContainer       _costs;
    DynamicsContainer    _dynamics;
    ConstraintsContainer _constraints;
};
}
}

#endif