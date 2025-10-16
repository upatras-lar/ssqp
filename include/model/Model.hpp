#ifndef SSQP_MODEL_HPP
#define SSQP_MODEL_HPP

#include "types.hpp"
#include "model/Container.hpp"
#include "model/stagewise/StageCost.hpp"
#include "model/stagewise/StageDynamics.hpp"
#include "model/stagewise/StageConstraints.hpp"
#include "model/stagewise/Factories.hpp"

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

    Model(const std::string &urdf_path, const uint horizon, const double dt,
          const std::shared_ptr<const stagewise::StageCost> &cost,
          const std::shared_ptr<const stagewise::StageCost> &terminal_cost,
          VecX init_state = VecX::Zero(0));

    const uint       horizon() const;
    const Dimensions dims() const;
    const Dimensions linear_dims() const;

    const CostsContainer       costs() const;
    const DynamicsContainer    dynamics() const;
    const ConstraintsContainer constraints() const;

  protected:
    Dimensions           _dims, _lin_dims;
    CostsContainer       _costs;
    DynamicsContainer    _dynamics;
    ConstraintsContainer _constraints;

    void _setup();
};
}
}

#endif