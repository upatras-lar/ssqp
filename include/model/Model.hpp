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

using CostsContainer       = Container<stagewise::StageCost>;
using DynamicsContainer    = Container<stagewise::StageDynamics>;
using ConstraintsContainer = Container<stagewise::StageConstraints>;

class Model
{
  public:
    Model(
      const std::vector<std::shared_ptr<stagewise::StageCost>>     &costs,
      const std::vector<std::shared_ptr<stagewise::StageDynamics>> &dynamics,
      const std::vector<std::shared_ptr<stagewise::StageConstraints>>
        &constraints = {});

    Model(const std::string &urdf_path, const uint horizon, const double dt,
          const std::shared_ptr<stagewise::StageCost> &cost,
          const std::shared_ptr<stagewise::StageCost> &terminal_cost,
          const std::vector<std::shared_ptr<stagewise::StageConstraints>>
            &constraints                       = {},
          const std::vector<VecX> &init_states = {});

    Model(const std::string &urdf_path, const uint horizon, const double dt,
          const std::vector<std::shared_ptr<stagewise::StageCost>> &costs,
          const std::shared_ptr<stagewise::StageCost> &terminal_cost,
          const std::vector<std::shared_ptr<stagewise::StageConstraints>>
            &constraints                       = {},
          const std::vector<VecX> &init_states = {});

    // Using this constructor requires that Model::finalize is manually called
    Model(const std::string &urdf_path, const uint horizon, const double dt,
          const std::vector<VecX> &init_states = {});

    void finalize(
      const std::shared_ptr<stagewise::StageCost> &cost,
      const std::shared_ptr<stagewise::StageCost> &terminal_cost,
      const std::vector<std::shared_ptr<stagewise::StageConstraints>>
        &constraints                       = {},
      const std::vector<VecX> &init_states = {});

    void finalize(
      const std::vector<std::shared_ptr<stagewise::StageCost>> &costs,
      const std::shared_ptr<stagewise::StageCost>              &terminal_cost,
      const std::vector<std::shared_ptr<stagewise::StageConstraints>>
        &constraints                       = {},
      const std::vector<VecX> &init_states = {});

    const uint       horizon() const;
    const Dimensions dims() const;
    const Dimensions linear_dims() const;

    const CostsContainer       costs() const;
    const DynamicsContainer    dynamics() const;
    const ConstraintsContainer constraints() const;

    const std::pair<VecX, MatX> forwardKinematics(
      const VecX &x, const std::string &frame) const;

    const VecX inverseKinematics(const VecX        &desired_translation,
                                 const MatX        &desired_rotation,
                                 const std::string &joint) const;

    const MatX frameJacobian(const VecX &x, const std::string &frame) const;

  protected:
    class ModelImpl;
    std::shared_ptr<ModelImpl> _impl;

    Dimensions           _dims, _lin_dims;
    CostsContainer       _costs;
    DynamicsContainer    _dynamics;
    ConstraintsContainer _constraints;

    void _setup();
};
}
}

#endif