#include "model/Container.hpp"

namespace ssqp
{
namespace model
{
template<typename Derived>
Container<Derived>::Container(const uint horizon)
  : _horizon(horizon)
  , _functions(horizon)
{
}

template<typename Derived>
Container<Derived>::Container(
  const std::vector<std::unique_ptr<const Derived>> &functions)
  : _horizon(functions.size())
  , _functions(std::move(functions))
{
}

template<typename Derived>
std::unique_ptr<const Derived> &Container<Derived>::operator[](const uint k)
{
    return _functions[k];
}

template<typename Derived>
const std::unique_ptr<const Derived> &Container<Derived>::operator[](
  const uint k) const
{
    return _functions[k];
}

template class Container<stagewise::StageCost>;
template class Container<stagewise::StageDynamics>;
template class Container<stagewise::StageConstraints>;
}
}