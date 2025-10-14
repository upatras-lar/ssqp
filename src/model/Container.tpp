#include "model/Container.hpp"

namespace ssqp
{
namespace model
{
template<typename Derived>
Container<Derived>::Container(const uint horizon)
  : _horizon(horizon)
  , _functions(horizon, nullptr)
{
}

template<typename Derived>
Container<Derived>::Container(
  const std::vector<std::shared_ptr<const Derived>> &functions)
  : _horizon(functions.size())
  , _functions(functions)
{
}

template<typename Derived>
std::shared_ptr<const Derived> &Container<Derived>::operator[](const uint k)
{
    return _functions[k];
}

template<typename Derived>
const std::shared_ptr<const Derived> &Container<Derived>::operator[](
  const uint k) const
{
    return _functions[k];
}

template<typename Derived>
const uint Container<Derived>::size() const
{
    return _functions.size();
}

}
}