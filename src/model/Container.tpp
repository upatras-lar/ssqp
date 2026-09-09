#include "model/Container.hpp"

namespace ssqp
{
namespace model
{
template<typename Derived>
Container<Derived>::Container(const uint horizon) : _functions(horizon, nullptr)
{
}

template<typename Derived>
Container<Derived>::Container(
  const std::vector<std::shared_ptr<Derived>> &functions)
  : _functions(functions)
{
}

template<typename Derived>
Container<Derived> &Container<Derived>::operator=(
  const std::vector<std::shared_ptr<Derived>> &functions)
{
    _functions = functions;

    return *this;
}

template<typename Derived>
std::shared_ptr<Derived> &Container<Derived>::operator[](const uint k) noexcept
{
    return _functions[k];
}

template<typename Derived>
const std::shared_ptr<Derived> &Container<Derived>::operator[](
  const uint k) const noexcept
{
    return _functions[k];
}

template<typename Derived>
uint Container<Derived>::size() const noexcept
{
    return static_cast<uint>(_functions.size());
}

}
}