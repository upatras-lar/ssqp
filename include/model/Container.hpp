#ifndef SSQP_MODEL_CONTAINER_HPP
#define SSQP_MODEL_CONTAINER_HPP

#include <vector>
#include <memory>

#include "types.hpp"
#include "model/stagewise/Function.hpp"

namespace ssqp
{
namespace model
{
/** Defines a container class for stagewise functions of any type
 * Equivalent to a vector for stage stagewise::Function objects
 */
template<typename Derived>
class Container
{
    static_assert(std::is_base_of<stagewise::Function, Derived>::value,
                  "Container can only contain Function derived objects");

  public:
    // Initialize empty function container, considering all stage functions to
    // be 0
    Container(const uint horizon);
    // Initialize function container with given functions
    // Horizon is taken as functions.size()
    Container(const std::vector<std::shared_ptr<Derived>> &functions);

    Container &operator=(
      const std::vector<std::shared_ptr<Derived>> &functions);

    // Get reference to k-th function for setting
    std::shared_ptr<Derived> &operator[](const uint k);

    // Get const reference to k-th function for reading
    const std::shared_ptr<Derived> &operator[](const uint k) const;

    const uint size() const;

  protected:
    uint                                  _horizon;
    std::vector<std::shared_ptr<Derived>> _functions;
};
}
}

#include "src/model/Container.tpp"

#endif