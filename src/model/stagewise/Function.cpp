#include "model/stagewise/Function.hpp"

namespace ssqp
{
namespace model
{
namespace stagewise
{
Function::Function(const std::string &name) : _name(name) {}

const std::string Function::name() const
{
    return _name;
}
}
}
}