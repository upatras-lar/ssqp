#include "model/stagewise/Function.hpp"

namespace ssqp
{
namespace model
{
namespace stagewise
{
Function::Function(const std::string &name) : _names({ name }) {}

const std::vector<std::string> Function::names() const
{
    return _names;
}
}
}
}