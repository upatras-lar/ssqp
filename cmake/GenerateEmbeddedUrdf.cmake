# Generates a C++ header embedding a URDF file's content as a string literal,
# so it can be baked into the ssqp binary and built without a filesystem path
# (see Model::Structure::from_preset / model/presets/Presets.hpp)
#
# Invoked in script mode:
#   cmake -DIN_URDF=<path> -DOUT_HEADER=<path> -DSYMBOL=<name> -P GenerateEmbeddedUrdf.cmake

file(READ "${IN_URDF}" _content)

if(_content MATCHES "\\)SSQP_URDF\"")
    message(FATAL_ERROR "${IN_URDF} contains the raw string delimiter )SSQP_URDF\", pick a different delimiter in GenerateEmbeddedUrdf.cmake")
endif()

string(TOUPPER "${SYMBOL}" _guard)
set(_guard "SSQP_MODEL_PRESETS_${_guard}_HPP")

file(WRITE "${OUT_HEADER}"
"// Generated from ${IN_URDF} at configure time, do not edit
#ifndef ${_guard}
#define ${_guard}

#include <string_view>

namespace ssqp
{
namespace model
{
namespace presets
{
inline constexpr std::string_view ${SYMBOL} = R\"SSQP_URDF(${_content})SSQP_URDF\";
}
}
}

#endif
")
