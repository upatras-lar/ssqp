# Initial cache for the PyPI wheels (see [tool.cibuildwheel] in
# pyproject.toml), also used for the release archive and Debian packages
# under .github/scripts. Pinocchio and HPIPM are prebuilt with these settings by
# .github/scripts/build-wheel-deps.sh, and ssqp must be built with the same
# ones, so both load this file
set(SSQP_MARCH_NATIVE OFF CACHE BOOL "")
set(SSQP_ARCH_FLAG "-march=x86-64-v3" CACHE STRING "")
set(SSQP_BLASFEO_AUTO_TARGET OFF CACHE BOOL "")
set(SSQP_BLASFEO_TARGET "X64_INTEL_HASWELL" CACHE STRING "")
