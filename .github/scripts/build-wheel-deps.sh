#!/bin/bash
# Runs once per manylinux container before the wheel builds (cibuildwheel's
# before-all). Installs Eigen and prebuilds Pinocchio and HPIPM, so every
# Python version reuses them instead of building them again
#
# Usage: build-wheel-deps.sh <project dir>
set -euo pipefail

project=$1
work=/tmp/ssqp-wheel-deps

eigen_version=3.4.0
eigen_sha256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72

mkdir -p "$work"
curl -fsSL -o "$work/eigen.tar.gz" \
    "https://gitlab.com/libeigen/eigen/-/archive/$eigen_version/eigen-$eigen_version.tar.gz"
echo "$eigen_sha256  $work/eigen.tar.gz" | sha256sum -c -
tar xzf "$work/eigen.tar.gz" -C "$work"
cmake -S "$work/eigen-$eigen_version" -B "$work/eigen-build" -DBUILD_TESTING=OFF
cmake --install "$work/eigen-build"

# Package locations must match SSQP_PREBUILT_*_DIR in pyproject.toml
cmake -S "$project" -B "$work/build" \
    -C "$project/cmake/WheelConfig.cmake" \
    -DSSQP_BUILD_PYTHON=OFF \
    -DSSQP_BUILD_TESTS=OFF \
    -DSSQP_PINOCCHIO_PACKAGE_DIR=/opt/ssqp-deps/pinocchio \
    -DSSQP_HPIPM_PACKAGE_DIR=/opt/ssqp-deps/hpipm
cmake --build "$work/build" --target package-pinocchio package-hpipm -j"$(nproc)"
