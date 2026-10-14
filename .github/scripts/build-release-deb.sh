#!/bin/bash
# Builds the libssqp-dev Debian package for the Ubuntu release it runs on,
# with the same settings as the PyPI wheels (cmake/WheelConfig.cmake). The
# package release suffix names that Ubuntu release, so the packages for
# different releases can sit side by side
#
# Usage: build-release-deb.sh <project dir> <output dir>
set -euo pipefail

project=$(realpath "$1")
out=$(realpath "$2")
work=/tmp/ssqp-deb
release="1~ubuntu$(. /etc/os-release && echo "$VERSION_ID")"

cmake -S "$project" -B "$work/build" \
    -C "$project/cmake/WheelConfig.cmake" \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DSSQP_BUILD_PYTHON=OFF \
    -DSSQP_BUILD_TESTS=OFF
cmake --build "$work/build" --target ssqp -j"$(nproc)"

cpack --config "$work/build/CPackConfig.cmake" -G DEB \
    -D CPACK_DEBIAN_PACKAGE_RELEASE="$release" \
    -B "$work/package"
# GitHub may rename release assets with a ~ in their name, so the file name
# uses - instead, giving e.g. libssqp-dev_0.1.1-1-ubuntu24.04_amd64.deb
mkdir -p "$out"
for deb in "$work"/package/*.deb; do
    name=$(basename "$deb")
    cp "$deb" "$out/${name//\~/-}"
done
