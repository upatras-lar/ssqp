# Use Ubuntu 24.04 (Noble Numbat) as base
FROM ubuntu:24.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Base dependencies: a C++17 toolchain, CMake, git (for fetching vendored
# dependencies at build time), Python (for the ssqpy bindings), and Eigen3 +
# Boost (the only two dependencies ssqp's CMake build doesn't fetch and
# build from source itself; see cmake/Vendor*.cmake).
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-dev \
    python3-numpy \
    python-is-python3 \
    libeigen3-dev \
    libboost-filesystem-dev \
    libboost-serialization-dev \
    && rm -rf /var/lib/apt/lists/*

# Default command
CMD ["/bin/bash"]
