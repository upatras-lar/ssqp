FROM ubuntu:24.04

ARG NPROC=4

ARG DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------
# Basic build environment
# ------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    pkg-config \
    ca-certificates \
    curl \
    wget \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-numpy \
    python3-scipy \
    libeigen3-dev \
    libboost-all-dev \
    libyaml-cpp-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    libboost-program-options-dev \
    libboost-serialization-dev \
    libboost-thread-dev \
    libboost-test-dev \
    libboost-python-dev \
    liburdfdom-dev \
    libcppad-dev \
    libsimde-dev \
    libfmt-dev \
    libassimp-dev \
    libqhull-dev \
    liboctomap-dev \
    && rm -rf /var/lib/apt/lists/*


# ------------------------------------------------------------
# Python environment
# ------------------------------------------------------------

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONPATH="/usr/local/lib/python3.12/site-packages"

RUN pip install --upgrade \
    pip \
    setuptools \
    wheel \
    numpy \
    scipy

# -------------------------
# eigenpy (Pinocchio's python bindings depend on it)
# -------------------------
RUN git clone --recursive --branch v3.10.3 https://github.com/stack-of-tasks/eigenpy.git /opt/eigenpy

WORKDIR /opt/eigenpy/build
RUN cmake .. \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DPYTHON_EXECUTABLE=/opt/venv/bin/python \
    -DCMAKE_PREFIX_PATH=/usr/local
RUN ninja -j${NPROC} || ninja -j${NPROC}
RUN ninja install
RUN mkdir -p /usr/local/lib/cmake/eigenpy \
    && touch /usr/local/lib/cmake/eigenpy/pypa-metadata.cmake

RUN touch /usr/local/lib/cmake/eigenpy/pypa-metadata.cmake

# ------------------------------------------------------------
# Build/install coal (formerly hpp-fcl)
#
# Not available as an Ubuntu apt package (libhpp-fcl-dev only ever
# existed via the third-party robotpkg repo). The project was renamed
# hpp-fcl -> coal in late 2025; COAL_BACKWARD_COMPATIBILITY_WITH_HPP_FCL
# keeps find_package(hpp-fcl), #include <hpp/fcl/...>, and `import
# hppfcl` working so Pinocchio's BUILD_WITH_HPP_FCL flag below needs
# no changes.
# ------------------------------------------------------------

WORKDIR /opt/src

RUN git clone --recursive --branch v3.0.4 \
    https://github.com/coal-library/coal.git

WORKDIR /opt/src/coal

RUN mkdir build && cd build && \
    cmake .. \
        -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_PYTHON_INTERFACE=ON \
        -DBUILD_TESTING=OFF \
        -DCOAL_BACKWARD_COMPATIBILITY_WITH_HPP_FCL=ON \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DPYTHON_EXECUTABLE=/opt/venv/bin/python && \
    cmake --build . -j${NPROC} && \
    cmake --install .


# ------------------------------------------------------------
# Build/install Pinocchio
# ------------------------------------------------------------

WORKDIR /opt/src

RUN git clone --recursive --branch v3.9.0 \
    https://github.com/stack-of-tasks/pinocchio.git

WORKDIR /opt/src/pinocchio

RUN mkdir build && cd build && \
    cmake .. \
        -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_PYTHON_INTERFACE=ON \
        -DBUILD_WITH_COLLISION_SUPPORT=ON \
        -DBUILD_WITH_HPP_FCL=ON \
        -DBUILD_TESTING=OFF \
        -DBUILD_BENCHMARK=OFF \
        -DBUILD_DOCUMENTATION=OFF \
        -DBUILD_WITH_AUTODIFF_SUPPORT=ON \
        -DBUILD_WITH_OPENMP_SUPPORT=ON \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DPYTHON_EXECUTABLE=/opt/venv/bin/python && \
    cmake --build . -j${NPROC} && \
    cmake --install .


# ------------------------------------------------------------
# Build/install ProxSuite
# ------------------------------------------------------------

WORKDIR /opt/src

RUN git clone --recursive --branch v0.7.2 \
    https://github.com/Simple-Robotics/proxsuite.git

WORKDIR /opt/src/proxsuite

RUN mkdir build && cd build && \
    cmake .. \
        -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_TESTING=OFF \
        -DBUILD_PYTHON_INTERFACE=ON \
        -DBUILD_WITH_VECTORIZATION_SUPPORT=ON \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DPYTHON_EXECUTABLE=/opt/venv/bin/python && \
    cmake --build . -j${NPROC} && \
    cmake --install .

# ------------------------------------------------------------
# Build/install mimalloc
# ------------------------------------------------------------

WORKDIR /opt/src

RUN git clone --depth 1 --branch v3.4.3 \
    https://github.com/microsoft/mimalloc.git

WORKDIR /opt/src/mimalloc

RUN cmake -S . -B build \
        -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DMI_BUILD_SHARED=ON \
        -DMI_BUILD_STATIC=ON \
        -DMI_BUILD_TESTS=OFF \
    && cmake --build build -j${NPROC} \
    && cmake --install build \
    && ldconfig

# ============================================================
# example-robot-data
# ============================================================

WORKDIR /opt/src

RUN git clone --recursive --branch v4.4.0 \
    https://github.com/Gepetto/example-robot-data.git

WORKDIR /opt/src/example-robot-data

RUN cmake -S . -B build \
        -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_PYTHON_INTERFACE=ON \
        -DBUILD_TESTING=OFF \
        -DPYTHON_EXECUTABLE=/opt/venv/bin/python \
    && cmake --build build -j${NPROC} \
    && cmake --install build \
    && ldconfig

# ------------------------------------------------------------
# Build/install Aligator
# ------------------------------------------------------------

WORKDIR /opt/src

RUN git clone --recursive --branch v0.19.0 \
    https://github.com/Simple-Robotics/aligator.git

WORKDIR /opt/src/aligator

RUN mkdir build && cd build && \
    cmake .. \
        -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_TESTING=OFF \
        -DBUILD_BENCHMARKS=OFF \
        -DBUILD_PYTHON_INTERFACE=ON \
        -DBUILD_EXAMPLES=ON \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DPYTHON_EXECUTABLE=/opt/venv/bin/python \
        -DCMAKE_PREFIX_PATH="/usr/local" && \
    cmake --build . -j${NPROC} && \
    cmake --install .


# ------------------------------------------------------------
# Python runtime sanity check
# ------------------------------------------------------------

RUN python -c "import numpy; print('NumPy:', numpy.__version__)" && \
    python -c "import pinocchio; print('Pinocchio:', pinocchio.__version__)" && \
    python -c "import aligator; print('Aligator imported successfully')"


# ------------------------------------------------------------
# Workspace
# ------------------------------------------------------------

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1

CMD ["/bin/bash"]