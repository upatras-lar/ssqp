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
# Build/install Crocoddyl
# ------------------------------------------------------------

RUN git clone --recursive --branch v3.0.0 https://github.com/loco-3d/crocoddyl.git && \
    cd crocoddyl && \
    cmake -B build -DCMAKE_BUILD_TYPE=Release \
                   -DCMAKE_INSTALL_PREFIX=/usr/local \
                   -DPYTHON_EXECUTABLE=/opt/venv/bin/python \
                   -DBUILD_PYTHON_INTERFACE=ON \
                   -DBUILD_EXAMPLES=OFF \
                   -DBUILD_TESTING=OFF \
                   -DBUILD_BENCHMARK=OFF \
                   -DGENERATE_PYTHON_STUBS=OFF && \
    make -C build -j${NPROC} && \
    make -C build install

# ------------------------------------------------------------
# Build/install mim_solvers
# ------------------------------------------------------------

RUN git clone --recursive https://github.com/machines-in-motion/mim_solvers.git && \
    cd mim_solvers && \
    cmake -B build -DCMAKE_BUILD_TYPE=Release \
                   -DCMAKE_INSTALL_PREFIX=/usr/local \
                   -DPYTHON_EXECUTABLE=/opt/venv/bin/python \
                   -DBUILD_PYTHON_INTERFACE=ON \
                   -DBUILD_WITH_PROXSUITE=ON \
                   -DBUILD_TESTING=OFF && \
    make -C build -j${NPROC} && \
    make -C build install

# Set runtime workspace
WORKDIR /app

# Verification step
RUN python3 -c "import eigenpy; import pinocchio; import proxsuite; import crocoddyl; import mim_solvers; print('All libraries built and imported successfully!')"

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1

CMD ["/bin/bash"]