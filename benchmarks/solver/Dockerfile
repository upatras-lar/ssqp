FROM ubuntu:24.04

ARG NPROC=4
ARG DEBIAN_FRONTEND=noninteractive

ENV DEBIAN_FRONTEND=${DEBIAN_FRONTEND}

# ============================================================
# System/build dependencies
# ============================================================

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
    python-is-python3 \
    swig \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libmetis-dev \
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
    libconsole-bridge-dev \
    libtinyxml-dev \
    libcppad-dev \
    libsimde-dev \
    libfmt-dev \
    libassimp-dev \
    libqhull-dev \
    liboctomap-dev \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Python environment
# ============================================================

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/usr/local/lib/python3.12/site-packages:/opt/venv/lib/python3.12/site-packages"
ENV LD_LIBRARY_PATH="/usr/local/lib"

RUN pip install --upgrade \
    pip \
    setuptools \
    wheel \
    numpy \
    scipy

WORKDIR /opt

# ============================================================
# MUMPS
# IPOPT requires a linear solver backend at runtime.
# ============================================================

RUN git clone https://github.com/coin-or-tools/ThirdParty-Mumps.git /opt/ThirdParty-Mumps \
    && cd /opt/ThirdParty-Mumps \
    && ./get.Mumps \
    && mkdir build \
    && cd build \
    && ../configure --prefix=/usr/local \
    && make \
    && make install

ENV PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH}"

# ============================================================
# IPOPT
# ============================================================

RUN git clone https://github.com/coin-or/Ipopt.git /opt/ipopt \
    && cd /opt/ipopt \
    && git submodule update --init --recursive \
    && mkdir build \
    && cd build \
    && ../configure \
        --prefix=/usr/local \
        --enable-shared \
    && make -j${NPROC} \
    && make install

# ============================================================
# CasADi
# Builds its pinned Fatrop / HPIPM / BLASFEO dependencies.
# ============================================================

RUN git clone --recursive https://github.com/casadi/casadi.git /opt/casadi

WORKDIR /opt/casadi
RUN mkdir build

WORKDIR /opt/casadi/build

RUN cmake .. \
    -G "Unix Makefiles" \
    -DWITH_PYTHON=ON \
    -DPYTHON_EXECUTABLE=/opt/venv/bin/python \
    -DWITH_FATROP=ON \
    -DWITH_BUILD_FATROP=ON \
    -DWITH_BUILD_HPIPM=ON \
    -DWITH_BUILD_BLASFEO=ON \
    -DWITH_IPOPT=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local

RUN make -j${NPROC} && make install

# ============================================================
# eigenpy
# Pinocchio's Python bindings depend on it.
# ============================================================

RUN git clone --recursive --branch v3.10.3 \
    https://github.com/stack-of-tasks/eigenpy.git /opt/eigenpy

WORKDIR /opt/eigenpy/build

RUN cmake .. \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DPYTHON_EXECUTABLE=/opt/venv/bin/python \
    -DCMAKE_PREFIX_PATH=/usr/local \
    && ninja -j${NPROC} \
    && ninja install \
    && mkdir -p /usr/local/lib/cmake/eigenpy \
    && touch /usr/local/lib/cmake/eigenpy/pypa-metadata.cmake

# ============================================================
# coal (formerly hpp-fcl)
# Provides collision support for Pinocchio.
# ============================================================

WORKDIR /opt/src

RUN git clone --recursive --branch v3.0.4 \
    https://github.com/coal-library/coal.git

WORKDIR /opt/src/coal

RUN cmake -S . -B build \
        -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_PYTHON_INTERFACE=ON \
        -DBUILD_TESTING=OFF \
        -DCOAL_BACKWARD_COMPATIBILITY_WITH_HPP_FCL=ON \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DPYTHON_EXECUTABLE=/opt/venv/bin/python \
    && cmake --build build -j${NPROC} \
    && cmake --install build

# ============================================================
# Pinocchio
# Includes:
#   - Python interface
#   - CasADi support
#   - collision / hpp-fcl compatibility
#   - autodiff support
#   - OpenMP support
# ============================================================

WORKDIR /opt/src

RUN git clone --recursive --branch v3.9.0 \
    https://github.com/stack-of-tasks/pinocchio.git

WORKDIR /opt/src/pinocchio

RUN cmake -S . -B build \
        -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_PREFIX_PATH=/usr/local \
        -DBUILD_PYTHON_INTERFACE=ON \
        -DBUILD_WITH_CASADI_SUPPORT=ON \
        -DBUILD_WITH_COLLISION_SUPPORT=ON \
        -DBUILD_WITH_HPP_FCL=ON \
        -DBUILD_WITH_AUTODIFF_SUPPORT=ON \
        -DBUILD_WITH_OPENMP_SUPPORT=ON \
        -DBUILD_TESTING=OFF \
        -DBUILD_BENCHMARK=OFF \
        -DBUILD_DOCUMENTATION=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DPYTHON_EXECUTABLE=/opt/venv/bin/python \
    && cmake --build build -j${NPROC} \
    && cmake --install build

# ============================================================
# ProxSuite
# ============================================================

WORKDIR /opt/src

RUN git clone --recursive --branch v0.7.2 \
    https://github.com/Simple-Robotics/proxsuite.git

WORKDIR /opt/src/proxsuite

RUN cmake -S . -B build \
        -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_TESTING=OFF \
        -DBUILD_PYTHON_INTERFACE=ON \
        -DBUILD_WITH_VECTORIZATION_SUPPORT=ON \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DPYTHON_EXECUTABLE=/opt/venv/bin/python \
    && cmake --build build -j${NPROC} \
    && cmake --install build

# ============================================================
# Crocoddyl
# ============================================================

WORKDIR /opt/src

RUN git clone --recursive --branch v3.0.0 \
    https://github.com/loco-3d/crocoddyl.git

WORKDIR /opt/src/crocoddyl

RUN cmake -S . -B build \
        -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_PREFIX_PATH=/usr/local \
        -DPython_EXECUTABLE=/opt/venv/bin/python \
        -DPython3_EXECUTABLE=/opt/venv/bin/python \
        -DBUILD_PYTHON_INTERFACE=ON \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_TESTING=OFF \
        -DBUILD_BENCHMARK=OFF \
        -DGENERATE_PYTHON_STUBS=OFF \
    && cmake --build build -j${NPROC} \
    && cmake --install build

# ============================================================
# mim_solvers
# ============================================================

WORKDIR /opt/src

RUN git clone --recursive https://github.com/machines-in-motion/mim_solvers.git

WORKDIR /opt/src/mim_solvers

RUN cmake -S . -B build \
        -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DPYTHON_EXECUTABLE=/opt/venv/bin/python \
        -DBUILD_PYTHON_INTERFACE=ON \
        -DBUILD_WITH_PROXSUITE=ON \
        -DBUILD_TESTING=OFF \
    && cmake --build build -j${NPROC} \
    && cmake --install build

# ============================================================
# mimalloc
# ============================================================

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

# ============================================================
# Aligator
# ============================================================

WORKDIR /opt/src

RUN git clone --recursive --branch v0.19.0 \
    https://github.com/Simple-Robotics/aligator.git

WORKDIR /opt/src/aligator

RUN cmake -S . -B build \
        -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_PREFIX_PATH=/usr/local \
        -DBUILD_TESTING=OFF \
        -DBUILD_BENCHMARKS=OFF \
        -DBUILD_PYTHON_INTERFACE=ON \
        -DBUILD_EXAMPLES=ON \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DPYTHON_EXECUTABLE=/opt/venv/bin/python \
    && cmake --build build -j${NPROC} \
    && cmake --install build

# ============================================================
# Runtime sanity checks
# ============================================================

ENV PYTHONPATH="bindings:${PYTHONPATH}"

RUN python -c "import numpy; print('NumPy:', numpy.__version__)" \
    && python -c "import scipy; print('SciPy:', scipy.__version__)" \
    && python -c "import eigenpy; print('eigenpy imported successfully')" \
    && python -c "import pinocchio; print('Pinocchio:', pinocchio.__version__)" \
    && python -c "import proxsuite; print('ProxSuite imported successfully')" \
    && python -c "import crocoddyl; print('Crocoddyl imported successfully')" \
    && python -c "import mim_solvers; print('mim_solvers imported successfully')" \
    && python -c "import aligator; print('Aligator imported successfully')" \
    && python -c "import casadi; print('CasADi:', casadi.__version__)"

# ============================================================
# Workspace
# ============================================================

WORKDIR /workspace

CMD ["/bin/bash"]
