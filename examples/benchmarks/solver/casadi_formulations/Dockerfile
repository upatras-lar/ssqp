FROM ubuntu:24.04

ARG NPROC=4

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    pkg-config \
    python3 \
    python3-dev \
    swig \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libmetis-dev \
    python3-numpy \
    python-is-python3 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# -------------------------
# MUMPS (linear solver IPOPT needs to actually run)
# -------------------------
RUN git clone https://github.com/coin-or-tools/ThirdParty-Mumps.git /opt/ThirdParty-Mumps \
    && cd /opt/ThirdParty-Mumps \
    && ./get.Mumps \
    && mkdir build \
    && cd build \
    && ../configure --prefix=/usr/local \
    && make -j${NPROC} \
    && make install

ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH

# -------------------------
# IPOPT
# -------------------------
RUN git clone https://github.com/coin-or/Ipopt.git ipopt \
    && cd ipopt \
    && git submodule update --init --recursive \
    && mkdir build \
    && cd build \
    && ../configure \
        --prefix=/usr/local \
        --enable-shared \
    && make -j${NPROC} \
    && make install

# -------------------------
# CasADi (builds its own pinned Fatrop/HPIPM/BLASFEO)
# -------------------------
RUN git clone --recursive https://github.com/casadi/casadi.git /opt/casadi

WORKDIR /opt/casadi
RUN mkdir build

WORKDIR /opt/casadi/build

RUN cmake .. \
    -G "Unix Makefiles" \
    -DWITH_PYTHON=ON \
    -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    -DWITH_FATROP=ON \
    -DWITH_BUILD_FATROP=ON \
    -DWITH_BUILD_HPIPM=ON \
    -DWITH_BUILD_BLASFEO=ON \
    -DWITH_IPOPT=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local

RUN make -j${NPROC}
RUN make install

# -------------------------
# System deps for Pinocchio
# -------------------------
RUN apt-get update && apt-get install -y \
    libeigen3-dev \
    libboost-all-dev \
    liburdfdom-dev \
    libconsole-bridge-dev \
    libtinyxml-dev \
    python3-scipy \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# eigenpy (Pinocchio's python bindings depend on it)
# -------------------------
RUN git clone --recursive --branch v3.10.1 https://github.com/stack-of-tasks/eigenpy.git /opt/eigenpy

WORKDIR /opt/eigenpy/build
RUN cmake .. \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    -DCMAKE_PREFIX_PATH=/usr/local
RUN ninja -j${NPROC} || ninja -j${NPROC}
RUN ninja install
RUN mkdir -p /usr/local/lib/cmake/eigenpy \
    && touch /usr/local/lib/cmake/eigenpy/pypa-metadata.cmake

RUN touch /usr/local/lib/cmake/eigenpy/pypa-metadata.cmake

# -------------------------
# Pinocchio (with CasADi support, built against our own CasADi)
# -------------------------
RUN git clone --recursive https://github.com/stack-of-tasks/pinocchio.git /opt/pinocchio

WORKDIR /opt/pinocchio/build
RUN cmake .. \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_PREFIX_PATH=/usr/local \
    -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    -DBUILD_WITH_CASADI_SUPPORT=ON \
    -DBUILD_WITH_COLLISION_SUPPORT=OFF \
    -DBUILD_TESTING=OFF

RUN ninja -j${NPROC} || ninja -j${NPROC}
RUN ninja install

WORKDIR /workspace

# Cover the possible python install locations for both CasADi and Pinocchio
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV PYTHONPATH=/usr/local/lib/python3/dist-packages:/usr/local/lib/python3.12/dist-packages:/usr/local/lib/python3.12/site-packages:/usr/local/lib/python3/site-packages

ENV LD_LIBRARY_PATH=/usr/local/lib
ENV PYTHONPATH=/usr/local/lib/python3.12/site-packages

WORKDIR /workspace

CMD ["bash"]