FROM ubuntu:24.04

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
    && rm -rf /var/lib/apt/lists/*


WORKDIR /opt

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
    && make -j$(nproc) \
    && make install

# -------------------------
# BLASFEO
# -------------------------
RUN git clone https://github.com/giaf/blasfeo.git \
    && cd blasfeo \
    && mkdir build \
    && cd build \
    && cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
    && make -j$(nproc) \
    && make install


# -------------------------
# HPIPM
# -------------------------
RUN git clone https://github.com/giaf/hpipm.git \
    && cd hpipm \
    && mkdir build \
    && cd build \
    && cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBLASFEO_PATH=/usr/local \
        -DHPIPM_BUILD_TESTS=OFF \
        -DHPIPM_BUILD_EXAMPLES=OFF \
    && make -j$(nproc) \
    && make install


# -------------------------
# FATROP
# -------------------------
RUN git clone https://github.com/meco-group/fatrop.git \
    && cd fatrop \
    && mkdir build \
    && cd build \
    && cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
    && make -j$(nproc) \
    && make install


# -------------------------
# CasADi
# -------------------------
RUN git clone --recursive https://github.com/casadi/casadi.git

WORKDIR /opt/casadi
RUN mkdir build

WORKDIR /opt/casadi/build

RUN cmake .. \
    -G Ninja \
    -DWITH_PYTHON=ON \
    -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    -DWITH_FATROP=ON \
    -DWITH_IPOPT=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_BUILD_REQUIRED=ON \
    -DCMAKE_INSTALL_PREFIX=/usr/local

RUN ninja
RUN ninja install


ENV LD_LIBRARY_PATH=/usr/local/lib
ENV PYTHONPATH=/usr/local/lib/python3/dist-packages

WORKDIR /workspace

CMD ["bash"]