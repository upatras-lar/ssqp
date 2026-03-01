# Use Ubuntu 24.04 (Noble Numbat) as base
FROM ubuntu:24.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install base dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    lsb-release \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    python3-numpy \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/apt/keyrings && curl http://robotpkg.openrobots.org/packages/debian/robotpkg.asc | tee /etc/apt/keyrings/robotpkg.asc

RUN echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" | tee /etc/apt/sources.list.d/robotpkg.list

# Install Eigen3 (from Ubuntu repositories)
RUN apt-get update && apt-get install -y libeigen3-dev && rm -rf /var/lib/apt/lists/*

# Install Pinocchio (from official packages)
RUN apt-get update && apt-get install -y robotpkg-py3*-pinocchio && rm -rf /var/lib/apt/lists/*

# Install nanobind
RUN python3 -m pip install --break-system-packages nanobind

# Install blasfeo
RUN git clone https://github.com/giaf/blasfeo.git && cd blasfeo && make shared_library -j$(nproc) && make install_shared && cd .. && rm -rf blasfeo

# Install hpipm
RUN git clone https://github.com/giaf/hpipm.git && cd hpipm && make shared_library -j$(nproc) && make install_shared && cd .. && rm -rf hpipm

# Compile nanobind.o
RUN cd /usr/local/lib/python3.12/dist-packages/nanobind && g++ src/nb_combined.cpp -std=c++17 -fvisibility=hidden -DNDEBUG -DNB_COMPACT_ASSERTIONS `python3-config --includes` -fPIC -I include -I ext/robin_map/include -O3 -fno-strict-aliasing -ffunction-sections -fdata-sections -c -o libnanobind.o

# Set environment variables for robotpkg
ENV PATH="/opt/openrobots/bin:${PATH}" \
    PKG_CONFIG_PATH="/opt/openrobots/lib/pkgconfig:${PKG_CONFIG_PATH}" \
    LD_LIBRARY_PATH="/opt/openrobots/lib:/opt/hpipm/lib:/opt/blasfeo/lib:${LD_LIBRARY_PATH}" \
    PYTHONPATH="/opt/openrobots/lib/python3/dist-packages:${PYTHONPATH}"

# Default command
CMD ["/bin/bash"]
