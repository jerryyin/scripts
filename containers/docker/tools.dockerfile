ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgmp-dev \
    autoconf \
    automake \
    libtool \
    wget \
    pkg-config \
    libncurses5-dev \
    libmpc-dev \
    python3-docutils \
    libseccomp-dev \
    libjansson-dev \
    libyaml-dev \
    libxml2-dev \
    zlib1g-dev \
    liblzma-dev \
    git \
    texinfo \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Build Universal Ctags
WORKDIR /tmp
RUN git clone https://github.com/universal-ctags/ctags.git && \
    cd ctags && \
    ./autogen.sh && \
    ./configure CFLAGS="-static" CXXFLAGS="-w" LDFLAGS="-static" --enable-static --disable-iconv && \
    make -j$(nproc) && \
    make install

# Build GNU Global -> static build doesn't work
# Right now failing because of network failure
#ENV GLOBAL_VERSION=6.6.14
#WORKDIR /tmp
#RUN wget -q https://ftp.gnu.org/pub/gnu/global/global-${GLOBAL_VERSION}.tar.gz && \
#    tar -xzf global-${GLOBAL_VERSION}.tar.gz && \
#    cd global-${GLOBAL_VERSION} && \
#    ./configure --with-universal-ctags=/tools/usr/local/bin/ctags && \
#    make -j$(nproc) && \
#    make install

# Build GDB -> use rocGDB instead
#ENV GDB_VERSION=15.1
#WORKDIR /tmp
#RUN wget -q http://ftp.gnu.org/gnu/gdb/gdb-${GDB_VERSION}.tar.gz && \
#    tar -xzf gdb-${GDB_VERSION}.tar.gz && \
#    cd gdb-${GDB_VERSION} && \
#    ./configure CFLAGS="-w -Wno-deprecated -static" CXXFLAGS="-w" LDFLAGS="-static" --enable-static --disable-shared && \
#    make -j$(nproc) && \
#    make DESTDIR=/tools install

# Replace /usr/local/bin with packed binary, and depedent so under /usr/local/lib/packelf_shared
WORKDIR /tmp
RUN git clone https://github.com/jerryyin/packelf.git && \
    cd packelf && \
    bash batchpack.sh
