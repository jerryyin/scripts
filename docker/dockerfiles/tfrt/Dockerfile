FROM ubuntu:16.04
MAINTAINER Deven Desai <deven.desai.amd@gmail.com>

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update

RUN apt-get install -y --no-install-recommends \
    apt-transport-https\
    autoconf \
    automake \
    build-essential \
    cmake \
    curl \
    emacs24-nox \
    ffmpeg \
    git \
    libcurl4-openssl-dev \
    libtool \
    libssl-dev \
    lsb-release \
    mlocate \
    openjdk-8-jdk \
    openjdk-8-jre-headless \
    pkg-config \
    python-virtualenv \
    python3-dev \
    python3-setuptools \
    rsync \
    software-properties-common \
    sudo \
    swig \
    unzip \
    vim \
    wget \
    xz-utils \
    zip \
    zlib1g-dev

# Install ca-certificates, and update the certificate store.
RUN apt-get install -y ca-certificates-java && update-ca-certificates -f

# tfrt specific requirements

# Install bazel 3.1.0
RUN curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
RUN echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
RUN apt-get update && apt-get install -y bazel-3.1.0
RUN update-alternatives --install /usr/bin/bazel bazel /usr/bin/bazel-3.1.0 310

# Install clang
RUN bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-10 10
RUN update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-10 10

# Install gcc8 (tfrt needs libstdc++8 or greater)
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y gcc-8 g++-8

ENV HOME  /root/
WORKDIR $HOME

# Clone the TFRT repo
RUN git clone https://github.com/tensorflow/runtime tfrt

# # deven specific customization

# ARG user=deven

# RUN useradd --create-home $user
# RUN usermod --append --groups sudo $user
# RUN passwd -d $user

# USER $user
# WORKDIR /home/$user

# # Run an dummy sudo command to suppress the following message
# # (which will otherwise appear evertyime you start the container)
# #
# # To run a command as administrator (user "root"), use "sudo <command>".
# # See "man sudo_root" for details.
# #
# RUN sudo echo

# RUN git clone https://github.com/deven-amd/.emacs.d.git

# RUN git config --global user.email "deven.desai.amd@gmail.com"
# RUN git config --global user.name "Deven Desai"
# RUN git config --global core.editor "emacs"

CMD /bin/bash
