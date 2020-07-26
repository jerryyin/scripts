#!/bin/sh
set -x

# PPA:  TODO remove when it becomes default ubuntu package
# vim8 packge ppa.
add-apt-repository -y ppa:jonathonf/vim
# universal ctags ppa.
add-apt-repository -y ppa:hnakamur/universal-ctags
# gnu global ppa.
add-apt-repository -y ppa:dns/gnu

# Install misc pkgs
apt-get update --allow-insecure-repositories && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  apt-utils \
  apt-transport-https \
  ssh \
  curl \
  universal-ctags \
  global \
  cscope \
  git \
  zsh \
  vim \
  xclip \
  locales \
  python-autopep8 \
  clang-format \
  gdb \
  wget \
  cmake \
  ninja-build \
  tmux && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# cmake, dependent on apt-transport-https. Refer to https://apt.kitware.com
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - && \
apt-add-repository 'deb https://apt.kitware.com/ubuntu/ xenial main'

apt-get update --allow-insecure-repositories && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  cmake && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# install tmux plugin manager
if [ ! -d .tmux/plugins/tpm ]; then
    git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm
fi

# zsh
if [ ! -d .oh-my-zsh ]; then
    echo "Y" | sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
fi
