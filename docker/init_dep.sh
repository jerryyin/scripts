#!/bin/sh
set -x

alias dockerInstall='DEBIAN_FRONTEND=noninteractive apt-get install -f -y'

apt-get update --allow-insecure-repositories 

dockerInstall software-properties-common  # Install add-apt-repository
dockerInstall apt-transport-https         # Dependency from kitware, for https
dockerInstall wget
                   
# PPA:  TODO remove when it becomes default ubuntu package
# vim8 packge ppa.
add-apt-repository -y ppa:jonathonf/vim
# universal ctags ppa.
add-apt-repository -y ppa:hnakamur/universal-ctags
# gnu global ppa.
add-apt-repository -y ppa:dns/gnu
# cmake, dependent on apt-transport-https. Refer to https://apt.kitware.com
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add -
apt-add-repository 'deb https://apt.kitware.com/ubuntu/ xenial main'

# Install misc pkgs
dockerInstall apt-utils
dockerInstall ssh
dockerInstall curl
dockerInstall universal-ctags
dockerInstall global
dockerInstall cscope
dockerInstall git
dockerInstall zsh
dockerInstall vim
dockerInstall stow
dockerInstall xclip
dockerInstall locales
dockerInstall python-autopep8
dockerInstall clang-format
dockerInstall gdb
dockerInstall tmux
dockerInstall cmake       # MLIR package
dockerInstall ninja-build # MLIR package

apt-get clean && rm -rf /var/lib/apt/lists/*

# install tmux plugin manager
if [ ! -d .tmux/plugins/tpm ]; then
    git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm
fi

# zsh
if [ ! -d .oh-my-zsh ]; then
    echo "Y" | sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
fi
