#!/bin/sh
set -x

shopt -s expand_aliases
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
dockerInstall apt-utils ssh curl universal-ctags global cscope git zsh
dockerInstall vim stow xclip locales python3-dev python3-autopep8 gdb tmux
# https://github.com/google/llvm-premerge-checks/blob/master/containers/base-debian/Dockerfile
dockerInstall clang-10 lld-10 clang-tidy-10 clang-format-10
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

# Make zsh default shell
chsh -s $(which zsh)

# rc files
if [ ! -d rc_files ]; then
    git clone https://github.com/jerryyin/rc_files.git
    for dotpath in $(find rc_files -name "\.*"); do
      rm "$(basename -- $dotpath)"
    done
    for dir in $(ls -d ~/rc_files/*/ | awk -F "/" "{print \$(NF-1)}"); do
      stow -d ~/rc_files $dir -v -R -t ~
    done
    git -C rc_files remote set-url origin git@github.com:jerryyin/rc_files.git
fi

# Git configurations
# git default user, password, ignore file
if [ ! -d .git ]; then
    git config --global user.email "zhuoryin@amd.com"
    git config --global user.name "jerryyin"
    git config --global pager.branch false
    git config --global core.excludesfile ~/.gitignore
fi

# Make vim-plug to intialize submodules: vimrc does it now
vim -E -s -u ~/.vimrc +PlugInstall +qall || true

# Clone scripts
if [ ! -d scripts ]; then
    git clone https://github.com/jerryyin/scripts.git
fi
git -C scripts remote set-url origin git@github.com:jerryyin/scripts.git

zsh
