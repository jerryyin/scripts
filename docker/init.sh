#!/bin/sh
set -x

apt-get update && apt-get -y install sudo

shopt -s expand_aliases
alias dockerInstall='DEBIAN_FRONTEND=noninteractive sudo apt-get install -f -y'

sudo apt-get update --allow-insecure-repositories

dockerInstall software-properties-common  # Install add-apt-repository
dockerInstall apt-transport-https         # Dependency from kitware, for https
dockerInstall wget

# PPA:  TODO remove when it becomes default ubuntu package
# vim8 packge ppa.
add-apt-repository -y ppa:jonathonf/vim
# cmake, dependent on apt-transport-https. Refer to https://apt.kitware.com
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ xenial main'

# Install misc pkgs (For macos: the_silver_searcher)
dockerInstall ca-certificates apt-utils ssh curl cscope git vim stow xclip locales python3-dev \
              python3-autopep8 gdb zsh fonts-powerline tmux silversearcher-ag
# https://github.com/google/llvm-premerge-checks/blob/master/containers/base-debian/Dockerfile
dockerInstall clang-10 lld-10 clang-tidy-10 clang-format-10 cmake ninja-build 
# https://github.com/universal-ctags/ctags/blob/master/docs/autotools.rst
dockerInstall gcc make pkg-config autoconf automake python3-docutils \
              libseccomp-dev libjansson-dev libyaml-dev libxml2-dev

sudo apt-get clean && rm -rf /var/lib/apt/lists/*

# install tmux plugin manager
if [ ! -d .tmux/plugins/tpm ]; then
    git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm
fi

# zsh
if [ ! -d .oh-my-zsh ]; then
    echo "Y" | sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
    git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
    git clone https://github.com/zsh-users/zsh-syntax-highlighting ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
    git clone https://github.com/zsh-users/zsh-history-substring-search ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-history-substring-search
fi

# Make zsh default shell
sudo chsh -s $(which zsh)

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

# Build latest universal ctags
git clone https://github.com/universal-ctags/ctags.git && cd ctags
./autogen.sh && ./configure && make -j$(nproc) && sudo make install
cd ~ && rm -rf ctags

# Build latest gtags(gnu global)
GLOBAL=global-6.6.9
wget https://ftp.gnu.org/pub/gnu/global/$GLOBAL.tar.gz
tar -xzf $GLOBAL.tar.gz && cd $GLOBAL
./configure --with-universal-ctags=/usr/local/bin/ctags && make -j$(nproc) && sudo make install
cd ~ && rm -rf $GLOBAL*

GDB=gdb-13.1
dockerInstall libgmp-dev
wget http://ftp.gnu.org/gnu/gdb/$GDB.tar.gz
tar -xzf $GDB.tar.gz && cd $GDB
./configure && make -j$(nproc) && sudo make install
