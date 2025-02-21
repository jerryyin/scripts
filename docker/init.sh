#!/bin/sh
set -x

# Define log files
REGULAR_LOG="regular.log"
# Clear previous logs
> "$REGULAR_LOG"
# Redirect stdout to regular.log and stderr remains visible
exec 1>>"$REGULAR_LOG"

apt-get update && apt-get -y install sudo
# Fixing /etc/host file, refer to https://askubuntu.com/questions/59458/error-message-sudo-unable-to-resolve-host-none
echo $(hostname -I | cut -d\  -f1) $(hostname) | sudo -h 127.0.0.1 tee -a /etc/hosts

shopt -s expand_aliases
alias dockerInstall='sudo DEBIAN_FRONTEND=noninteractive apt-get install -f -y -qq '

sudo apt-get update --allow-insecure-repositories -qq

dockerInstall software-properties-common  # Install add-apt-repository
dockerInstall apt-transport-https         # Dependency from kitware, for https
dockerInstall wget gpg

# PPA:  TODO remove when it becomes default ubuntu package
# vim8 packge ppa.
add-apt-repository -y ppa:jonathonf/vim

# cmake, dependent on apt-transport-https. Refer to https://apt.kitware.com
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
sudo apt-get update --allow-insecure-repositories -qq
dockerInstall kitware-archive-keyring

# Install misc pkgs (For macos: the_silver_searcher)
dockerInstall ca-certificates apt-utils ssh curl cscope git vim stow xclip locales python3-dev \
              python3-autopep8 zsh fonts-powerline tmux silversearcher-ag less
# https://github.com/google/llvm-premerge-checks/blob/master/containers/base-debian/Dockerfile
dockerInstall clang-10 lld-10 clang-tidy-10 clang-format-10 cmake ninja-build

sudo apt-get clean && rm -rf /var/lib/apt/lists/*

# install tmux plugin manager
if [ ! -d .tmux/plugins/tpm ]; then
    git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm
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

# User needs explicitly add tools argument to turn on expensive setups
if [ "$1" = "tools" ]; then
  # Build latest universal ctags
  # https://github.com/universal-ctags/ctags/blob/master/docs/autotools.rst
  dockerInstall gcc make pkg-config autoconf automake python3-docutils \
              libseccomp-dev libjansson-dev libyaml-dev libxml2-dev texinfo
  git clone https://github.com/universal-ctags/ctags.git && cd ctags
  ./autogen.sh && ./configure CFLAGS="-w -Wno-deprecated" CXXFLAGS="-w"
  cd ~ && rm -rf ctags
  
  # Build latest gtags(gnu global)
  GLOBAL=global-6.6.13
  wget -q https://ftp.gnu.org/pub/gnu/global/$GLOBAL.tar.gz
  tar -xzf $GLOBAL.tar.gz && cd $GLOBAL
  ./configure --with-universal-ctags=/usr/local/bin/ctags CFLAGS="-w -Wno-deprecated" CXXFLAGS="-w" && make -j$(nproc) && sudo make install
  cd ~ && rm -rf $GLOBAL*
  
  GDB=gdb-15.1
  dockerInstall libgmp-dev
  wget -q http://ftp.gnu.org/gnu/gdb/$GDB.tar.gz
  tar -xzf $GDB.tar.gz && cd $GDB
  ./configure CFLAGS="-w -Wno-deprecated" CXXFLAGS="-w" && make -j$(nproc) && sudo make install
  cd ~ && rm -rf $GDB*
  # GDB pretty printers
  git clone https://github.com/koutheir/libcxx-pretty-printers.git
fi

# Install nodejs and neovim
curl -fsSL https://deb.nodesource.com/setup_current.x | sudo -E bash -
add-apt-repository -y ppa:neovim-ppa/stable
dockerInstall nodejs neovim
mkdir -p ~/.local/share/nvim && ln -s ~/.vim ~/.local/share/nvim/site

wget -q https://gist.githubusercontent.com/jerryyin/8da8f21024b5d4ef853b171771def28c/raw/d769419709807acfeff1a3c7a7f4acbc44b76b28/AMD_CA.crt
sudo cp AMD_CA.crt /usr/local/share/ca-certificates
sudo update-ca-certificates

# Create a heredoc that will be executed in zsh
zsh << EOF
# Ensure we're still redirecting to regular.log
exec 1>>regular.log

# Supressing tput output
export TERM=xterm-256color

# Install zsh plugins
source .zshrc

EOF
