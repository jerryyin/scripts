#!/bin/sh
set -x

# Delete rocm sources if any, they tend to cause problem with apt update
find /etc/apt \( -name "*amdgpu*" -o -name "*rocm*" \) -delete

apt-get update && apt-get -y install sudo software-properties-common apt-utils

# Fixing /etc/host file, refer to https://askubuntu.com/questions/59458/error-message-sudo-unable-to-resolve-host-none
if ! grep -q "$HOSTNAME" /etc/hosts; then
    echo $(hostname -I | cut -d\  -f1) $(hostname) | sudo -h 127.0.0.1 tee -a /etc/hosts
fi

shopt -s expand_aliases

sudo apt-get update --allow-insecure-repositories

# PPA:  TODO remove when it becomes default ubuntu package
# vim8 packge ppa. This is unecessary in ubuntu 24.04
#sudo add-apt-repository -y ppa:jonathonf/vim
# neovim ppa
sudo add-apt-repository -y ppa:neovim-ppa/stable
# setup nodejs
curl -fsSL https://deb.nodesource.com/setup_current.x | sudo -E bash -

# Install misc pkgs (For macos: the_silver_searcher)
sudo DEBIAN_FRONTEND=noninteractive apt-get install -f -y  \
     git zsh fonts-powerline tmux silversearcher-ag less stow nodejs neovim curl vim wget \
     python-is-python3 gdb

# rc files
if [ ! -d rc_files ]; then
    git clone https://github.com/jerryyin/rc_files.git
    git -C rc_files remote set-url origin git@github.com:jerryyin/rc_files.git
    bash rc_files/install.sh
fi

# Clone scripts
if [ ! -d scripts ]; then
    git clone https://github.com/jerryyin/scripts.git
    git -C scripts remote set-url origin git@github.com:jerryyin/scripts.git
fi

wget -q https://gist.githubusercontent.com/jerryyin/8da8f21024b5d4ef853b171771def28c/raw/d769419709807acfeff1a3c7a7f4acbc44b76b28/AMD_CA.crt
sudo cp AMD_CA.crt /usr/local/share/ca-certificates
sudo update-ca-certificates

sudo apt-get install -y locales && locale-gen en_US.UTF-8
