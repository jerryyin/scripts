#!/bin/sh
set -x

# Delete rocm sources if any, they tend to cause problem with apt update
#find /etc/apt \( -name "*amdgpu*" -o -name "*rocm*" \) -delete
# Re-import gpg key to not have warnings all over the place
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/rocm.gpg

sudo apt-get update && sudo apt-get -y install sudo software-properties-common apt-utils curl

# Fixing /etc/host file, refer to https://askubuntu.com/questions/59458/error-message-sudo-unable-to-resolve-host-none
if ! grep -q "$HOSTNAME" /etc/hosts; then
    echo $(hostname -I | cut -d\  -f1) $(hostname) | sudo -h 127.0.0.1 tee -a /etc/hosts
fi

shopt -s expand_aliases

sudo apt-get update --allow-insecure-repositories

# Get Ubuntu codename (focal, jammy, noble, etc.)
CODENAME=$(lsb_release -sc 2>/dev/null || . /etc/os-release && echo "$VERSION_CODENAME")

if [ "$CODENAME" = "focal" ] || [ "$CODENAME" = "jammy" ]; then
  sudo add-apt-repository -y ppa:jonathonf/vim
else
  echo "Skipping jonathonf/vim PPA on $CODENAME"
fi

# neovim ppa
sudo add-apt-repository -y ppa:neovim-ppa/stable
# setup nodejs
curl -fsSL https://deb.nodesource.com/setup_current.x | sudo -E bash -

# Install misc pkgs (For macos: the_silver_searcher)
sudo DEBIAN_FRONTEND=noninteractive apt-get install -f -y  \
     git zsh fonts-powerline tmux silversearcher-ag less stow nodejs neovim vim wget \
     python-is-python3 gdb gist openssh-client

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

# Only download AMD_CA.crt if not already present (PVC persists across pods)
if [ ! -f /usr/local/share/ca-certificates/AMD_CA.crt ]; then
    wget -q -O /tmp/AMD_CA.crt https://gist.githubusercontent.com/jerryyin/8da8f21024b5d4ef853b171771def28c/raw/d769419709807acfeff1a3c7a7f4acbc44b76b28/AMD_CA.crt
    sudo cp /tmp/AMD_CA.crt /usr/local/share/ca-certificates
    sudo update-ca-certificates
    rm -f /tmp/AMD_CA.crt  # Clean up downloaded file
else
    echo "AMD_CA.crt already installed, skipping"
fi

sudo apt-get install -y locales && sudo locale-gen en_US.UTF-8
