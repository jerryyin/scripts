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

# PPA:  TODO remove when it becomes default ubuntu package
# vim8 packge ppa.
add-apt-repository -y ppa:jonathonf/vim
# neovim ppa
add-apt-repository -y ppa:neovim-ppa/stable

# Install misc pkgs (For macos: the_silver_searcher)
dockerInstall git vim zsh fonts-powerline tmux silversearcher-ag less stow nodejs neovim
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

# Install nodejs and neovim
curl -fsSL https://deb.nodesource.com/setup_current.x | sudo -E bash -
mkdir -p ~/.local/share/nvim && ln -s ~/.vim ~/.local/share/nvim/site
mkdir -p ~/.config/nvim && ln -s ~/.vimrc ~/.config/nvim/init.vim

wget -q https://gist.githubusercontent.com/jerryyin/8da8f21024b5d4ef853b171771def28c/raw/d769419709807acfeff1a3c7a7f4acbc44b76b28/AMD_CA.crt
sudo cp AMD_CA.crt /usr/local/share/ca-certificates
sudo update-ca-certificates

# Create a heredoc that will be executed in zsh
zsh << EOF
# Ensure we're still redirecting to regular.log
exec 1>>regular.log

# Supressing tput output
export TERM=xterm

# Install zsh plugins
source .zshrc

EOF
