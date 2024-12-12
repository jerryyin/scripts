#!/bin/sh
set -x

# Delete rocm sources if any, they tend to cause problem with apt update
find /etc/apt \( -name "*amdgpu*" -o -name "*rocm*" \) -delete

apt-get update && apt-get -y install sudo software-properties-common
# Fixing /etc/host file, refer to https://askubuntu.com/questions/59458/error-message-sudo-unable-to-resolve-host-none
echo $(hostname -I | cut -d\  -f1) $(hostname) | sudo -h 127.0.0.1 tee -a /etc/hosts

shopt -s expand_aliases

sudo apt-get update --allow-insecure-repositories -qq

# PPA:  TODO remove when it becomes default ubuntu package
# vim8 packge ppa.
sudo add-apt-repository -y ppa:jonathonf/vim
# neovim ppa
sudo add-apt-repository -y ppa:neovim-ppa/stable
# setup nodejs
curl -fsSL https://deb.nodesource.com/setup_current.x | sudo -E bash -

# Install misc pkgs (For macos: the_silver_searcher)
sudo DEBIAN_FRONTEND=noninteractive apt-get install -f -y -qq  \
     git zsh fonts-powerline tmux silversearcher-ag less stow nodejs neovim curl vim wget

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

# install tmux plugin manager
if [ ! -d .tmux/plugins/tpm ]; then
    git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm
    # Install plugins, this is dependent on existence of .tmux.conf
    tmux start-server && tmux new-session -d && \
    ~/.tmux/plugins/tpm/scripts/install_plugins.sh && tmux kill-server
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
# Install coc dependencies
vim --not-a-term +":CocInstall coc-json coc-tsserver coc-pyright" +q

# Clone scripts
if [ ! -d scripts ]; then
    git clone https://github.com/jerryyin/scripts.git
fi
git -C scripts remote set-url origin git@github.com:jerryyin/scripts.git

# Configure neovim
mkdir -p ~/.local/share/nvim && ln -s ~/.vim ~/.local/share/nvim/site
mkdir -p ~/.config/nvim && ln -s ~/.vimrc ~/.config/nvim/init.vim

wget -q https://gist.githubusercontent.com/jerryyin/8da8f21024b5d4ef853b171771def28c/raw/d769419709807acfeff1a3c7a7f4acbc44b76b28/AMD_CA.crt
sudo cp AMD_CA.crt /usr/local/share/ca-certificates
sudo update-ca-certificates

sudo apt-get install -y locales && locale-gen en_US.UTF-8

# Make zsh default shell, and install zsh dependencies
sudo chsh -s $(which zsh)
# Remove wait because otherwise waited plugins won't be initialized
sed "/zinit ice/s/wait'[^']*'//g" .zshrc > /tmp/zshrc_processed
zsh -c "source /tmp/zshrc_processed; exit"
