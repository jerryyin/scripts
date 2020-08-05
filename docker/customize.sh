#!/bin/sh
set -x

echo "# Launch Zsh
if [ -t 1 ]; then
  cd ~
  exec zsh
fi" >> ~/.bashrc

# Git configurations
# git default user, password, ignore file
if [ ! -d .git ]; then
    git config --global user.email "zhuoryin@amd.com" && \
    git config --global user.name "jerryyin" && \
    git config --global core.excludesfile ~/.gitignore
fi

# Set the locale
locale-gen en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# clone the configuration to root
rm -rf .ssh && cp -r /data/.ssh ./.ssh
chmod 400 .ssh/id_rsa && ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts

if [ ! -d rc_files ]; then
    rm ~/.zshrc ~/.oh-my-zsh/themes/robbyrussell.zsh-theme
    rm ~/.stow-global-ignore
    rm ~/.gitignore
    rm ~/.notags
    rm ~/.tmux.conf
    rm ~/.vimrc
    rm ~/.emacs
    git clone git@github.com:jerryyin/rc_files.git
    cd rc_files
    stow zsh
    stow stow
    stow git
    stow gtags
    stow tmux
    stow vim
    stow emacs
    cd ~
fi

# Make vim-plug to intialize submodules: vimrc does it now
vim -E -s -u ~/.vimrc +PlugInstall +qall || true

# Clone scripts
if [ ! -d scripts ]; then
    git clone git@github.com:jerryyin/scripts.git
fi

if [ ! -d Documents/notes ]; then
    mkdir -p Documents/notes
    git clone git@github.com:jerryyin/notes.git Documents/notes
fi

# Project specific
# Clone tf, run cscope
if [ ! -d tensorflow-upstream ]; then
    git clone git@github.com:ROCmSoftwarePlatform/tensorflow-upstream.git && \
    cd tensorflow-upstream && \
    git remote add google-upstream git@github.com:tensorflow/tensorflow.git && \
    find $(pwd)/tensorflow -type f -print > gtags.files && \
    gtags && \
    export TF2_BEHAVIOR=1 && \
    cd ~
fi

# Clean up
# temporary fix to #30287, TF fail to apply update when HOME dir is a git directory
rm -rf ~/.git

# Hack from https://askubuntu.com/questions/64387/cannot-successfully-source-bashrc-from-a-shell-script
eval "$(cat ~/.bashrc | tail -n +10)"
