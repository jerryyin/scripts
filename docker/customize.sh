#!/bin/sh
set -x

if ! grep -q zsh ~/.bashrc; then
    echo "# Launch Zsh
    if [ -t 1 ]; then
      cd ~
      exec zsh
    fi" >> ~/.bashrc
fi

# clone the configuration to root
if [ ! -f ~/.ssh/id_rsa ]; then
    rm -rf ~/.ssh
    # Note: Docker needs to mount home directory to /data
    cp -r /data/.ssh ~/.ssh
    chmod 400 ~/.ssh/id_rsa && ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts
fi

if [ ! -d rc_files ]; then
    git clone git@github.com:jerryyin/rc_files.git
    for dotpath in $(find rc_files -name "\.*"); do
      rm "$(basename -- $dotpath)"
    done
    for dir in $(ls -d ~/rc_files/*/ | awk -F "/" "{print \$(NF-1)}"); do
      stow -d ~/rc_files $dir -v -R -t ~
    done
fi

# Git configurations
# git default user, password, ignore file
if [ ! -d .git ]; then
    git config --global user.email "zhuoryin@amd.com" && \
    git config --global user.name "jerryyin" && \
    git config --global core.excludesfile ~/.gitignore
fi

# Make vim-plug to intialize submodules: vimrc does it now
vim -E -s -u ~/.vimrc +PlugInstall +qall || true

# Clone scripts
if [ ! -d Playground/scripts ]; then
    mkdir -p Playground/scripts
    git clone git@github.com:jerryyin/scripts.git Playground/scripts
fi

if [ ! -d Documents/notes ]; then
    mkdir -p Documents/notes
    git clone git@github.com:jerryyin/notes.git Documents/notes
fi

# Hack from https://askubuntu.com/questions/64387/cannot-successfully-source-bashrc-from-a-shell-script
# eval "$(cat ~/.bashrc | tail -n +10)"
