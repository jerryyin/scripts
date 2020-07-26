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

# clone the configuration to root
rm -rf .ssh && cp -r /data/.ssh ./.ssh
chmod 400 .ssh/id_rsa && ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts

git init && \
git remote add origin git@github.com:jerryyin/rc_files.git && \
git fetch origin && \
git checkout master

# Make vim-plug to intialize submodules: vimrc does it now
vim -E -s -u ~/.vimrc +PlugInstall +qall || true

# Set the locale
locale-gen en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

cp ~/.zsh/.zshrc ~/ && cp ~/.zsh/robbyrussell.zsh-theme ~/.oh-my-zsh/themes/

# Clone scripts
if [ ! -d scripts ]; then
    git clone git@github.com:jerryyin/scripts.git
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

source ~/.bashrc
