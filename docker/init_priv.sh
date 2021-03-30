#!/bin/sh
set -x

# clone the configuration to root
if [ ! -f ~/.ssh/id_rsa ]; then
    rm -rf ~/.ssh
    # Note: Docker needs to mount home directory to /zyin
    cp -r /zyin/.ssh ~/.ssh
    chmod 400 ~/.ssh/id_rsa && ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts
fi

if [ ! -d Documents/notes ]; then
    mkdir -p Documents/notes
    git clone git@github.com:jerryyin/notes.git Documents/notes
fi
