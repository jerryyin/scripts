#!/bin/sh
set -x

wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
sudo apt-get update --allow-insecure-repositories -qq && sudo apt-get install -f -y -qq kitware-archive-keyring

apt-get update && apt-get install -f -y -qq cmake ccache ninja-build clang lld

pip install numpy

ln -sf /zyin/iree ~/
