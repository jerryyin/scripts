#!/usr/bin/env bash
set -ex

#install emacs
apt-get update && apt-get install -y emacs25-nox
cd $HOME && rm -rf .emacs.d && git clone https://github.com/deven-amd/.emacs.d.git

cd $HOME
git clone https://github.com/ROCmSoftwarePlatform/tensorflow-upstream rocm-tf
cd $HOME/rocm-tf
/common/scripts/misc/git_config.sh

git remote add google_upstream https://github.com/tensorflow/tensorflow.git
git fetch google_upstream
# git checkout develop-upstream-deven-rocsolver
git checkout google_upstream_rocm_updates_for_jax_201210
git rebase google_upstream/master

cd $HOME/jax
/common/scripts/misc/git_config.sh
git remote add inailuig https://github.com/inailuig/jax
git fetch inailuig
git checkout inailuig/rocm-gpukernels
