#!/usr/bin/env bash
set -ex

# assumption is that this script is executed from  the root of the JAX repo 

git fetch upstream

git checkout master
git merge --ff-only upstream/master
git push origin master
