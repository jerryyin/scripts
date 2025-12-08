SRC_REPO=https://github.com/tensorflow/mlir
MIRROR_NAME=mlir-mirror
MIRROR_REPO

git clone --mirror $SRC_REPO $MIRROR_NAME
cd $MIRROR_NAME
git push --mirror $MIRROR_REPO


# https://help.github.com/en/articles/duplicating-a-repository
# https://stackoverflow.com/questions/3959924/whats-the-difference-between-git-clone-mirror-and-git-clone-bare
