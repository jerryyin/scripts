set -e

WORKDIR=$1
VERSION=${2:-"3.1.0"}

cd $WORKDIR

curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$VERSION/bazel-$VERSION-installer-linux-x86_64.sh
chmod +x ./bazel-$VERSION-installer-linux-x86_64.sh
./bazel-$VERSION-installer-linux-x86_64.sh

