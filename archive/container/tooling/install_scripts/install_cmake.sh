set -e

WORKDIR=$1
VERSION=${2:-"3.17.3"}

cd $WORKDIR

apt-get purge -y cmake

wget https://github.com/Kitware/CMake/releases/download/v$VERSION/cmake-$VERSION-Linux-x86_64.sh
chmod +x cmake-$VERSION-Linux-x86_64.sh
./cmake-$VERSION-Linux-x86_64.sh --skip-license --prefix=/usr/local
