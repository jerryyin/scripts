set -e

WORKDIR=$1
VERSION=${2:-8}

add-apt-repository -y ppa:ubuntu-toolchain-r/test 
apt-get update
apt-get install -y gcc-${VERSION} g++-${VERSION}

update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-${VERSION} ${VERSION}
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${VERSION} ${VERSION}
