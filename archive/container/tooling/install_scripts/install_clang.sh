set -e

WORKDIR=$1
VERSION=${2:-10}

cd $WORKDIR

wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
./llvm.sh $VERSION

update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-${VERSION} ${VERSION}
update-alternatives --install /usr/bin/clang clang /usr/bin/clang-${VERSION} ${VERSION}
