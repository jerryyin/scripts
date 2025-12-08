SUDO=
SUDO=sudo

$SUDO apt-get install -y --reinstall build-essential
$SUDO apt-get install -y software-properties-common
$SUDO add-apt-repository -y ppa:ubuntu-toolchain-r/test
$SUDO apt-get update
$SUDO apt-get install -y gcc-8 g++-8
$SUDO update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 60
$SUDO update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 60
