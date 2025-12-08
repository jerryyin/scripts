set -eux


WORKDIR=/root/temp

mkdir -p $WORKDIR && cd $WORKDIR

# apt purge --auto-remove cmake
# apt update && apt install -y software-properties-common lsb-release && apt clean all

# wget https://apt.kitware.com/keys/kitware-archive-latest.asc
# gpg -o /etc/apt/trusted.gpg.d/kitware.gpg --dearmor kitware-archive-latest.asc 

apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"

apt update && apt install -y cmake
