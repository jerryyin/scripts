set -e

WORKDIR=$1

cd $WORKDIR

# OFED_VERSION="4.4-2.0.7.0"
# OFED_VERSION="4.6-1.0.1.1"
OFED_VERSION="5.1-0.6.6.0"
PKG_NAME=MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu16.04-x86_64
# PKG_NAME=MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu18.04-x86_64

DEBIAN_FRONTEND=noninteractive
apt-get update 

# install net-tools and openssh
apt-get install -y \
	net-tools \
	openssh-client \
	openssh-server 

# download and install Mellanox OFED
wget http://content.mellanox.com/ofed/MLNX_OFED-${OFED_VERSION}/${PKG_NAME}.tgz
tar -zxf ${PKG_NAME}.tgz
cd ${PKG_NAME}
./mlnxofedinstall --user-space-only --without-fw-update --all -q --force

# uninstall some OFED packages
apt-get remove -y \
	ucx 

# Allow OpenSSH to talk to containers without asking for confirmation
cp /etc/ssh/ssh_config /etc/ssh/ssh_config.bkup
cat /etc/ssh/ssh_config | \
    grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config
mkdir -p /var/run/sshd
