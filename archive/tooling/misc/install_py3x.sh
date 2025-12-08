set -eux

yes | add-apt-repository ppa:deadsnakes/ppa
apt-get update

PY_3X=python3.8

apt-get install -y --upgrade $PY_3X
apt-get install -y $PY_3X-dev
apt-get install -y $PY_3X-distutils
apt-get install -y python3-pip 

ln -sf /usr/bin/$PY_3X /usr/local/bin/$PY_3X
update-alternatives --install /usr/bin/python3 python3 /usr/bin/$PY_3X 1
update-alternatives --set python3 /usr/bin/$PY_3X

pip3 install --upgrade pip
pip3 install --upgrade setuptools
