set -e

apt-get update --allow-insecure-repositories

apt-get install -y --allow-unauthenticated \
	rccl \
	rocm-dev \
	rocm-libs \
	rocm-utils

apt-get install -y --allow-unauthenticated \
	cmake \
	kmod \
	libnuma-dev \
	pciutils

apt-get clean
rm -rf /var/lib/apt/lists/*
