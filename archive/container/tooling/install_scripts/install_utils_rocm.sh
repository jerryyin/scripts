set -e

apt-get update --allow-insecure-repositories

apt-get install -y --no-install-recommends \


apt-get clean
rm -rf /var/lib/apt/lists/*
