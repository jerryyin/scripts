cd /
sudo su
lsblk -l
mkfs -t ext4 /dev/nvme01n1
mkdir /data
mount /dev/nvme01n1 /data
