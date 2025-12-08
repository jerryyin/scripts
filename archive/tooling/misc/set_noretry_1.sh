#sudo sh -c 'echo "options amdkfd noretry=0"        >> /etc/modprobe.d/amdkfd.conf'
sudo sh -c 'echo "options amdgpu vm_update_mode=0" >> /etc/modprobe.d/amdgpu.conf'
sudo update-initramfs -u
sudo reboot
