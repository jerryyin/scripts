#install emacs
apt-get update && apt-get install -y emacs25-nox
cd /root && rm -rf .emacs.d && git clone https://github.com/deven-amd/.emacs.d.git

# download ROCm TF source
cd /root && git clone https://github.com/ROCmSoftwarePlatform/tensorflow-upstream tensorflow
cd /root/tensorflow && git remote add google_upstream https://github.com/tensorflow/tensorflow.git
cd /root/tensorflow && git fetch google_upstream
cd /root/tensorflow && /common/scripts/misc/git_config.sh

# download the benchamrks repo
cd /root && git clone https://github.com/tensorflow/benchmarks

# # download the models repo
# cd /root && git clone https://github.com/tensorflow/models

# # download the BERT repo
# cd /root && git clone https://github.com/ROCmSoftwarePlatform/BERT

# check/uninstall rocprim
# apt --installed list | grep rocprim
# apt autoremove -y rocprim
