set -eux

# cd /root/ && git clone https://github.com/tensorflow/model-optimization.git && cd model-optimization && python3 setup.py install
pip3 install --upgrade tensorflow-model-optimization
cd /root/
# git clone https://streamhsa:4bacc961f3924402aeaf8f6df2776e8b8ec8a98f@github.com/ROCmSoftwarePlatform/MLPerf-mGPU/
git clone https://github.com/ROCmSoftwarePlatform/MLPerf-mGPU/
cd /root/MLPerf-mGPU/
# git checkout fixes_for_tf_27
# git checkout mlperf-0.7-BU
git checkout deven/mlperf-0.7-BU-eval-XLA
