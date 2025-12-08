# apt-get update && apt-get install -y  python3-numpy python3-dev python3-pip python3-wheel

# pip3 install wheel==0.31.1
# pip3 install virtualenv
# pip3 install --upgrade six==1.10.0
# pip3 install --upgrade absl-py
# pip3 install --upgrade werkzeug==0.11.10
# pip3 install --upgrade bleach==2.0.0
# pip3 install --upgrade markdown==2.6.8
# pip3 install --upgrade protobuf==3.6.0
# pip3 install --upgrade numpy==1.14.5
# pip3 install scipy==1.1.0
# pip3 install scikit-learn==0.18.1
# pip3 install pandas==0.19.2
# pip3 install psutil
# pip3 install py-cpuinfo
# pip3 install pylint==1.6.4
# pip3 install pep8
# pip3 install portpicker
# pip3 install grpcio
# pip3 install --upgrade astor
# pip3 install --upgrade gast
# pip3 install --upgrade termcolor
# pip3 install --upgrade setuptools==39.1.0
# pip3 install keras_applications==1.0.6 --no-deps
# pip3 install keras_preprocessing==1.0.5 --no-deps
# pip3 install --upgrade h5py==2.8.0
# pip3 install tensorflow_estimator --no-deps

# git config --global user.email "deven.desai.amd@gmail.com"
# git config --global user.name "Deven Desai"


HOME=/root

# #################################################################################

# apt-get update && apt-get install -y wget unzip libssl-dev libboost-dev libboost-system-dev libboost-filesystem-dev

# #################################################################################

# rm -rf $HOME/hcc/
# cd $HOME && git clone --recursive https://github.com/RadeonOpenCompute/hcc.git
# cd $HOME/hcc && git checkout -b issue812-fix d530be7
# cd $HOME/hcc && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make package -j$(nproc) && dpkg -i *.deb

# #################################################################################

#cd $HOME && git clone -b roc-1.8.x-pr457 https://github.com/parallelo/HIP.git 
# cd $HOME && git clone -b roc-1.9.x-module-hash-fix https://github.com/ROCm-Developer-Tools/HIP.git 

# rm -rf $HOME/HIP
# cd $HOME && git clone https://github.com/ROCm-Developer-Tools/HIP.git 
# cd $HOME/HIP && rm -rf build && mkdir build && cd build && cmake .. && make package -j$(nproc) && dpkg -i *.deb
## ROCm 2.7 and onwards
# cd $HOME/HIP && rm -rf build && mkdir build && cd build && cmake .. && make -j$(nproc) && make hiprtc && make package && dpkg -i *.deb
# cd $HOME/HIP/build && make package -j$(nproc) && dpkg -i *.deb

# #################################################################################

# rm -rf $HOME/rocm-cmake
# cd $HOME && git clone https://github.com/RadeonOpenCompute/rocm-cmake.git 
# cd $HOME/rocm-cmake && mkdir build && cd build && cmake .. && make package -j$(nproc) && dpkg -i ./rocm-cmake*.deb

# #################################################################################

# rm -rf $HOME/MIOpenGEMM
# cd $HOME && git clone https://github.com/ROCmSoftwarePlatform/MIOpenGEMM.git
# cd $HOME/MIOpenGEMM &&  mkdir build && cd build && cmake .. && make package -j$(nproc) && dpkg -i ./miopengemm*.deb

# #################################################################################

# rm -rf $HOME/half
# cd $HOME &&  mkdir half && cd half && wget https://downloads.sourceforge.net/project/half/half/1.12.0/half-1.12.0.zip && unzip *.zip

# #################################################################################

# rm -rf $HOME/rocRAND
# cd $HOME && git clone https://github.com/ROCmSoftwarePlatform/rocRAND.git
# cd $HOME/rocRAND && rm -rf build && mkdir build
# cd $HOME/rocRAND && cd build && CXX=/opt/rocm/bin/hcc cmake -DBUILD_BENCHMARK=ON .. && make package -j$(nproc) && dpkg -i *.deb

# rm -rf $HOME/rocFFT
# cd $HOME && git clone https://github.com/ROCmSoftwarePlatform/rocFFT.git
# cd $HOME/rocFFT && ./install.sh -i

# rm -rf $HOME/rocBLAS
# cd $HOME && git clone https://github.com/ROCmSoftwarePlatform/rocBLAS.git
# cd $HOME/rocBLAS && ./install.sh -i
    
# #################################################################################

# rm -rf $HOME/miopen
# # #manual# cd $HOME && git clone -b develop https://github.com/AMDComputeLibraries/MLOpen.git miopen
# cd $HOME && git clone -b master https://github.com/ROCmSoftwarePlatform/MIOpen.git miopen

# cd $HOME/miopen && cmake -P install_deps.cmake
#
# cd $HOME/miopen && cd build && \
cd $HOME/miopen && rm -rf build && mkdir build && cd build && \
    CXX=/opt/rocm/bin/hcc cmake \
       -DMIOPEN_BACKEND=HIP \
       -DCMAKE_PREFIX_PATH="/opt/rocm/hcc;/opt/rocm/hip;/opt/rocm/bin/clang-ocl" \
       -DCMAKE_BUILD_TYPE=Release \
       ..  && \
    make package -j$(nproc) && dpkg -i ./MIOpen*.deb


#        -DCMAKE_CXX_FLAGS="-isystem /usr/include/x86_64-linux-gnu/" \
#        -DHALF_INCLUDE_DIR=$HOME/half/include \
#        -DMIOPEN_USE_ROCBLAS=false \

# #################################################################################

# rm -rf $HOME/pkgs/rocblas
# mkdir -p $HOME/pkgs/rocblas
# cd $HOME/pkgs/rocblas && wget https://github.com/ROCmSoftwarePlatform/rocBLAS/releases/download/v14.3.0/rocblas-0.14.3.158-Linux.deb && dpkg -i *.deb

# rm -rf $HOME/pkgs/MIOpen
# mkdir -p $HOME/pkgs/MIOpen
# cd $HOME/pkgs/MIOpen && wget https://www.dropbox.com/s/ppk5cqeylfsgnya/MIOpen-HIP-1.6.0-fa431f0-Linux.deb && dpkg -i *.deb

# #################################################################################

# rm -rf $HOME/rccl
# cd $HOME && git clone https://github.com/ROCmSoftwarePlatform/rccl.git
# cd $HOME/rccl && mkdir build && cd build && CXX=/opt/rocm/bin/hcc cmake .. && make package -j$(nproc) && dpkg -i *.deb

# #################################################################################

# version=0.15.0
# cd $HOME/
# wget https://github.com/bazelbuild/bazel/releases/download/$version/bazel-$version-installer-linux-x86_64.sh
# chmod a+x bazel-$version-installer-linux-x86_64.sh
# ./bazel-$version-installer-linux-x86_64.sh
