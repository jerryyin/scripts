rm -rf build
mkdir build
cd build
CXX=/opt/rocm/hcc/bin/hcc \
   cmake \
   -DMIOPEN_BACKEND=HIP \
   -DCMAKE_PREFIX_PATH="/opt/rocm/hcc;/opt/rocm/hip" \
   -DCMAKE_CXX_FLAGS="-isystem /usr/include/x86_64-linux-gnu/" \
   -DMIOPEN_MAKE_BOOST_PUBLIC=on \
   ..
#make VERBOSE=1 install -j$(nproc)
make install -j$(nproc)
