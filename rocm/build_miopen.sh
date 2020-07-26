sudo apt-get update && sudo apt-get install sqlite3 libsqlite3-dev -y

cd /usr/local
wget https://www.dropbox.com/s/kw7caurj82zozsd/half.hpp

export ROCM_PATH=/opt/rocm-3.3.0

cd /root/MLOpen

cmake -P install_deps.cmake --minimum --prefix /usr/local

mkdir -p build && cd build

CXX=$ROCM_PATH/hcc/bin/hcc cmake \
   -DMIOPEN_BACKEND=HIP \
   -DCMAKE_PREFIX_PATH="$ROCM_PATH/hcc;$ROCM_PATH/hip;/usr/local" \
   -DCMAKE_CXX_FLAGS="-isystem /usr/include/x86_64-linux-gnu/" \
   -DMIOPEN_MAKE_BOOST_PUBLIC=ON \
   -DBoost_USE_STATIC_LIBS=Off  \
   -DCMAKE_INSTALL_PREFIX=$ROCM_PATH/miopen ..

make package -j$(nproc)

# dpkg -i *.deb
