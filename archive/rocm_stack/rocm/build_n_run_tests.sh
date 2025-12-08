rm -rf build
mkdir build
cd build
cmake ..
make -j$(nproc)
make install
make build_tests && make test
