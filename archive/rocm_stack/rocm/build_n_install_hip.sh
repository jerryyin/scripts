rm -rf build
mkdir build && cd build && cmake .. && make package -j$(nproc) && dpkg -i *.deb
