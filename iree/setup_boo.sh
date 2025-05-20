#!/bin/bash
set -e

cd ~/iree/third_party/tracy
cmake -B csvexport/build -S csvexport -DCMAKE_BUILD_TYPE=Release && \
  cmake --build csvexport/build --parallel --config Release

cd ~
git clone git@github.com:iree-org/iree-turbine.git
pip install -e ~/iree-turbine && pip uninstall -y iree-base-compiler iree-base-runtime

# Debug for boo driver:
#TURBINE_DEBUG="log_level=DEBUG"
