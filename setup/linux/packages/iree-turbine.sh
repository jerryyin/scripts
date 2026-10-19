#!/bin/bash
set -e

cd ~/iree/third_party/tracy
cmake -B csvexport/build -S csvexport -DCMAKE_BUILD_TYPE=Release && \
  cmake --build csvexport/build --parallel --config Release

cd ~
# Clone only when absent. Never rm -rf here: ~/iree-turbine can hold un-pushed
# work. A dir without .git is an anomaly (failed clone or corruption) — surface it
# for manual cleanup instead of silently destroying a possible repo.
if [ -d ~/iree-turbine/.git ]; then
    :  # already present
elif [ ! -e ~/iree-turbine ]; then
    git clone git@github.com:iree-org/iree-turbine.git
else
    echo "⚠️  ~/iree-turbine exists but has no .git; refusing to touch it. Inspect and remove manually, then re-run." >&2
    exit 1
fi
pip install -r ~/iree-turbine/pytorch-rocm-requirements.txt
pip install -e ~/iree-turbine && pip uninstall -y iree-base-compiler iree-base-runtime

# Debug for boo driver:
#TURBINE_DEBUG="log_level=DEBUG"
