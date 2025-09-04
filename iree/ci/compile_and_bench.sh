#!/bin/bash
set -euo pipefail

TEST_SUITE_REPO="iree-test-suites"
# From pkgci_test_sharktank.yml
TEST_SUITE_COMMIT="615c14ea2dd082d132cd64cd92806bcc7fdb5c75"

echo '=== Step: Setup environment ==='
export ROCM_CHIP=gfx942
export SKU=mi300
export BACKEND=rocm
# Set the test id from collect-only run
export TEST=clip_rocm

# Only clone & install if not already present
if [ ! -d "$TEST_SUITE_REPO" ]; then
  echo ">>> Cloning $TEST_SUITE_REPO at $TEST_SUITE_COMMIT"
  git clone git@github.com:iree-org/iree-test-suites.git "$TEST_SUITE_REPO"
  pushd "$TEST_SUITE_REPO" >/dev/null
  git checkout "$TEST_SUITE_COMMIT"
  popd >/dev/null

  echo ">>> Installing editable package"
  pip install -e "$TEST_SUITE_REPO/sharktank_models"
else
  echo ">>> Using existing $TEST_SUITE_REPO checkout"
fi

# To compile all tests
#pytest \
#  iree-test-suites/sharktank_models/quality_tests \
#  -rpFe \
#  --log-cli-level=info \
#  --durations=0 \
#  --timeout=1200 \
#  --capture=no \
#  --test-file-directory=/home/runner/_work/iree/iree/tests/external/iree-test-suites/sharktank_models/quality_tests \
#  --external-file-directory=/home/runner/_work/iree/iree/tests/external/iree-test-suites/test_suite_files

# To gather test ids: 
# 8b_f16_decode_rocm, 8b_f16_prefill_rocm, clip_rocm, mmdit_rocm, vae_rocm, punet_int8_fp16_rocm, punet_int8_fp8_rocm,
# scheduler_rocm, unet_fp16_960_1024_rocm, unet_fp16_rocm
#pytest \
#  --collect-only -v \
#  iree-test-suites/sharktank_models/quality_tests \
#  --test-file-directory=tests/external/iree-test-suites/sharktank_models/quality_tests \
#  --external-file-directory=tests/external/iree-test-suites/test_suite_files


# Compile a single model
echo '=== Step: Compile model ==='
pytest \
  "$TEST_SUITE_REPO/sharktank_models/quality_tests/model_quality_run.py" \
  -k $TEST \
  -s -v \
  --test-file-directory="$TEST_SUITE_REPO/sharktank_models/quality_tests" \
  --external-file-directory="$TEST_SUITE_REPO/test_suite_files"

# Run benchmark tests
echo '=== Step: Run benchmark tests ==='
pytest \
  "$TEST_SUITE_REPO/sharktank_models/benchmarks" \
  -k $TEST \
  --log-cli-level=info \
  --retries=7 \
  --timeout=600 \
  --test-file-directory="$TEST_SUITE_REPO/sharktank_models/benchmarks" \
  --external-file-directory="$TEST_SUITE_REPO/test_suite_files"

