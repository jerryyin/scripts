#!/usr/bin/env bash
# MI450 (heliosr B0) post-boot bring-up.
#
# Order matters: CSC_Feature_enable must run before the amdgpu driver loads,
# and every tuning script below assumes the driver is already up.
# Each step is reported but non-fatal, so one flaky download does not abort
# the whole bring-up and leave the box half-configured.
set -uo pipefail

# --post-driver-only skips CSC_Feature_enable and the modprobe, for the case
# where the host rebooted itself (Power Restore Policy is always-on) and the
# driver is already up and healthy. CSC cannot be applied this way -- it needs
# the driver down -- so results gathered afterwards carry that caveat.
POST_DRIVER_ONLY=0
[[ "${1:-}" == "--post-driver-only" ]] && POST_DRIVER_ONLY=1

STEP=0
run_step() {
  local desc="$1"; shift
  STEP=$((STEP + 1))
  printf '\n[step %02d] %s\n' "$STEP" "$desc"
  if "$@"; then
    printf '[step %02d] OK\n' "$STEP"
  else
    printf '[step %02d] FAILED (rc=%d) -- continuing\n' "$STEP" "$?"
    echo "$desc" >> /tmp/heliosr-postboot-failures.txt
  fi
}

# Fetch a tuning script, keep a copy, and run it. On a fetch failure, run the kept copy.
#
# Twelve of the steps below live on dcgpuval-storage.amd.com and were piped straight into
# python3, so tuning worked only while that host was serving. On 2026-08-15 it was serving
# at 10:53 and refusing connections by 12:25, which turned a routine AC cycle into a board
# that could not be tuned at all: every fetched step failed with curl's exit 7 while the
# purely local steps passed. A boot we own must be tunable from what we already have.
#
# The cache fills as a side effect of any successful boot, so it needs no separate
# maintenance and cannot drift from what the steps actually run. A cached run is announced,
# because a row measured on cache-tuned hardware should be traceable to that fact.
TUNING_CACHE="${TUNING_CACHE:-/opt/mi45x-tuning-cache}"

curl_py() {
  local url="$1" name tmp
  name="${url##*/}"
  tmp="$(mktemp "/tmp/${name}.XXXXXX")"

  if curl -fsSL --retry 3 --retry-delay 5 --max-time 120 "$url" -o "$tmp" && [[ -s "$tmp" ]]; then
    sudo mkdir -p "$TUNING_CACHE" && sudo cp "$tmp" "$TUNING_CACHE/$name"
    sudo python3 "$tmp"
    local rc=$?
    rm -f "$tmp"
    return $rc
  fi
  rm -f "$tmp"

  if [[ -r "$TUNING_CACHE/$name" ]]; then
    echo "    [cache] $name -- fetch failed, running $TUNING_CACHE/$name"
    sudo python3 "$TUNING_CACHE/$name"
    return $?
  fi

  echo "    [MISSING] $name -- fetch failed and no cached copy exists" >&2
  return 7
}

: > /tmp/heliosr-postboot-failures.txt

echo "=== pre-flight ==="
echo "uptime: $(uptime)"
echo "amdgpu loaded: $(lsmod | grep -c '^amdgpu ')"

# --- driver must not have been loaded yet ------------------------------------
# CSC_Feature_enable has to run before amdgpu initializes. Reloading the driver
# is NOT a valid way to get back to that state: on this board an unload/reload
# fails device discovery ("SMN base address query not supported", "discovery
# failed: -2") and leaves the module loaded with no GPU node behind it, which
# silently breaks every later SMU message. Only a power cycle recovers it, so
# refuse to continue rather than tuning a dead GPU.
if (( POST_DRIVER_ONLY )); then
  echo "[info] --post-driver-only: skipping CSC_Feature_enable and modprobe"
  echo "[CAVEAT] CSC_Feature_enable was NOT applied on this boot; it requires the"
  echo "         driver to be down. Performance data gathered now is not directly"
  echo "         comparable to a full cold bring-up."
  if ! lsmod | grep -q '^amdgpu '; then
    echo "[FATAL] --post-driver-only given but amdgpu is not loaded"
    exit 2
  fi
elif lsmod | grep -q '^amdgpu '; then
  echo "[FATAL] amdgpu is already loaded. Do not reload it -- discovery will fail."
  echo "        Power cycle and run this script before anything loads the driver,"
  echo "        or pass --post-driver-only to tune the running driver in place."
  exit 2
else
  run_step "CSC_Feature_enable (pre-driver)" \
    curl_py http://dcgpuval-storage.amd.com/users/jelui/mi45x_scripts/CSC_Feature_enable.py

  run_step "modprobe amdgpu (RW mode)" \
    sudo modprobe amdgpu gpu_recovery=0 halt_if_hws_hang=1 mtype_local=0 noretry=1
fi

# lsmod only proves the module inserted; the probe can still have failed. A real
# GPU node is the only trustworthy signal.
echo "[info] amdgpu loaded: $(lsmod | grep -c '^amdgpu ')"
echo "[info] gpu_recovery=$(cat /sys/module/amdgpu/parameters/gpu_recovery 2>/dev/null)"
if sudo dmesg | tail -60 | grep -q 'probe with driver amdgpu failed'; then
  echo "[FATAL] amdgpu probe failed -- module is loaded but there is no GPU node."
  sudo dmesg | grep -E 'discovery failed|Fatal error during GPU init|probe with driver amdgpu failed' | tail -3
  exit 3
fi
if [[ ! -e /sys/class/kfd/kfd/topology/nodes/1 ]] && [[ -z "$(ls /sys/class/drm/card* 2>/dev/null)" ]]; then
  echo "[FATAL] no DRM/KFD GPU node present after modprobe"
  exit 3
fi
echo "[info] GPU node present: OK"

run_step "disable NUMA balancing" \
  sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'

cd /opt/amd-apps/ || echo "[warn] /opt/amd-apps missing"

# --- post-driver tuning ------------------------------------------------------
run_step "set_dram_settings (UMC, disable all-bank refresh)" \
  curl_py http://dcgpuval-storage.amd.com/users/jelui/mi45x_scripts/set_dram_settings.py
# KNOWN TO FAIL PERMANENTLY on this board: its SMU firmware rejects the TDC write. The
# step is kept because the failure is informative and because a board with different SMU
# firmware would take it; it is expected in the failure list, not a sign of a bad boot.
run_step "set_tdc_limits_mi450 (HBM rail TDC) -- known to fail on this board's SMU firmware" \
  curl_py http://dcgpuval-storage.amd.com/users/kstraube/set_tdc_limits_mi450.py
run_step "disable_gcea_link_mgr (HBM unloaded latency)" \
  curl_py http://dcgpuval-storage.amd.com/users/jelui/mi45x_scripts/disable_gcea_link_mgr.py
run_step "set_cp_hpd_enable_offload_check (kernel launch latency)" \
  curl_py http://dcgpuval-storage.amd.com/users/jelui/mi45x_scripts/set_cp_hpd_enable_offload_check.py
run_step "MI450_disTxIdle (kernel launch latency)" \
  curl_py http://dcgpuval-storage.amd.com/users/muku/MI450x_ScaleUp_PerfScripts/MI450_disTxIdle_may13.py
run_step "kll_optimization_mi450 (chicken bits)" \
  curl_py http://dcgpuval-storage.amd.com/users/harnsing_pharaoh/kll_optimization/kll_optimization_mi450.py
run_step "MI450_DF_DisSDPdis (50us issue)" \
  curl_py http://dcgpuval-storage.amd.com/users/tifyeung/debug_scripts/MI450_DF_DisSDPdis.py
run_step "df_mi450_sec_lvl (stream write / OAI BW GEMMs)" \
  curl_py http://dcgpuval-storage.amd.com/users/abalam/MI45X/scripts/df_mi450_sec_lvl.py
run_step "disable_mgcg_override (enable GFX MGCG)" \
  curl_py http://dcgpuval-storage.amd.com/users/jelui/mi45x_scripts/disable_mgcg_override.py
run_step "wr6_enable_deepsleeps (VCN/DMA deep sleep, SMU MGCG)" \
  curl_py http://dcgpuval-storage.amd.com/users/jelui/mi45x_scripts/wr6_enable_deepsleeps.py
run_step "wr6_disable_socpcc (disable SOC PCC)" \
  curl_py http://dcgpuval-storage.amd.com/users/jelui/mi45x_scripts/wr6_disable_socpcc.py
run_step "CSC post-driver-load parameters" \
  curl_py http://dcgpuval-storage.amd.com/users/jelui/mi45x_scripts/CSC_after_driverload_allclk_perf_mode_default.py

run_step "agt smcmsgsend 0x65,0x32" \
  sudo /opt/amd-apps/agt_internal/agt_internal -i=0 -smcmsgsend=0x65,0x32
run_step "agt smcmsgsend 0x3D,0x1004B,0x0" \
  sudo /opt/amd-apps/agt_internal/agt_internal -i=0 -smcmsgsend=0x3D,0x1004B,0x0

echo
echo "=== bring-up summary ==="
if [[ -s /tmp/heliosr-postboot-failures.txt ]]; then
  echo "FAILED STEPS:"
  cat /tmp/heliosr-postboot-failures.txt
else
  echo "all steps OK"
  # Record which boot this tuning belongs to. The health check used to infer
  # that from numa_balancing being 0, but the other users on this box run their
  # own tuning and set it to 0 exactly as we do, so a boot whose modprobe race
  # we lost still read as ours -- the tuning was never applied, the agent was
  # correctly refusing to measure, and the check said everything was fine. A
  # marker naming the boot cannot be confused that way, and cannot go stale
  # either, since the id changes on every boot.
  cat /proc/sys/kernel/random/boot_id > /tmp/heliosr-tuned-boot
  # Which path tuned this boot decides whether its rows may be compared across
  # boots, and it cannot be reconstructed reliably afterwards: both paths end
  # with the same marker and the same module parameters. Record it now, so the
  # provenance row states it rather than inferring it.
  if (( POST_DRIVER_ONLY )); then
    echo "post-driver-only" > /tmp/heliosr-tuning-path
  else
    echo "cold" > /tmp/heliosr-tuning-path
  fi
fi
echo "amdgpu loaded: $(lsmod | grep -c '^amdgpu ')"
/opt/rocm/bin/rocm-smi --showid 2>&1 | head -25 || true
