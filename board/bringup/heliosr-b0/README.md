# board/bringup/heliosr-b0/ — post-boot tuning for ONE board

```
heliosr-b0/
├── README.md       # this file
└── post-boot.sh    # fourteen-step post-boot bring-up, run after every power cycle
```

`post-boot.sh` brings an MI450 (heliosr B0) board up to a tuned, measurable state after a
power cycle: it applies a pre-driver CSC feature enable, loads `amdgpu` with the intended
parameters, verifies a real GPU node actually appeared, then applies eleven tuning scripts
and two raw SMU messages.

```bash
./post-boot.sh                     # full cold bring-up, run BEFORE anything loads amdgpu
./post-boot.sh --post-driver-only  # driver already up; skips CSC (see the caveat it prints)
```

Each step is reported but non-fatal, so one flaky download does not abort the bring-up and
leave the box half-configured. Failures are collected in
`/tmp/heliosr-postboot-failures.txt`, and a successful run records the boot id in
`/tmp/heliosr-tuned-boot` and the path taken in `/tmp/heliosr-tuning-path`.

---

## This is validated on ONE board at ONE silicon stepping. Read this before reusing it.

It is not a portable bring-up script and should not be treated as one:

- **Twelve of the fourteen steps `curl` Python from individual users' scratch directories**
  on an internal host, `dcgpuval-storage.amd.com` — paths like
  `/users/jelui/mi45x_scripts/`, `/users/kstraube/`, `/users/muku/`. These are personal
  directories, not a release artifact. Any of them can move or disappear without notice.
- **Two steps send raw SMU messages** via `/opt/amd-apps/agt_internal`
  (`-smcmsgsend=0x65,0x32` and `-smcmsgsend=0x3D,0x1004B,0x0`). Raw opcodes against one
  firmware revision; they are not documented interfaces and carry no compatibility promise.
- **One script is date-stamped**: `MI450_disTxIdle_may13.py`. That name is the version. A
  later one presumably exists and this is not it.
- **`set_tdc_limits_mi450` is known to fail permanently on this board** — its SMU firmware
  rejects the write. Expect it in the failure list on every single run. That is not a sign
  of a bad boot, and the step is kept because a board with different SMU firmware would
  take it.

Treat the whole thing as a record of what this board needed, not as a recipe.

## Why it caches the scripts it fetches

Because the fetch is the single point of failure and it has already failed.

On **2026-08-15** the internal host was serving at **10:53** and refusing connections by
**12:25**. That turned a routine AC power cycle into a board that could not be tuned at all:
every fetched step failed with curl's exit 7, while the purely local steps passed. A boot we
own must be tunable from what we already have.

So `curl_py` writes every successfully fetched script into `$TUNING_CACHE` (default
`/opt/mi45x-tuning-cache`) and falls back to the cached copy when the fetch fails. The cache
fills as a side effect of any successful boot, so it needs no separate maintenance and cannot
drift from what the steps actually run. A cached run is **announced** in the output, because a
measurement taken on cache-tuned hardware should be traceable to that fact.

## CSC_Feature_enable must run BEFORE the driver loads — and reloading is not a way back

`CSC_Feature_enable` has to be applied before `amdgpu` initializes. There is no second
chance within a boot, and in particular **unloading and reloading the driver does not get you
back to that state.**

On this board an unload/reload **fails device discovery**:

```
SMN base address query not supported
amdgpu: discovery failed: -2
```

and leaves the module loaded with **no GPU node behind it**, which silently breaks every
later SMU message — you get a tuned-looking box with nothing underneath. **Only a power cycle
recovers it.**

The script therefore refuses to continue if `amdgpu` is already loaded, rather than tuning a
dead GPU. If the host rebooted itself (Power Restore Policy is always-on) and the driver is
already up and healthy, `--post-driver-only` tunes it in place — but CSC is *not* applied on
that path, the script says so, and data gathered afterwards is not directly comparable to a
full cold bring-up.

Two further checks exist because `lsmod` is not enough: the module can insert while the probe
fails. The script greps `dmesg` for `probe with driver amdgpu failed` and requires a real
`/sys/class/kfd` or `/sys/class/drm/card*` node before it will tune anything.

## Why the boot marker exists

A successful run writes the boot id to `/tmp/heliosr-tuned-boot`. The health check used to
infer "this boot is tuned" from `numa_balancing` being 0 — but other users on this box run
their own tuning and set it to 0 exactly as we do. So a boot whose modprobe race we *lost*
still read as ours: the tuning was never applied, the agent was correctly refusing to
measure, and the check said everything was fine. A marker naming the boot cannot be confused
that way, and cannot go stale either, since the id changes on every boot.

`/tmp/heliosr-tuning-path` records `cold` or `post-driver-only`, because which path tuned a
boot decides whether its results may be compared across boots and cannot be reconstructed
afterwards — both paths end with the same marker and the same module parameters.

## Configuration

| Variable | Effect | Default |
| --- | --- | --- |
| `TUNING_CACHE` | where fetched scripts are cached and read back from | `/opt/mi45x-tuning-cache` |
