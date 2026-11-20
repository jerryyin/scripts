# Running test groups on SPUR

SPUR assigns GPUs to a time-limited job. Put that assignment around a test group, benchmark command or development session. Individual GPU operations and individual tests inside that job do not each need another allocation. Submit when work is ready; there is no need to book a future start time. The request can wait in the scheduler's queue.

## Connect and inspect

The work vault's `ssh/config` owns the private host aliases and key selection. Activate that inventory through `rc_files`, then use `ssh spur` from the workstation. `spur` and `spur-005` select login node 005; `spur-006` and `spur-007` select the alternatives. Login nodes are for editing, scheduler commands and file transfers. Run builds and GPU work in allocations. Windows and WSL have separate SSH homes, keys and known-host files; configuring one does not configure the other.

Authorize only the workstation's public key on SPUR, keeping private keys local. Verify new SSH host fingerprints through a trusted source. Do not clone the vault or run its credential activation on the cluster. The connection aliases disable agent forwarding.

These read-only commands run on the login node:

```bash
sacctmgr show user | awk -v u="$USER" '$1 == u'
spur accounts show qos | grep -E '^(Name|amd-triton-qos|amd-burst-qos) '
scontrol show partition amd-spur-v2
squeue -u "$USER"
sacct -u "$USER" -S now-1day
```

Some SPUR filters are ignored: the partition query returned every partition in our checks, so read the `PartitionName=amd-spur-v2` block. Recheck current settings before choosing resource defaults. `scontrol show job <jobid>` inspects a request, and `scancel <jobid>` releases or cancels it.

## Current evidence

Live checks on 2026-11-20 confirmed account `amd-triton` with both `amd-triton-qos` and `amd-burst-qos` on SPUR 0.15.0. QoS is the scheduler's set of priority and resource rules. Both combinations retain `-A amd-triton` and successfully submitted one-GPU jobs on `amd-spur-v2`. Triton's rules report priority 10000, preemption off and a two-node group cap; burst reports priority 110, cancellation preemption, a 182-node group cap, four submissions per user and 24 hours per job. Triton has no separate QoS time limit shown; the partition maximum is three days. These are live configuration observations, not proof of every policy's enforcement.

After authorizing the workstation's enterprise public key, noninteractive SSH worked. Both allocation checks completed, each exposing one CPU and one GPU's device files. Early checks reported `/dev/kfd` permission errors, but later allocations on the same node passed the device-open checks; the reason for that change is unknown. Docker socket access was denied, and unattended sudo required a password. Later host GPU memory checks succeeded on two nodes after correcting the extra physical-index filter described below. The supplied “Getting started with SPUR” article provides the scheduler examples; its official IT AISE sources still need checking.

A two-command shell script ran in one burst allocation and deliberately exited 7; `srun` reported status 7 to its caller. The corresponding Triton attempt had waited with `QOSGrpNodeLimit` and was cancelled before starting. This demonstrates the shared cap affecting a real request and the need to keep any retry with different rules explicit. No jobs remained running or pending after the checks.

## One group, one allocation

Run from a source checkout on the cluster login node. `test-group.sh` represents an existing script with several tests, not a new required interface. The count and duration are explicit examples, subject to the account's verified limits. Use the launcher so resource checks and the tested device-filter correction happen before your command:

```bash
python3 ~/scripts/board/spur-run.py \
  --account amd-triton --qos amd-triton-qos \
  --gpus 1 --cpus 16 --time 01:00:00 -- bash ./test-group.sh
```

SPUR queues the request, assigns the resources, runs the script and releases the allocation when it ends. A noninteractive group that must survive loss of the login connection should instead use `sbatch` with a script and explicit stdout and stderr paths. A development session can use `salloc` followed by `srun --pty bash`, then run multiple commands until the allocation is released or expires. Keep outputs outside node-local scratch when they need to survive a node change.

The requested duration limits the running allocation; it is not a promised start time. Work must fit that duration, finish early, or save enough state to resume in a new job. The configured preemption modes differ as described above; interruption behavior has not been tested. CPU-only builds need scheduled CPU resources but need not hold GPUs while compiling.

## Existing tools and the changes they would need

| Owner | Observed behavior today | SPUR requirement |
| --- | --- | --- |
| `board/gpu-queue` | Finds idle GPUs and holds their free memory between queued tasks on machines without a scheduler. | SPUR owns assignment. Do not use this queue's device selection or memory holder to claim cluster GPUs. Preserve it for existing unscheduled hosts. |
| Top-level test and benchmark scripts | Run commands in an existing environment. | Run the whole script inside an allocation. Do not add scheduler calls to individual tests or worker processes. |
| `containers/docker/bin/dockrun` | Passes `--device=/dev/kfd --device=/dev/dri` and delegates to `session.py`. | A SPUR mode must expose only allocation-approved devices and add `--label spur_job_id=<jobid>`. The visible-device mapping needs live verification. |
| `containers/docker/session.py` and `docker-compose.yml` | Require the selected vault, mount it into the container and activate credentials during startup. | Cluster startup must work without the vault or private keys. Reusing an image must not implicitly activate workstation credentials. |
| `rc_files/ssh` and the work vault | Configure the SSH client and private host inventory. | Keep connection settings here; scheduler execution belongs in `scripts/board`, with container changes in their current owner. |

## Code and storage

Use home for small code/configuration files and logs, `/shared_nfs/$USER` for checkpoints and results needed across nodes, and `/mnt/m2m_nobackup/$USER` for node-local builds and scratch. Check free space before creating large files and copy durable results out of node-local storage. The supplied article says Docker users can access shared storage as root; do not place vaults, private keys or confidential data there.

For public models, prefer the managed cache: in batch shells, run `source /etc/profile.d/lmod.sh` followed by `module load hfc`. Request missing models with `hfc` and wait for them to arrive. Do not use home as a large model cache.

Transfer source with `scp`, or create a Git bundle on an approved workstation and copy it over. A bundle contains committed history only; copy any required uncommitted files separately. This avoids leaving credentials on the cluster or requiring a workstation tunnel during unattended work. GitHub EMU access from SPUR was restricted in the supplied article; its current allow-list status has not been verified. Custom job SSH servers and agent tunnels are outside this launcher.

## Chosen design

Keep resource assignment at the outer launch boundary. A future SPUR launcher in `board` should require GPU count, CPU count and duration, and use an explicit account/rules selection. It should submit one group, report the job id and output locations, and preserve useful failure information. Any automatic choice between Triton and burst rules would obscure different waiting and interruption behavior, so keep that choice explicit until the policies are known.

An existing allocation should be reused only after checking that the command really runs inside the assigned compute environment with the required resources. Merely inheriting `SPUR_JOB_ID` is insufficient evidence. Do not submit a nested allocation when a test script already runs inside the allocation. Avoid automatic resubmission until commands have a clear resume contract; repeating a partly completed command can repeat side effects.

Container support belongs in the existing container owners. It must receive the allocation's verified device set and job id and offer a startup path without vault activation. Native SPUR containers are another candidate, but image import, mount behavior and GPU visibility need a real test first. Do not copy the article's device enumeration loop into a reusable launcher until it is demonstrated that the job sees only its assigned devices.

The first implementation is `board/spur-run.py`: a Linux foreground launcher using one `srun` allocation for one command or test group. It requires account, QoS, GPU count, CPU count and duration. Before executing the command in the allocation, it checks the visible render-device count, CPU allowance and ability to open `/dev/kfd` and the render devices. A failed check prevents the test command from starting. It refuses to create an allocation when job environment variables are already present; run the command directly in an existing allocation. Existing Docker and GPU-queue launchers are unchanged. Detached batch jobs and editor sessions remain outside this implementation.

After these checks, the launcher removes `ROCR_VISIBLE_DEVICES` for the child command. In one controlled single-GPU allocation, the original value `3` made the HIP runtime report no devices; removing only that filter allowed a GPU memory round trip. The job's device isolation remains in place, and the launcher does not select devices itself. This was validated on host jobs with one GPU; multi-GPU jobs and containers remain untested. Other visibility variables are preserved.

`--dry-run` prints the allocation command without submitting it. `--node NAME` optionally requests a known compute node through SPUR, for example when scratch data already lives there. Neither option bypasses scheduling. An installed copy can be invoked as `~/.local/bin/spur-run` when the scripts checkout is absent. The source and tests live together here; update the installed copy after changing the source.

## Checks

The normal tests use a fake scheduler and need neither a GPU nor an allocation:

```bash
python3 board/test_spur_run.py -v
```

The optional live check explicitly needs a one-GPU allocation. It writes four bytes on the GPU and reads back `[42, 42, 42, 42]`; it does not validate a framework, a compiled kernel or performance:

```bash
python3 board/spur-run.py \
  --account amd-triton --qos amd-burst-qos \
  --gpus 1 --cpus 1 --time 00:01:00 -- \
  env SPUR_RUN_LIVE_GPU_TEST=1 python3 "$PWD/board/test_spur_run.py" LiveGpuTests -v
```

## Verification before adopting it

1. SSH login and enterprise public-key authorization are complete. Independently verify the host fingerprint through a trusted source; first-connection trust was used during setup.
2. Account associations, both QoS policies and the partition configuration have been read on the installed SPUR version.
3. One-GPU `hostname` and device-file checks completed under both rule sets. Host HIP memory allocation, fill and readback passed on two nodes through the launcher. No allocations remained after the checks. Framework and kernel execution need their own validation.
4. A two-command group and deliberate failure passed: both commands used one allocation, and status 7 reached the caller. One requested CPU was visible in the separate allocation probes. The launcher tests also cover argument preservation, invalid requests, nested-allocation refusal, dry runs and device-permission failure before command execution.
5. Before integrating Docker, verify only the assigned GPU devices reach the container, the job label is present, and neither a vault mount nor credential activation occurs.
6. Test expiration and cancellation with disposable commands; record how child processes, containers and logs are cleaned up. Check whether a disconnected foreground submission keeps running or ends.

## Alternatives Considered

- Adding allocation calls to every test would repeat policy, create nested requests and require changes across unrelated projects. One outer allocation covers the existing group unchanged.
- Extending `gpu-queue` to select cluster GPUs would duplicate the scheduler's assignment and could target resources outside the allocation. Keep that queue specific to unscheduled hosts; add a scheduler launch path at the same ownership level.
- Running current `dockrun` or `drun` unchanged would retain whole-device-directory exposure and vault mounts. Container launch policy must change before those commands are adopted on SPUR.
- Automatically keeping a long-running idle allocation could make interactive use convenient but would hold unused resources and still expire or be interrupted. Start with allocations scoped to actual work; make longer development sessions explicit.
