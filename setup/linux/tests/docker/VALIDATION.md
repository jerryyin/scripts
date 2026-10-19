# Validation results — 2026-10-19

## Full migration

The repository now has one canonical location for each implementation. Removed:

- 43 forwarding entry points, including the old Docker setup entry point.
- The old Docker environment, workspace, library, and host directories.
- Six forwarding profiles and the migration manifest.
- Layout fallbacks in the Dockerfile, Compose startup, and remote provisioning.

Docker and Kubernetes select setup profiles explicitly. Service names stay the
same; CK and rocMLIR select `base`, and all three Triton services select `triton`.
New structure tests enforce the directory boundaries and declarative profiles.

Both `scripts` and `rc_files` are tested from local working-tree snapshots.
Default builds clone GitHub; publish `scripts` before the matching `rc_files`
change. Existing development images need rebuilding for the migrated startup command.

## Shallow subject layout — final checks

`setup/` owns Linux and Windows provisioning. Docker utilities and Kubernetes
workflows live under `containers/`. Frequently used `am/`, `board/`, `profiling/`,
and `gdb/` directories remain at the root, alongside project directories,
`bench/`, and `numerics/`. Agent utilities live under `agent/`. There is no
`tools/` bucket or compatibility directory.

These checks include the latest upstream Claude settings changes. The credential
integration fixture now calls `setup/linux/credentials/activate.sh`. Additional
regressions check IREE's numerical helpers and the shell's AM plugin path.

The first fresh image build stopped during vim-plug installation with an
intermittent missing dictionary key (`context.vim`). A separate fresh Vim
installation passed. Unattended plugin installation now uses one job at a time
and reports the throw location on failure. A real Vim driver regression covers
install/update and rejects incomplete checkouts; a fresh sequential plugin install
also passed, including LeaderF's native extension. A new full image was then
built and passed the six offline runtime groups below.

| Check | Result |
| --- | --- |
| Linux setup, profiles, structure, vault, and caller tests | 39 passed |
| Kubernetes catalog/launcher tests | 5 passed |
| `rc_files/.setup` tests | 56 passed |
| tmux setup tests | 8 passed |
| Vim setup tests | 9 passed, 1 Windows-only test skipped |
| Claude settings and credential integration tests | 15 passed; installed CLI checked separately |
| GDB configuration test | 1 passed |
| Profiling and decoder discovery | 11 passed, including 8 decoder subcases |
| AM wrapper fixtures in Docker | 4 passed; decoder test also passed there |
| Docker service matrix | 18 passed |
| Fresh Ubuntu installation through actual `dbuild base` | Passed |
| Offline installed-image runtime | 6 passed |
| Installed Claude CLI against a local fixture gateway | Passed; all 16 Claude tests passed inside the image |
| Board, bind-mount, and unattended-verdict self-checks | 3 passed |
| Moved callers and utility entry points | 11 passed |
| Bash/Python/Zsh syntax, warning-level targeted ShellCheck, Compose rendering, local documentation links, and diff checks | Passed |

The AM tests use synthetic environment/version files and do not run a simulator.
All 86 detected renames preserve Git file permissions, including the moved Windows
ignore file. Runtime credentials are disposable fixture data.

Final logs and source snapshots are under `/tmp/scripts-subject-layout/`:
`final-*-tests.log`, `final-contracts/summary.json`,
`verified-full-install/dbuild-base.log`, `verified-full-install/offline-runtime.log`,
`am-container-tests.log`, `final-entrypoint-checks.log`, and
`final-static-checks.json`, and `verified-full-install/installed-claude-tests.log`.
The initial failed build is retained in
`final-full-install/dbuild-base.log`; standalone Vim logs are in `vim-sequential/`.

## Setup rename and utility moves — earlier checks

Environment setup moved from `devenv/` to `setup/`. Docker utilities now live in
`tools/docker/`, and these Docker tests live with Linux setup. Kubernetes,
profiling, simulator, and board utilities also moved under `tools/`. Claude,
Cursor, and unattended-run helpers moved under `agent/`. Both `rc_files` Docker
entry points, remote provisioning, and active callers use the new paths.

After these moves, 34 Linux setup tests, five Kubernetes tests, ten profiling
tests, and a decoder-discovery test covering eight relocation/override cases
passed. Four simulator-wrapper tests passed in a disposable container with
synthetic package environment/version files; no simulator ran. The same decoder
test also passed there. All 18 Docker integration checks passed, exercising the
actual `dbuild` function and startup/restart for all seven development services
with external installers replaced by recorders.

Board contention, bind-mount validation, and unattended-verdict self-checks passed.
Eleven additional checks covered moved callers and utility entry points using
disposable inputs. Shell/Python syntax, targeted ShellCheck, Compose configuration,
local documentation links, and diff checks passed. All moved files and their
permissions were preserved. Full package installation was not repeated for these
moves; native Windows, live Kubernetes, and GPU execution remain outside this run.

Logs and validation drivers are under `/tmp/scripts-setup-rename/`, including
`contracts/summary.json`, `linux-tests.log`, `kubernetes-tests.log`,
`analysis-tools-tests.log`, `am-container-tests.log`, and `entrypoint-summary.json`.

## Combined workflows and packages — earlier checks

The `projects/` directory is removed. Triton and IREE each keep their plan,
dependency installer, and workspace script together under `profiles/`. All
standalone install/build recipes are flat files under `packages/`. The `base`
choice is built into `setup.sh`, with no separate profile file. The earlier
14 profiles for individual installers are also removed.

After these moves, 33 Linux tests, five Kubernetes tests, and all 18 Docker
integration checks passed. The Docker suite exercised the actual `dbuild`
function and startup/restart for all seven development services, with external
installers replaced by recorders. Shell/Python syntax checks, Compose configuration,
and diff checks passed. Full package installation was not repeated for this move.

Logs from that move: `/tmp/scripts-combined-layout/contracts/`,
`/tmp/scripts-combined-layout-linux.log`, and
`/tmp/scripts-combined-layout-kubernetes.log`.

## Earlier full migration validation run

Environment: Ubuntu 24.04, x86_64; Docker Engine 29.9.1, Compose 5.6.0,
Buildx 0.38.0. Docker commands used `sudo`; user group membership was unchanged.

| Check | Result |
| --- | --- |
| Linux setup and structure tests | 33 passed |
| Kubernetes catalog/launcher tests | 5 passed |
| `rc_files/.setup` tests | 56 passed |
| tmux setup tests | 8 passed |
| Vim setup tests | 8 passed, 1 Windows-only test skipped |
| Docker integration suite | 18 passed |
| Fresh Ubuntu installation through actual `dbuild base` | Passed |
| Offline runtime checks on that image | 6 passed |
| Shell syntax, Python parsing, Compose configuration, both repositories' diff checks | Passed |

Total: **110 unit tests passed, one skipped; 18 Docker integration checks and six
full-install runtime checks passed.**

### Docker service matrix

The integration suite invokes the actual `rc_files` `dbuild` function through
its deployed Compose-file symlink for all seven development services:
`base`, `ck`, `iree`, `rocmlir`, `triton`, `triton-b0`, and `triton-mi450`.
Each uses the actual Dockerfile and Compose startup command. External installers
are replaced with recorders in this matrix.

Checks cover profile routing, build-time credential/storage isolation, startup,
restart, `SERVICE_NAME` overrides, `SETUP_PROFILE` precedence, source-reference
selection, and failure propagation. Clone failures and unknown profiles fail
real image builds. Installer failures return their nonzero status.

### Complete installation and runtime

A separate test builds a fresh Ubuntu 24.04 image through the actual `dbuild base`
function, Dockerfile, and canonical setup runner with real installers. A temporary
seed contains local Git snapshots of `scripts` and `rc_files`; the scripts clone
uses `SCRIPTS_REPOSITORY=file:///validation-sources/scripts` and
`SCRIPTS_REF=validation`. No installer is substituted in this run.

The resulting image passes six runtime check groups with networking disabled:

1. Canonical module layout with retired Docker entry points absent.
2. Installed tools, dotfile links, Vim's native LeaderF extension, tmux plugins,
   login shell, and absence of baked-in credentials.
3. Real SSH, vault, OAuth, and credential-template activation.
4. Repeated activation and credential status without configuration changes.
5. Token rotation and explicit persistent-storage opt-out.
6. Actual Compose startup activates credentials and starts Zsh.

All credentials are generated dummy data. No personal home, credential store,
or Docker socket is mounted in a test container.

## Scope and artifacts

This pass does not build full Triton/IREE dependency stacks, exercise a GPU,
authenticate to private AMD registries, execute native Windows setup, or connect
to a Kubernetes cluster. Their service routing is covered by the matrix above.
The standalone tools Dockerfile and ROCm setup were exercised in the earlier
validation pass; those earlier results are not counted as fresh builds here.

Repeatable integration and runtime commands are in [README.md](README.md).
Logs, the full-install driver, and source snapshots from the earlier full migration
are retained under `/tmp/scripts-full-migration/`. Key artifacts:

- `summary.json`
- `contracts/summary.json` and per-service build/startup logs
- `full-install/dbuild-base.log`
- `full-install/offline-runtime.log`
- `*-tests.log`
- `cleanup.log`

Validation containers, images, and recorded build-cache entries are removed after
verification. Docker remains installed and running.
