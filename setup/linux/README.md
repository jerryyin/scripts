# Linux development setup

Ubuntu provisioning shared by hosts, WSL, Docker, and Kubernetes.

```sh
bash ~/scripts/setup/linux/bootstrap.sh                  # Base development tools
bash ~/scripts/setup/linux/setup.sh triton --plan        # Inspect the full sequence
bash ~/scripts/setup/linux/setup.sh triton               # Install, activate, prepare workspace
bash ~/scripts/setup/linux/setup.sh iree --phase build   # Image preparation
bash ~/scripts/setup/linux/credentials/activate.sh       # Session credential setup
bash ~/scripts/setup/linux/setup.sh --list
```

| Location | Responsibility |
| --- | --- |
| `bootstrap.sh` | Machine prerequisites and certificate trust, then complete `rc_files` setup |
| `packages/` | Standalone install/build scripts, including one-off and historical recipes |
| `credentials/` | SSH/vault availability, delegation to rc_files credential hooks |
| `profiles/triton/`, `profiles/iree/` | Each environment's plan, dependencies, and workspace preparation |
| `remote/` | Provision remote hosts over SSH |
| `lib/` | Internal helpers for package management, shells, storage, workspaces |
| `tests/` | Local installer and integration regression checks |

`packages/` holds individual scripts such as `aiter.sh`, `llvm.sh`,
and `tokenspeed.sh`. A workflow directory under `profiles/` keeps its
`profile.sh` plan together with `deps.sh` and `workspace.sh`.
`bootstrap.sh` establishes the machine baseline, delegates configured applications, their dependencies, plugins and shell setup to the independent `rc_files` installer, and links this repository's Docker commands into `~/.local/bin`. `setup.sh` coordinates that baseline, credentials, and the selected
workflow. Its built-in `base` default needs no separate profile file.

```text
linux/
  bootstrap.sh
  setup.sh
  packages/
    aiter.sh
    tokenspeed.sh
    ...
  profiles/
    triton/
      profile.sh
      deps.sh
      workspace.sh
    iree/
      profile.sh
      deps.sh
      workspace.sh
```

For example, `setup.sh triton` runs:

```text
bootstrap.sh
credentials/activate.sh
profiles/triton/deps.sh
packages/tokenspeed.sh
profiles/triton/workspace.sh
```

## Phases and profiles

`setup.sh PROFILE --phase PHASE` supports:

| Phase | Operations |
| --- | --- |
| `all` (default) | Bootstrap, activate credentials, install project dependencies, prepare workspaces |
| `install` | Base installation and selected project dependencies |
| `activate` | Credential preparation only |
| `workspace` | Selected project workspaces only |
| `build` | Install and prepare workspaces; disable persistence and skip credential activation |

`--plan` (or `--dry-run`) prints commands without executing them. Unknown profiles
and phases fail before installation. Each workflow's `profile.sh` declares
`INSTALL_STEPS` and `WORKSPACE_STEPS`, with paths relative to this directory.
The runner owns the shared bootstrap and activation phases. Normal setup activates
credentials before project installers, preserving access to private repositories.

There are three setup choices:

- `base`: built-in shared bootstrap and credential activation, with no project setup.
- `triton`: the Triton development environment.
- `iree`: the IREE development environment.

The `triton` profile includes Triton prerequisites, TokenSpeed preparation, and a
Triton checkout. Docker and Kubernetes select this profile for their `triton`,
`triton-mi450`, and `triton-b0` services. `iree` includes its dependencies and checkout.
The `ck` and `rocmlir` services select `base` because their images already contain
project dependencies.

Other installers remain standalone recipes. Some capture one-off or historical
setups; review their assumptions before running them directly. For example:

```sh
bash ~/scripts/setup/linux/packages/aiter.sh
```

A standalone install/build script belongs in `packages/`. A workflow directory
under `profiles/` is useful when dependencies and workspace preparation need to
run together, as they do for Triton and IREE.

Installation can require a public checkout to read its dependency requirements
(TokenSpeed does). Private workspace preparation must follow credential activation
when the selected project requires it. Build mode does not run personal credential setup.

## Locations

- `DEVENV_PERSISTENT_ROOT` selects mounted persistent storage. Set it to an empty
  string to disable discovery. Omit it to retain the old host/PVC discovery behavior.
- `DEVENV_WORKSPACE_ROOT` selects the parent of IREE, Triton, and TokenSpeed checkouts;
  the default is `$HOME`. Other standalone project scripts retain their documented paths.
- Helpers and preset references resolve from the running scripts checkout, so these
  entry points also work when the repository is cloned somewhere other than `~/scripts`.

Docker supplies `/zyin` during startup; Kubernetes supplies its PVC mount or disables
persistence for an ephemeral pod. Docker-specific base-image pulls live in
`containers/docker/pull-base-images.sh`; remote provisioning invokes that optional operation
after preparing the host.

## Docker build and startup

`scripts/containers/docker/base.dockerfile` clones scripts and runs:

```sh
bash /root/scripts/setup/linux/setup.sh "${SETUP_PROFILE:-${SERVICE_NAME:-base}}" --phase build
```

`SETUP_PROFILE` falls back to `SERVICE_NAME`, then `base`. The existing service names
and `dbuild` commands stay the same. Compose startup runs
`scripts/setup/linux/credentials/activate.sh` with `DEVENV_PERSISTENT_ROOT=/zyin`
before starting Zsh.

Image builds do not clone or read the vault. Activation restores missing SSH keys without replacing SSH configuration, links a mounted vault or clones it when needed, then invokes `python3 "$HOME/rc_files/.setup" credentials --vault "$VAULT_DIR"`. It passes `--persistent-root` when storage is available. Application token formats, private writes and login persistence belong to rc_files package hooks; the credential phase runs no installers. `VAULT_DIR` and `RC_FILES_DIR` override the default locations.

To save new or refreshed container logins, run `python3 ~/rc_files/.setup credentials --persistent-root /zyin --save`. Codex shares only `auth.json`; its preferences, instructions, plugins and sessions remain local. Native OS credential stores are not copied.

Update `scripts` and `rc_files` together and rebuild development images so the
Dockerfile, installed scripts, and startup command use the same paths. Default
builds clone `https://github.com/jerryyin/scripts.git`; local working changes are
included only when the build uses a snapshot containing them. To use a published
branch or alternate repository:

```sh
dbuild --build-arg SCRIPTS_REF=my-refactor-branch triton
dbuild --build-arg SCRIPTS_REPOSITORY=https://github.com/OWNER/scripts.git triton
```

[Docker regression tests](tests/docker/README.md) build local working-tree snapshots
through the installed `dbuild` command and check startup/restart for every development
service. The separate runtime check verifies a fully installed image with disposable
credentials and networking disabled.

## Verification

```sh
python3 -B -m unittest discover -s ~/scripts/setup/linux/tests -v
python3 -B -m unittest discover -s ~/scripts/containers/kubernetes/tests -v
```

Integration checks use temporary homes/checkouts and substitute external installers.
Docker configuration checks use the sibling `rc_files` checkout (override with
`RC_FILES_ROOT`) and PyYAML. They verify routing, phase selection, canonical paths,
and failure propagation without installing packages or reading personal credentials.
A real Docker build still requires a Docker daemon, registry access, and package feeds.
