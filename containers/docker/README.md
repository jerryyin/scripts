# Docker utilities

Docker-specific operations live here. Linux environment setup is owned by
[setup/linux](../../setup/linux/README.md).

- `pull-base-images.sh [service ...]` resolves and pulls Compose build base images.
  Defaults to `triton triton-mi450`; `COMPOSE_FILE` can select another Compose file.
- `verify_bind_mount_path.py` verifies host/container bind mount relationships.
- `dockerfile2bash.sh` translates a limited set of Dockerfile instructions to shell.
- `run_dockerfile.sh` executes supported Dockerfile instructions directly on the host.
  These helpers implement only part of Dockerfile syntax; `dbuild` uses Docker itself.

Dockerfiles and Compose services live here. Commands in `bin/` are owned by this repository:

- `dbuild [build options...] SERVICE` builds the selected service.
- `drun SERVICE` starts it in the background without building the image.
- `dcompose ARGS...` forwards arguments to Docker Compose.
- `dockrun [options...] IMAGE [command...]` runs an image directly with GPU access,
  host networking, and the existing `/data` and home mounts. It retains the old
  alias's `sudo` invocation and gives each launch a distinct name. It does not use Compose.

Linux bootstrap links these commands into `~/.local/bin`, which the shell includes on `PATH`. They also work directly from `containers/docker/bin/`, without loading a shell profile. The Compose commands resolve their installed links to find this checkout's model; `COMPOSE_FILE` selects another file and relative overrides are interpreted from the caller's directory.

`drun` validates the model and service before starting anything. It prints a distinct user/service/time/process project name, or preserves an explicit `COMPOSE_PROJECT_NAME`. It never deletes the legacy tools container. Reuse that project name for later commands; `dcompose` requires a project for lifecycle operations and never invents one. Put `-p NAME` first or export `COMPOSE_PROJECT_NAME`. `build`, `config`, `ls`, and `version` can run without selecting a project.

```sh
dbuild triton
COMPOSE_PROJECT_NAME=my-triton drun triton
dcompose -p my-triton logs triton
dcompose -p my-triton down triton
```

Services use Compose profiles. Name the service in `logs` and `down` as above,
or explicitly enable its profile (`dcompose -p my-triton --profile triton down`).

`pull-base-images.sh` locates its default Compose file beside itself and returns nonzero on missing prerequisites, invalid configuration, or failed pulls. Remote provisioning explicitly treats prefetch as optional and reports its failure separately from setup completion.

See the [Linux setup contract](../../setup/linux/README.md#docker-build-and-startup)
and [Docker regression tests](../../setup/linux/tests/docker/README.md) for builds
and startup verification.

Personal Docker configuration remains in `rc_files`; Docker command implementations and Compose locations stay here. Build contexts are this directory. Reload Zsh after updating an existing session to retire its old function definitions; newly opened shells discover the installed commands through `PATH`.

## Checks

```sh
python3 -B -m unittest discover -s containers/docker/tests
```

Command checks use relocated checkouts, installed links and a fake Docker executable. They verify arguments, Compose overrides, naming and failure handling without a daemon or image build.

## Alternatives considered

Keeping the commands in `.zshrc` makes personal configuration depend on this repository's layout. Sourcing a Docker shell fragment would retain shell-specific loading. Executable commands keep the existing interface, work from Bash and Zsh, and locate their own Compose model. Bootstrap reuses the configuration installer's generic link operation rather than adding another link manager.
