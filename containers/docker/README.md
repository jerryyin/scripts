# Docker utilities

Docker-specific operations live here. Linux environment setup is owned by
[setup/linux](../../setup/linux/README.md).

- `pull-base-images.sh [service ...]` resolves and pulls Compose build base images.
  Defaults to `triton triton-mi450`; `COMPOSE_FILE` can select another Compose file.
- `verify_bind_mount_path.py` verifies host/container bind mount relationships.
- `dockerfile2bash.sh` and `run_dockerfile.sh` are the existing Dockerfile utilities.

Dockerfiles and Compose services live here. Commands in `bin/` are owned by this repository:

- `dbuild [build options...] SERVICE` builds the selected service.
- `drun SERVICE` starts it in the background without building the image.
- `dcompose ARGS...` forwards arguments to Docker Compose.

Linux bootstrap links these commands into `~/.local/bin`, which the shell includes on `PATH`. They also work directly from `containers/docker/bin/`, without loading a shell profile. Each command resolves its installed link to find this checkout's Compose file; `COMPOSE_FILE` selects another file and relative overrides are interpreted from the caller's directory. The commands retain the user/service/date/time project naming and `drun` removes the old `zyin-tools-setup` container when present.

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
