# Docker utilities

Docker-specific operations live here. Linux environment setup is owned by
[setup/linux](../../setup/linux/README.md).

- `pull-base-images.sh [service ...]` resolves and pulls Compose build base images.
  Defaults to `triton triton-mi450`; `COMPOSE_FILE` can select another Compose file.
- `verify_bind_mount_path.py` verifies host/container bind mount relationships.
- `dockerfile2bash.sh` and `run_dockerfile.sh` are the existing Dockerfile utilities.

Dockerfiles and Compose services live here. The optional `dbuild` shell command delegates to this Compose model. See the [Linux setup contract](../../setup/linux/README.md#docker-build-and-startup)
and [Docker regression tests](../../setup/linux/tests/docker/README.md) for builds
and startup verification.

Personal Docker configuration remains in `rc_files`. Shell commands use `~/scripts/containers/docker/docker-compose.yml`; `COMPOSE_FILE` can select another model. Build contexts are this directory.
