# Container workflows

| Directory | Responsibility |
| --- | --- |
| [docker](docker/README.md) | Pull base images, inspect bind mounts, work with Dockerfiles |
| [kubernetes](kubernetes/README.md) | Configure cluster connections, storage, and interactive pods |

Linux provisioning lives in [setup/linux](../setup/linux/README.md). Container launchers select images, storage, and setup profiles, then call that public setup interface.

The separate `rc_files` repository owns Dockerfiles, Compose services, and the `dbuild` shell command. Their integration tests live under [setup/linux/tests/docker](../setup/linux/tests/docker/README.md).
