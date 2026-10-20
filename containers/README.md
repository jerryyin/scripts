# Container workflows

| Directory | Responsibility |
| --- | --- |
| [docker](docker/README.md) | Pull base images, inspect bind mounts, work with Dockerfiles |
| [kubernetes](kubernetes/README.md) | Configure cluster connections, storage, and interactive pods |

Linux provisioning lives in [setup/linux](../setup/linux/README.md). Container launchers select images, storage, and setup profiles, then call that public setup interface.

This repository owns Dockerfiles, Compose services, and the `dbuild`, `drun`, and `dcompose` commands. Command checks live in [docker/tests](docker/tests); image build and startup checks live under [setup/linux/tests/docker](../setup/linux/tests/docker/README.md).
