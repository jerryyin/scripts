# Machine and development setup

Installation, credentials, and workspace preparation for development machines.

- [Linux setup](linux/README.md) supports Ubuntu hosts, WSL, Docker, and Kubernetes.
- [Native Windows setup](windows/README.md) installs tools, deploys configuration,
  prepares notes and Obsidian, and installs WSL when needed.
- Coding-agent sessions and unattended-run utilities live under
  [agent](../agent/README.md). Other scripts live with their subjects, such as
  [profiling](../profiling/README.md) and [containers](../containers/README.md).

`scripts` owns installation and orchestration. The separate
[rc_files repository](https://github.com/jerryyin/rc_files) owns authored application
settings and its independent Python setup. Notes and Obsidian preferences belong to
the notes repository; credentials and application runtime data keep their own storage.

## Entry points

Use `linux/setup.sh PROFILE` for Linux setup, `windows/bootstrap.ps1` for Windows,
and each utility's path under its owning directory. Linux offers built-in `base`
setup plus the `triton` and `iree` workflows under `linux/profiles/`. Standalone
install/build recipes live in `linux/packages/`.
Docker and Kubernetes select profiles in their own configurations.

The `scripts` and `rc_files` checkouts must both use this layout. Rebuild development
images after updating the repositories so Compose startup uses the same paths.
