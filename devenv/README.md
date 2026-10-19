# Development environment utilities

Tools for setting up and maintaining workstations and development sessions, including editors, shells, authentication and application integration.

- [Native Windows setup](windows/README.md) installs tools, links personal configuration, prepares notes and Obsidian, and installs WSL when needed.
- [Linux setup](../docker/env/min.sh) remains the entry point for provisioning an Ubuntu environment, including inside WSL.

`scripts` owns system installation and setup orchestration. The separate [rc_files repository](https://github.com/jerryyin/rc_files) owns authored application settings and its independent Python setup. Notes and Obsidian preferences belong to the notes repository; credentials and application runtime data keep their own storage.

Find individual utilities with `rg --files devenv` and inspect their usage before running them on the intended host.
