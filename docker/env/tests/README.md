# Environment installer checks

On Linux, run:

```sh
python3 -B -m unittest discover -s ~/scripts/docker/env/tests
```

The Codex checks use local CLI substitutes and temporary homes. Vault checks cover Docker/GitHub credentials and ensure the retired Claude profile cannot overwrite application state. They perform no network requests or authentication. Agent configuration placement and ownership checks live with the shared deployer in `rc_files/.setup/tests`; Claude settings generation checks live in `rc_files/claude/.setup/tests`. Application installers do not link personal rules into projects.
