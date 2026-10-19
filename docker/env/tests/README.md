# Environment installer checks

On Linux, run:

```sh
python3 -B -m unittest discover -s ~/scripts/docker/env/tests
```

The Codex checks use local CLI substitutes and temporary homes. They perform no
network requests or authentication. Agent configuration placement and ownership checks live with the shared deployer in `rc_files/.setup/tests`; application installers do not link personal rules into projects.
