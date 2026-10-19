# Docker regression tests

These checks are opt-in and need a running Docker Engine, Compose, Git, and Python 3.
They use disposable containers and never mount the user's home or Docker socket
inside a test container.

See [the recorded validation results](VALIDATION.md) for coverage, fixes, and
the limits of the latest run.

## Build and startup contracts

```sh
python3 -B setup/linux/tests/docker/integration.py \
    --docker-command 'sudo -n docker' \
    --logs /tmp/scripts-docker-contracts
```

Use `--rc-files /path/to/rc_files` for a non-sibling checkout. Omit
`--docker-command` when Docker is available without sudo.

The suite takes a snapshot of the local working tree, substitutes external
installers with recorders, includes the Git snapshot in a temporary base image,
and builds the actual
`rc_files` Dockerfile through every development Compose service. Checks include
the actual `dbuild` command for every service, build-phase isolation, startup/restart, argument precedence,
and failed builds. The inherited entry point is deliberately invalid to verify
that the Dockerfile clears it. Temporary services have no devices, host mounts,
or network access at runtime. Only the builds need network access.

Images and containers created by the suite are removed
when it finishes. Docker's build cache remains. Logs stay in the requested directory.
This suite validates the integration contract, not external package installation.

## Real installed image

Build the actual Dockerfile against a local Git snapshot using `SCRIPTS_REPOSITORY`
and `SCRIPTS_REF`, or a published branch. For an unpublished snapshot served on
localhost, build with `--network host`. To test local `rc_files` changes as well,
use a temporary base image containing a Git snapshot at `/root/rc_files`.

Then run the real runtime check with network disabled:

```sh
sudo docker compose -f ~/rc_files/docker/.docker/docker-compose.yml \
    --profile '*' config --no-consistency --format json > /tmp/validation-compose.json
python3 - <<'PY'
import json
from pathlib import Path
model = json.loads(Path('/tmp/validation-compose.json').read_text())
command = [part.replace('$$', '$') for part in model['services']['base']['command']]
Path('/tmp/validation-startup.json').write_text(json.dumps(command))
PY
sudo docker run --rm --network none \
    --mount type=bind,src="$PWD/setup/linux/tests/docker/runtime.sh",dst=/validation-runtime.sh,readonly \
    --mount type=bind,src=/tmp/validation-startup.json,dst=/validation-startup.json,readonly \
    YOUR_TEST_IMAGE bash /validation-runtime.sh
```

This checks the canonical directory layout, installed binaries, dotfile links, absence of baked-in credentials,
real activation with dummy credentials, permissions, rotation, repeat activation,
canonical credential entry points, and the actual Compose startup command. It must run in a
disposable container with no host-home mount; its dummy `/zyin` is created inside
the container. It does not authenticate to external services or exercise a GPU.

To check the installed Claude CLI and settings refresh against a local fixture
gateway, run the opt-in `rc_files` tests inside the same disposable image:

```sh
sudo docker run --rm --network none YOUR_TEST_IMAGE bash -c '
  PATH="$HOME/.local/bin:$PATH" RC_CLAUDE_CLI_TEST=1 SCRIPTS_ROOT="$HOME/scripts" \
    python3 -B -m unittest discover -s "$HOME/rc_files/claude/.setup/tests" -v
'
```

The fixture gateway uses loopback and dummy keys; no external service is contacted.
