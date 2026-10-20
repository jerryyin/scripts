# Setup regression checks

```sh
python3 -B -m unittest discover -s setup/linux/tests -v
python3 -B -m unittest discover -s containers/kubernetes/tests -v
```

Checks cover installer configuration ownership, shell setup, module boundaries,
profile order, failure propagation, explicit storage paths, relocated checkouts,
and the `rc_files` Dockerfile and Compose entry points. They use temporary homes and
substitute external installation steps. Fixture commands reject unexpected package
installation, privileged operations, downloads, and remote access.

Docker entry-point unit tests need PyYAML and the sibling `rc_files` checkout; set
`RC_FILES_ROOT` to override that location. These execute the Dockerfile's setup RUN
shell command and Compose startup command against fixture checkouts. They do not
validate a complete image build, package feeds, native Windows execution, or a live cluster.

For opt-in checks using a real Docker Engine and installed images, see
[Docker regression tests](docker/README.md).

To validate Compose without starting services:

```sh
docker compose -f ~/scripts/containers/docker/docker-compose.yml \
    --profile '*' config --no-consistency --quiet
```

The shared container name is deliberate (only one development service runs at a
time), so rendering every profile requires `--no-consistency`.

Session credential checks verify phase separation, vault and storage arguments, repeated activation and failure propagation. App rendering and persistence checks live in `rc_files/.setup/tests/test_credentials.py`; Claude settings checks live in `rc_files/claude/.setup/tests`.

Enable the cross-repository check against real package hooks with fake credentials:

```sh
RC_FILES_TEST_ROOT="$HOME/rc_files" python3 -B -m unittest discover -s setup/linux/tests -p test_credentials.py -v
```
