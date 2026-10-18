# Environment installer checks

On Linux, run:

```sh
python3 -B -m unittest discover -s ~/scripts/docker/env/tests
```

The Codex checks use local CLI substitutes and temporary homes. They perform no
network requests or authentication. Cursor checks use the actual deployer from
`~/rc_files`, or the checkout selected by `RC_ROOT`, with temporary workspaces.
They verify local edits survive setup and source replacements remain visible
through installed links.
