# devenv/ — developer-machine hygiene

Things that fix or inspect **your own workstation, container or editor session**. Nothing
here touches a GPU, a simulator or a trace. If a script is about the machine you work *on*
rather than the hardware you measure, it goes here.

```
devenv/
├── README.md                     # this file
├── clean-cursor-attach.sh        # clean up Cursor server state left behind in a container
├── list_claude_sessions.py       # list Claude Code sessions, incl. headless ones
├── download_github_attachment.py # fetch a GitHub issue/PR attachment from the CLI
├── fix-windows-ssh-perms.sh      # fix SSH key permissions from WSL/Git-Bash
├── fix-windows-ssh-perms.ps1     # the same, as PowerShell for a Windows host
├── gdb_print_lanes.py            # GDB helper: print a variable across all wave lanes
└── torch_event_replay.py         # replay torch profiler events from a TraceLens trace
```

## Files

- `clean-cursor-attach.sh <container> [--all]` — removes stale Cursor server state that
  accumulates when repeatedly attaching to a dev container.
- `list_claude_sessions.py [filter]` — lists sessions on this machine including headless
  (`-p`) ones that the `claude --resume` picker does not show. Prints the session id and its
  cwd; `cd` there and `claude --resume <id>`.
- `download_github_attachment.py` — pulls an attachment off a GitHub issue or PR so you can
  inspect it directly instead of through a browser. Needs `requests` and `bs4`.
- `fix-windows-ssh-perms.sh` / `.ps1` — Windows refuses SSH keys whose ACLs are too
  permissive, which is the usual cause of `UNPROTECTED PRIVATE KEY FILE` when using a key
  from a Windows filesystem. Use the `.sh` from WSL or Git Bash, the `.ps1` from PowerShell.
- `gdb_print_lanes.py` — source this inside GDB to print one variable's value across all
  lanes of a wave in a compact table, rather than stepping through them one at a time.
- `torch_event_replay.py` — replays selected ops (default `aten::addmm`) out of a TraceLens
  event trace. Needs `TraceLens` and `pandas`.
