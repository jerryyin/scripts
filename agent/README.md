# Coding-agent utilities

Manage coding-agent sessions and track unattended work.

| Directory | Utility |
| --- | --- |
| `claude/` | `list_sessions.py`: inspect local Claude sessions |
| `cursor/` | `clean-attach.sh`: clean stale container or SSH attach sessions |
| [unattended](unattended/README.md) | Record progress and check that verdicts correspond to the latest run |

Agent installation lives in [setup/linux/packages](../setup/linux/packages/). Engineering scripts live with their subjects, listed in the [repository overview](../README.md). Application settings and agent instructions are owned by the separate `rc_files` repository.
