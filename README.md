# Scripts

Development setup, execution infrastructure, and GPU engineering tools.

| Directory | Owns |
| --- | --- |
| [setup](setup/README.md) | Linux and Windows installation, credentials, project workspaces, remote provisioning |
| [agent](agent/README.md) | Claude and Cursor sessions, progress and verdict tracking for unattended runs |
| [tools](tools/README.md) | Independent, single-purpose utilities: Windows/WSL execution, GitHub attachments, GDB lane printing |
| [containers](containers/README.md) | Docker image utilities and Kubernetes connections, pods, and storage |
| [profiling](profiling/README.md) | Hardware trace capture, analysis, and event replay |
| [am](am/README.md) | AM/FFM simulator execution, downloads, and instruction trace analysis |
| [board](board/README.md) | GPU access coordination and board preparation |
| `triton`, `iree`, `llvm`, `tokenspeed` | Project experiments, reproducers, analysis, build configuration, and project-specific benchmarks (for example [tokenspeed](tokenspeed/README.md), `iree/thirdparty_bench/`) |
| [numerics](numerics/README.md) | Numerical comparison and test-data generation |
| `archive` | Historical workflows |

## Placement rules

- **Category first, project second.** If a tool does what a cross-project subject owns
  (`profiling/` for traces and kernel timing, `numerics/` for comparing numbers, `board/` for GPU
  access, `agent/unattended/` for generic looped-execution infrastructure, `am/` for the simulator),
  it goes there even when one project is its first caller. A project directory holds only what
  depends on that project: its server, its source, its log and dump formats. A tool that mixes
  both is split: the generic core in the subject directory, a thin project wrapper or a
  project-specific input file in the project directory. Records of a particular run (results,
  provenance) stay with the investigation that produced them, never here.
- **One owner per job.** Before writing a tool, search the subject READMEs and `git grep` here, and
  the team repositories the work already uses (for example a benchmark repository whose harness
  runs the same measurement), for one that already does the job. If one exists, use it; if it falls
  short, change it (a new flag, mode or importable function) rather than adding a sibling that does
  nearly the same thing. When the owner is a team repository, the change goes there, on a branch
  through its review, not into a tool here. Configuration an owner defines, such as a server's
  flags, is part of its job. A new file here is for a job nothing here or in those repositories
  does; a shared helper that two tools need lives once, with its owner.
- **Analysis is a tool too.** A script that turns measurements into a reported number lands here (or
  in the team repository whose format it reads) before the result is admitted; the full rule is
  "Tools behind a result" in triton-investigations' `orchestration/local/prompts/worker.md`.
- Group files by subject, and keep frequently used subjects directly accessible
  from the root. Add a parent directory when it helps navigation: `containers/`
  groups Docker and Kubernetes; `agent/` groups coding-agent sessions and unattended work.
- Each subject owns its scripts, tests, configuration, and experiments. Project
  directories such as `triton/`, `iree/`, and `tokenspeed/` own work on those projects.
  For example, Triton's MoE capture script calls the shared `profiling/prof.sh` wrapper.
- Installation and workspace preparation belong under `setup/<platform>`.
- Standalone install/build recipes live in `setup/linux/packages/`. The Triton
  and IREE workflows keep their plans, dependencies, and workspace setup together
  under `setup/linux/profiles/`. Root project directories own the work done with those projects.
- Docker and Kubernetes select an image, storage, and a setup profile. They call the
  public setup interface; environment installers do not discover Docker services.
- Keep helpers and tests with their owner. `setup/linux/lib` is internal to Linux setup.
- Keep utilities with their subject: `profiling/` owns hardware trace capture,
  `am/` owns simulator execution, and `agent/claude/` owns Claude session inspection.
- `tools/` is for standalone utilities with one dedicated use case and no dependency
  on another tool or repository module. Shared implementations and cooperating
  workflows belong in a subject directory. Its local `AGENTS.md` and CI tests
  enforce this boundary; prerequisites and usage stay in its README.
- `rc_files` owns configured applications, their dependencies, settings, plugins and application credential hooks. This repository owns machine provisioning, vault availability and interactive sign-in, everyday utilities and project workflows. Credential values and runtime data stay outside both repositories.

See [Linux setup](setup/linux/README.md) for profiles, lifecycle phases, and the
`rc_files` build and startup contract. [Docker helpers](containers/docker/README.md)
operate on images and Dockerfiles; [Docker regression tests](setup/linux/tests/docker/README.md)
live with the Linux setup they verify.
