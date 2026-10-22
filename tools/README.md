# Standalone tools

Each utility here serves one dedicated use case and works independently when copied out of the checkout. Tools must not import, load, or call each other or repository modules. There is no shared runtime library, registry, or configuration. Families that need to cooperate belong in an existing or new subject directory. See [AGENTS.md](AGENTS.md) for the contribution rules.

| Utility | Dedicated use case | Prerequisites |
| --- | --- | --- |
| `windows_wsl.py` | Execute a noninteractive script on Windows or WSL without nested quoting | Calling-host Python 3.10+; destination Bash or PowerShell 7; WSL interop for cross-OS calls |
| `github_attachments.py` | Download attachment links from one GitHub issue into a directory | Python, `requests`, `beautifulsoup4`; network access to the issue and files |
| `gdb_print_lanes.py` | Print one variable across selected GPU lanes | GDB Python with a `lane` command and an active debugging session |

## GitHub issue attachments

Pass the full issue URL, including owner and repository. The default output directory is `downloads` in the current directory. This retains the existing anonymous HTTP downloader; it does not implement GitHub sign-in.

```bash
python3 -m pip install requests beautifulsoup4
python3 tools/github_attachments.py https://github.com/OWNER/REPO/issues/123 --output-dir ./attachments
```

## GDB lane printing

Load the script in the debugger, then pass a variable expression and an inclusive lane range. Without a range it uses lanes 1 through 32. It switches the selected lane and frame while inspecting values.

```text
(gdb) source /path/to/scripts/tools/gdb_print_lanes.py
(gdb) python print_variable_for_lanes("first.data", 0, 3)
```

## Windows / WSL execution

`windows_wsl.py` runs a UTF-8 script on a named operating system without nesting PowerShell, WSL's configured shell, and Bash command strings. It needs Python 3.10+ on the calling host, Bash in the selected WSL distribution, and PowerShell 7 (`pwsh.exe` on PATH) for Windows commands. WSL-to-Windows execution needs Windows executable interop enabled.

### Usage

Write the commands into a local file using the destination language, then invoke the helper. The script path belongs to the calling host; `--cwd` belongs to the destination. No copy of this helper is needed on the destination.

From Windows PowerShell, with a `check.sh` containing `git status --short --branch`:

```powershell
python \\wsl.localhost\Ubuntu\home\zyin\scripts\tools\windows_wsl.py wsl .\check.sh --distro Ubuntu --cwd /home/zyin/scripts
```

From WSL, with a `check.ps1` containing `Get-Command git, gh | Select-Object Name, Source`:

```bash
python3 ~/scripts/tools/windows_wsl.py windows ./check.ps1
```

The same interface works for `windows` from Windows and `wsl` from WSL. Omit `--distro` to use Windows' configured default distribution or, when already in WSL, the current distribution. Requesting that current distribution executes Bash directly. Paths above are examples of this machine's checkouts; discover the checkout on other hosts.

### Execution contract

- Commands run without shell profiles. Windows uses PowerShell 7 and refreshes PATH from the machine and user settings, retaining inherited additions. WSL uses Bash explicitly, not the configured interactive shell; commands available only through a shell profile need an explicit path or setup in the script.
- The destination directory defaults to that user's home. `--cwd` must be absolute and in the destination's path format. Argument paths inside the script are not translated. A failed directory change stops execution.
- Script text travels through standard input, bypassing outer-shell interpolation and command-line size limits. UTF-8 BOMs and CRLF line endings are accepted. Output and errors stay on separate streams; the child exit code is returned.
- This is a noninteractive command block, not an interactive terminal or a file installation. Standard input carries the script and is unavailable for prompts or program input. Do not use it for editors, sign-in, or UAC approval.
- The block has no destination filename: `$PSScriptRoot`, `$PSCommandPath`, and Bash's source filename do not refer to the original local file. To run an existing destination script that needs its filename, invoke that file from the block, for example `& 'C:\path\task.ps1'`.
- Bash enables `errexit` and `pipefail`, subject to Bash's normal conditional-command exceptions. PowerShell stops on cmdlet errors and returns the last native program's exit code; check `$LASTEXITCODE` after each native command when earlier failures must stop a multi-command block. An explicit `exit N` is preserved.
- The helper uses existing permissions and credentials. It neither installs missing tools nor requests elevation. Script contents are executed with the same authority as a normal command, so review them normally.

## Alternatives Considered

- A separate directory for every small utility adds little navigation value. A flat `tools/` directory with explicit use cases keeps these isolated scripts discoverable.
- Shared helpers or a dispatcher would couple unrelated utilities. Each tool remains independently runnable; cooperating tools must move to a subject directory.
- Direct `wsl.exe --exec program arguments` is sufficient for simple commands. It still leaves the caller responsible for quoting its own arguments, choosing the host, and transferring multiline code.
- Nested `-Command` / `bash -c` strings require escaping for each intervening shell. That is the recurring source of failures this helper removes.
- A JSON command protocol or remote service would add another interface and dependencies. A local script file plus a fixed shell launcher covers the maintenance work here.

## Tests

From the repository root, install the downloader's two dependencies in a test environment, then run all tests:

```bash
python3 -B -m unittest discover -s tools/tests -v
```

The structure check rejects nested runtime directories, undocumented utilities, repository/sibling imports, dynamic import helpers, import-path access, and literal sibling-script dependencies. Its negative cases deliberately introduce these violations to verify rejection. Copied-file tests exercise each utility away from its source location. The one-use-case rule and indirect dependencies still require code review.

The [CI workflow](../.github/workflows/tools.yml) runs portable checks on pushes and pull requests that touch tools. Windows/WSL integration tests are skipped on Linux hosts without WSL. Run them on both Windows and WSL when changing that tool:

```powershell
python \\wsl.localhost\Ubuntu\home\zyin\scripts\tools\tests\test_windows_wsl.py
```

```bash
python3 ~/scripts/tools/tests/test_windows_wsl.py
```

Tests use temporary script files and read-only destination probes. They cover literal shell characters, Unicode, BOM/CRLF input, source paths with spaces and quotes, destination directories, separate output streams, failure status, scripts beyond the Windows command-line limit, and direct execution inside the current WSL distribution.
