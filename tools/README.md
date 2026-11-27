# Standalone tools

Each utility here serves one dedicated use case and works independently when copied out of the checkout. Tools must not import, load, or call each other or repository modules. There is no shared runtime library, registry, or configuration. Families that need to cooperate belong in an existing or new subject directory. See [AGENTS.md](AGENTS.md) for the contribution rules.

| Utility | Dedicated use case | Prerequisites |
| --- | --- | --- |
| `windows_wsl.py` | Execute a noninteractive script on Windows or WSL without nested quoting | Calling-host Python 3.10+; destination Bash or PowerShell 7; WSL interop for cross-OS calls |
| `github_attachments.py` | Download attachment links from one GitHub issue into a directory | Python, `requests`, `beautifulsoup4`; network access to the issue and files |
| `gdb_print_lanes.py` | Print one variable across selected GPU lanes | GDB Python with a `lane` command and an active debugging session |
| `ckpt_index_check.py` | Check a safetensors checkpoint for completeness against its `model.safetensors.index.json` and split its bytes by tensor kind, from `stat` and shard headers only; optionally check its identity against a Hugging Face Hub repo revision without downloading it | Python 3.8+ standard library; read access to the checkpoint directory; for `--hub-repo`, HTTPS access to the Hub (and `HF_TOKEN` for a gated repo) |
| `py_ast_equal.py` | Prove a Python file's code unchanged between two versions (comments, formatting and optionally docstrings ignored), listing the definitions that differ | Python 3.8+ standard library; `git` on PATH for `--git` |

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

## Checkpoint completeness

`ckpt_index_check.py [--split-bytes] [--by-pattern] [--json] CHECKPOINT...` checks each checkpoint directory (a Hugging Face snapshot's symlinks are followed) against its `model.safetensors.index.json` from `stat` alone: shard and tensor counts, indexed shards missing or empty, `*.safetensors` files the index does not name, and shard bytes against the index's `total_size`. `--split-bytes` also reads each shard's JSON header (never tensor data) and sums tensor bytes into routed experts, shared experts, the MTP layer (layers at or beyond `config.json` `num_hidden_layers`, or `mtp`/`nextn` names) and other, checked against `total_size`. `--by-pattern` (implies `--split-bytes`) also sums bytes and tensor counts per tensor-name pattern (every all-digit name component replaced by `N`), with the dtypes and the first tensor's shape: the input of a per-rank memory budget (`bytes_by_pattern` in `--json`). `--json` prints one object per checkpoint. Exit status 1 when an index, an indexed shard or (with `--split-bytes`) an indexed tensor is missing.

```bash
python3 tools/ckpt_index_check.py --split-bytes /models/SOME-CHECKPOINT
```

`--hub-repo REPO [--hub-revision REV]` also compares the checkpoint with that Hub revision, without downloading it. It lists the revision with the Hub API (`?blobs=true`, which resolves REV to a commit) and, for every file on both sides, compares the size and the content identity the Hub records: the LFS sha256 for LFS files, the git blob SHA-1 of the bytes for the rest. The local side is hashed; `*.safetensors` only with `--hash-shards` (reads every shard; `--jobs N` in parallel, default 6). Each common shard's safetensors header is fetched with two HTTP Range requests and compared with the local one, and for each `--sample-shard NAME` (repeatable) `--sample N` tensors (default 10: the smallest `min(4, N)`, then a `--seed S` random choice of the rest; at most the first 64 MiB of each) are byte-compared over HTTP Range. A file in a Hugging Face cache snapshot also reports the blob it links to, the identity the download recorded (a stat-only claim, not a hash of the bytes). A server that answers a Range request with anything but 206 and the exact length is an error, so a multi-GB shard is never read whole. `HF_TOKEN` is sent when set; without it a gated, private or missing repo is reported as an `error`, not a crash. `--hub-endpoint` (default `HF_ENDPOINT` or `https://huggingface.co`) points it at a mirror. The facts go under `hub` in the `--json` object (`files`, `headers_equal`, `tensor_samples`, `only_hub`, `only_local`, `summary`, `all_checked_equal`); the text output prints one summary line plus the differing files and the samples. With `--hub-repo`, exit status 1 also when a checked fact differs or a file is on one side only.

```bash
python3 tools/ckpt_index_check.py --json --hub-repo deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --hub-revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --sample 2 --sample-shard model-00048-of-00048.safetensors /models/DeepSeek-V4.1-Flash-1017
```

## Python AST equivalence

`py_ast_equal.py [--ignore-docstrings] OLD.py NEW.py` or `py_ast_equal.py [--ignore-docstrings] --git REPO REV_A REV_B PATH...` parses both versions of each file with `ast` (which drops comments and formatting) and compares them; `--ignore-docstrings` also drops the module's and every class's and function's docstring. Use it to prove a comment-, docstring- or move-only edit leaves what a module executes unchanged. With `--git`, each repository-relative PATH is read at both revisions with `git show`. It prints `equal PATH` or `differs PATH` per file, and under a differing file each definition whose own statements differ, by qualified name (`Class.method`, `outer.inner`, `<module>` for statements outside any definition; a repeated name gets `#2`), as `changed`, `added` or `removed`. A change inside a nested definition is listed under that definition only. Exit status 0 when every file is equal, 1 when any differs, 2 on a usage, read or syntax error.

```bash
python3 tools/py_ast_equal.py --ignore-docstrings --git /path/to/repo BASE_COMMIT HEAD_COMMIT path/to/module.py
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
