# Native Windows development environment

One repeatable bootstrap for **PowerShell on Windows**. It also installs WSL
and Ubuntu 24.04 when missing. Initialize Ubuntu separately with the existing
`docker/env/min.sh`. Requires Windows 10 build 19041+ or Windows 11, WinGet,
and access to GitHub and the package feeds.

## Design

PowerShell coordinates Windows installers and settings; WinGet installs Windows
applications, Git owns personal configuration, and upstream tools supply current
plugin code. The existing Ubuntu `min.sh` remains the Linux installer. Native
Windows Vim/PowerShell use small adapters around the shared rc files. They don't
try to reproduce every Linux shell feature. Native psmux is a separate tmux-like
implementation and should be evaluated in daily use before assuming full parity.

This keeps the dependency chain to Windows/WinGet, Git and PowerShell. WinGet
Configuration is another supported way to express package/system state if this
grows to multiple machine profiles; it isn't required for this personal bootstrap.

## First Windows machine setup

In PowerShell, install Git if needed and obtain this repository:

```powershell
winget install --id Git.Git --exact --source winget
# Restart PowerShell once if git isn't on PATH yet.
git clone https://github.com/jerryyin/scripts.git "$env:USERPROFILE\scripts"
powershell -NoProfile -ExecutionPolicy Bypass -File "$env:USERPROFILE\scripts\devenv\windows\bootstrap.ps1" -Apply
```

The execution policy flag applies to this process only, including scripts read
over a UNC path; no machine-wide policy is changed. Package installers can
request UAC elevation. Close Obsidian before applying changes. With no switches
the script prints a plan. Fresh WinGet installs use current available packages;
existing packages are retained. Node older than 22.19 is upgraded because the
shared CoC runtime requires it. Missing plugins are installed from current
upstream releases. Ordinary reruns retain installed plugins; `-Apply -Update`
explicitly refreshes Vim/CoC/Obsidian plugins, themes and the Codex ACP adapter.

The starting point is a clean Windows account: neither WSL nor `rc_files`,
notes, fonts, native Vim, PowerShell 7 or psmux needs to exist. Git and this
scripts checkout are obtained by the commands above; the bootstrap installs
the remaining tools and Git clones `rc_files` and notes into the new account's
home. All destinations derive from that account, with no dependency on this
machine's user name, WSL filesystem, or existing repositories. On reruns,
existing Git checkouts are reused without discarding local changes.

```powershell
.\bootstrap.ps1                              # plan, no changes
.\bootstrap.ps1 -Check                       # read-only inventory; nonzero if incomplete
.\bootstrap.ps1 -Apply                       # install/configure or repair
.\bootstrap.ps1 -Apply -Update               # explicitly refresh installed plugins and ACP
.\bootstrap.ps1 -Apply -SkipWsl              # explicitly opt out of WSL installation
.\bootstrap.ps1 -Apply -InstallCodexDesktop  # also run the official desktop installer
.\bootstrap.ps1 -Apply -NotesPath C:\notes -RcPath C:\dev\rc_files
.\bootstrap.ps1 -Apply -SkipCredentials      # provision tools/settings without credential setup
```

## Native credentials

`priv.ps1` is the Windows counterpart to the credential phase in Ubuntu.
Bootstrap calls it after installing tools and before cloning private notes.
Its initial scope is GitHub, Docker, SSH and Codex:

| Tool | First run / rerun |
|---|---|
| GitHub | Keeps a working `gh` login; otherwise opens `gh auth login --web`. Sign into the account with access to the private vault and notes. GitHub CLI manages credential storage; Linux token templates are not copied. Additional accounts can be added with `gh auth login` and selected with `gh auth switch`. |
| Vault | Clones the existing private `jerryyin/vault` repository over HTTPS into `%USERPROFILE%\vault` if absent. Reuses an existing checkout with the expected origin, without pulling or resetting it. |
| Docker | Restores `docker_mkmhub_auth.txt` for `mkmhub.amd.com`, preserving other registries/settings. Uses an existing Docker credential helper through `docker login --password-stdin`; if no helper is configured, writes Docker's base64 auth entry with restricted Windows permissions. Docker Desktop is not installed by this phase. |
| SSH | Preserves existing Windows keys/config, restricts their permissions, enables the Windows `ssh-agent`, and loads keys. Optional `-SshSource` imports missing `id_*` private/public key pairs from a trusted folder. Encrypted keys may ask for a passphrase. Linux SSH configuration is not copied. |
| Codex | Keeps a working native login; otherwise runs `codex login` for browser sign-in. Existing auth/cache directories are not migrated from WSL. |

```powershell
.\priv.ps1                                  # plan only
.\priv.ps1 -Status                          # read-only presence/login inventory
.\priv.ps1 -Apply                           # native login + credential setup
.\priv.ps1 -Apply -SshSource E:\backup\.ssh   # import missing keys; never overwrite existing ones
.\priv.ps1 -Apply -SkipSshAgent              # leave agent service/loading to the operator
.\bootstrap.ps1 -Apply -VaultPath C:\private\vault -SshSource E:\backup\.ssh
```

The vault repository comes from `defaults/environment.json`; credentials themselves
stay outside scripts/rc_files and general configuration backups. Vault and key
files written by this phase allow only the current Windows user and SYSTEM.
Native credential destinations cannot be Git checkouts, symlinks or junctions.
If no SSH key exists, supply a trusted backup or create/register a key with
`ssh-keygen`; setup never generates or uploads a replacement identity silently.
Only agent service configuration requests UAC; credential reads/writes run as
the original user. When unset, Git's `core.sshCommand` is pointed at Windows
OpenSSH so Git and PowerShell use the same agent. Existing overrides are kept.
Pull the private vault yourself to obtain rotated Docker credentials, then
rerun `priv.ps1 -Apply`. GitHub and Codex token refresh stays with their tools.

## What is reused

| Environment | Source / behavior |
|---|---|
| WSL | Ensures Ubuntu 24.04 is installed; existing distributions and Linux configuration are preserved |
| Native Vim | A four-line `_vimrc` supplies machine paths and sources `rc_files/vim/.vimrc`; configuration is read directly from that Git checkout |
| Vim plugins / CoC | One shared plugin/extension list in `rc_files/vim/.vim/config/plugins.vim`; Git for Windows Bash runs existing POSIX helpers |
| Vim runtime | Native Python for LeaderF, Node >=22 for CoC, native ripgrep for search, native clangd for C/C++ semantic highlighting |
| Multiplexer | Native `psmux`, exposed as `tmux` in PowerShell; uses Windows ConPTY and native PowerShell panes |
| tmux keys | Generated from `rc_files/tmux/.tmux.conf`: backtick prefix, `v` / `S` / `s` splits, vi modes, `r` reload, `b` command-copy, F10 nested-session toggle, portable status settings |
| PowerShell | `vim`/`vi`, `tmux`, `dev`, `ll`, `la`, history search and word navigation; a Gruvbox prompt |
| Terminal | Your existing Gruvbox palettes, font settings, actions, and keybindings, merged with unrelated customizations |
| Everyday Windows tools | PowerToys, KDE Connect, Rufus, WizTree, VLC, GitHub CLI and 7-Zip, installed through WinGet without release pins |
| PowerToys Awake | Enabled with indefinite keep-awake and display allowed to sleep, matching the current machine; other settings are preserved |
| Notes | Reuses a registered `notes` Git checkout; otherwise clones once to `%USERPROFILE%\notes` |
| Obsidian | Uses the vault's Git-tracked plugin/theme choices, installs their latest code, registers the vault, installs the latest Codex ACP adapter |
| Git | Reuses identity if absent; preserves native credential helpers and existing settings |
| Codex | Seeds a missing config from the existing template with a Windows trust path; reuses shared instructions/skills without replacing existing ones |

`dev` runs `psmux new-session -A -s main`. Ordinary Terminal startup opens
PowerShell without forcing automatic attachment, matching the current
workstation's `.no-auto-tmux` preference. Native psmux is a separate
implementation: the Unix TPM shell scripts, zsh commands, and work-host-specific
environment aren't loaded. LLVM/GDB mappings remain in the shared vimrc; their
toolchains must be installed separately if used locally on Windows.

Pane navigation uses the same sequence as Ubuntu: press and release backtick,
then an arrow or `h/j/k/l` key. The Vim keys come from your Linux
`tmux-pain-control` declaration, which the original Windows adapter omitted.
The Windows adapter now reads those plugin choices and adds native equivalents;
it does not run Linux shell plugins. Arrow bindings are explicitly repeatable.
The status bar reuses the shared background and colored time/host
format; its Linux shell variable for the nested-session indicator is omitted.
The native PowerShell prompt inside each pane uses the same Gruvbox colors,
folder, Git branch/change marker and `➜`; psmux draws the pane borders and status
bar. No external prompt engine loads at startup. Oh My Posh remains installed;
set the user environment variable `DEVENV_PROMPT=oh-my-posh` to opt back in.

The native keyboard audit uses real key input in an isolated Windows home:

| Habit | Audit result with psmux 3.4.0 |
| --- | --- |
| Backtick then `h/j/k/l`, Ctrl variants | Pane movement verified |
| `H/J/K/L` | Resize directions verified; amount differs from Linux |
| `v`, `S`, `s`, `\|`, `\`, `-`, `_`, `%`, `"` | Pane splitting verified |
| `<`, `>`, Ctrl-p, Ctrl-n | Window swapping/navigation verified |
| F10 twice | Disable/restore keys and backtick prefix verified |
| `[`, then `q` | Enter/exit vi copy mode verified |
| `y`, `Y` | Command line without prompt / directory copied into buffer |
| Ctrl-s | Native session save verified |
| Ctrl-r, plugin install/update/clean, mouse clipboard | Configured; not verified by the keyboard audit |

Full parity is not claimed. In the isolated test, a requested five-column resize
changed 59 columns to 65, and five rows changed 19 to 21. New panes/windows also
reported the account home instead of inheriting the test directory despite the
shared `-c` bindings. Clipboard delivery to desktop applications could not be
verified from the automation session. Native session restoration is separate
from Linux tmux-resurrect's Vim session support.

The shared vimrc chooses plugins and commands by platform. LeaderF uses its
Python implementation on Windows and the existing build hook on Linux. Windows
uses Vim's built-in syntax plus CoC semantic highlighting; Polyglot remains
enabled on Linux. There is no plugin-autocommand rewriting and no separate
`vim-windows.vim` shim. Search and multiplexer keys have one shared definition,
selecting `rg`/`psmux` on Windows and the existing `ag`/`tmux` on Linux.

CoC reads `rc_files/vim/.vim/coc-settings.json` directly on Windows. Its existing
generic clangd server and `semanticTokens.enable` settings are reused; no extra
`coc-clangd` extension is installed. Native clangd is installed with WinGet.
C/C++ outline/navigation uses CoC on Windows; Ctags is not installed. Project
compile commands and include/toolchain paths must be valid for Windows; a Linux
build database cannot generally be reused unchanged. LLVM/GDB builds remain
project-specific. After pulling rc_files, reopen Vim to use the updated settings.

The existing MonacoLigaturized Nerd Font is retained when installed. Otherwise
the bootstrap installs that same family (Regular, Bold, Italic, BoldItalic)
per user from its latest upstream release. The family and archive name are
preferences; no release versions, revisions or download hashes are stored.
Font artifacts are downloaded over HTTPS and never committed.
Terminal uses your GruvboxMaterialHardDark palette; native Vim loads the shared
gruvbox-material configuration and explicitly enables Windows true color.

WSL installation runs `wsl --install --distribution Ubuntu-24.04 --no-launch`,
elevating that installer only when needed. It never reboots automatically.
If Windows requires a restart or distribution initialization, the bootstrap
finishes native setup, reports the pending step and exits with code 3010.
Restart Windows, launch Ubuntu once to create the Linux account, and rerun
`-Apply`. Run `min.sh` yourself inside Ubuntu when ready. Existing distributions
aren't converted, updated, relaunched, or made the default by this utility.

## Storage and reruns

### Terminal Vim and PowerShell

Vim is used through its terminal executable, `vim.exe`. The official Windows
package also bundles gVim; its base installer component cannot be deselected.
Bootstrap disables the installer's desktop/Start menu shortcuts and graphical
Explorer integration on fresh installs, then creates one terminal Vim Start menu
entry. On existing installs it backs up and removes the standard extra Vim
shortcuts that target that same installation. Custom shortcuts are preserved.

`PowerShell` is the single visible Windows Terminal profile launching native
PowerShell 7 with the chosen colors, font and shell profile. It uses the same
Windows account and files. The stock Windows PowerShell 5 and automatically
detected PowerShell 7 entries are hidden, retaining their settings. PowerShell's `ls` and `cat` are aliases for
`Get-ChildItem` and `Get-Content`; Bash syntax and flags are separate.
Vim shell commands use Git for Windows Bash at `Git/bin/bash.exe`. That Bash
runs on Windows; use `wsl` explicitly to enter the separate Ubuntu environment.

### Configuration ownership

* `defaults/environment.json`: repository URLs, Windows package IDs, WSL choice
  and font family/source. No plugin list or artifact version/hash locks.
* `notes/.obsidian/community-plugins.json`: the authoritative enabled plugin list.
* `notes/.obsidian/appearance.json` and plugin `data.json` files: the authoritative
  theme choice and settings. Setup preserves these Git-tracked files.
* `rc_files/vim/.vimrc`: the shared entry point; `config/plugins.vim` owns plugin
  and CoC extension choices, and `config/windows.vim` holds native OS settings.
* `defaults/terminal.json`: portable snapshot of the current Terminal preferences.
* `defaults/powertoys.json`: portable PowerToys preferences; merges Awake settings
  into the new account's settings without copying device-specific state.
* `%LOCALAPPDATA%\PersonalDevenv`: generated shell profile, state and backups.
* `%USERPROFILE%\_vimrc`, `.vim`, `.psmux.conf`: native editor/multiplexer state.
* Both PowerShell 5 and PowerShell 7 all-host profiles get one managed loader
  block. Existing profile content is kept and backed up.

The installer does not pull/reset existing checkouts, replace a separate WSL
notes checkout, or terminate existing sessions. Native sign-in and the limited
vault/key restoration above are handled by `priv.ps1`. In Copilot, choose Codex's Auto-detect
once if the fresh device hasn't selected the standard npm adapter path yet.
Open/restart PowerToys after setup to load Awake preferences. Pair KDE Connect
with your phone/devices on first use. PowerToys' existing startup preference is
preserved; the bootstrap does not add a separate startup task or background service.

Adding/removing an enabled plugin through Obsidian changes the vault's tracked
configuration. A fresh clone uses those choices directly, looks up repositories
in Obsidian's current community registry, and downloads their latest published
release. Ordinary reruns skip complete installations without querying upstream;
missing plugins/files are repaired. `-Apply -Update` checks for newer code while
preserving `data.json`. Themes use current upstream files, including CSS changes
that did not bump a version. Plugin identity is checked before activating code;
downloads finish in staging before replacing existing files. No plugin code is
tracked by this utility. Obsidian's own updater continues to work normally.

The current plugin version is discovered from the publisher's root manifest,
as used for Obsidian releases. A repository's GitHub "latest" release can belong
to a different plugin/channel, so it is not used to silently switch identities.

Vim downloads vim-plug only when absent, then uses `PlugInstall --sync` for
missing plugins. `-Update` refreshes vim-plug and runs `PlugUpdate --sync`, respecting
any branch or version choice explicitly declared in the shared vimrc. CoC
extensions and the Codex ACP adapter skip npm installation when already present;
fresh installs and explicit updates use current npm versions. To get newer tracked
configuration on an existing machine, pull the appropriate Git checkout and
rerun setup; the bootstrap never resets local work.

## Tests

The configuration suite runs on Windows PowerShell 5.1 with Git, without Pester.
It uses disposable directories and checks actual Git cloning into absent home
and repository directories, plus first-run/rerun
behavior, JSONC, preservation of unrelated settings, profile merging, vault
registration, latest-release discovery and updates, settings preservation,
font archives, and manifest identity validation.
WSL orchestration is tested with simulated command results (existing install,
fresh install, restart, deferred registration and failures); these tests don't
install WSL on the current machine. `-Network` also installs current real plugin
and theme releases in isolation and validates the latest font archive and its
family name in memory.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tests\Test-Devenv.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\tests\Test-Devenv.ps1 -Network
powershell -NoProfile -ExecutionPolicy Bypass -File .\tests\Test-Credentials.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\tests\Test-MuxConfiguration.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\tests\Test-TerminalProfile.ps1
pwsh -NoProfile -File .\tests\Test-MuxShortcuts.ps1
```

The credential suite uses fake vault entries and SSH key fixtures. It checks
Windows ACLs, idempotence, preservation of existing credentials/settings,
rejection of Git/junction destinations, native stdin/argument handling, and
the Docker helper path. It never logs in or changes real credentials. Browser
login, UAC/service changes and real registry authentication require an
interactive run on the target machine.

The native integration suite uses real Windows Vim and psmux binaries. It edits
a Windows file, starts interactive Vim in an unnamed buffer, loads the shared
vimrc, verifies the actual Gruvbox theme, true color, clipboard support and shell
integration, and exercises prefix/keybindings, panes, windows and a detached
session. `-InstallPlugins` runs a real install through ConPTY in an isolated
Windows home, including CoC extensions. Use installed tools or portable builds:

```powershell
.\tests\Test-NativeTools.ps1 -VimExe C:\tools\vim\vim.exe `
  -PsmuxExe C:\tools\psmux\psmux.exe -RcRoot C:\dev\rc_files

.\tests\Test-NativeTools.ps1 -VimExe C:\tools\vim\vim.exe `
  -PsmuxExe C:\tools\psmux\psmux.exe -RcRoot C:\dev\rc_files `
  -PythonRoot C:\Python314 -NodeDirectory C:\tools\node -InstallPlugins
```

Add `-ClangdExe C:\tools\clangd\bin\clangd.exe` to validate native clangd
availability and request semantic tokens from a real Windows clangd process.

The test's psmux server uses a unique `-L` namespace. It kills only that server.
Full-plugin fixtures are kept for diagnosis; ordinary fixtures are removed
after validating the cleanup path. A clean Windows VM is still needed to test
the complete WinGet/UAC/installer path, reboot behavior, and visual rendering.

### Empty-folder deployment and rerun

`Test-Deployment.ps1` runs the actual bootstrap twice under native ConPTY with a
new, absent Windows home, then runs `-Check`, the native integration suite,
PowerShell profile/prompt checks and a real ACP initialization handshake. It
Git-clones a candidate rc_files snapshot and a notes fixture containing your
actual plugin/theme choices with fake plugin settings. Every installed plugin
comes from its current upstream publisher. It compares hashes and write times
of managed files, plugin code/settings and plugin Git heads across both runs,
checks skip messages, and rejects extra backups on the second run.
Successful runs leave `verification.json`, first/second transcripts and file
snapshots beside the isolated home. Runtime checks use their own ConPTY sessions;
the Vim driver acknowledges a first-start Enter prompt and still rejects startup
errors recorded in the runtime log. The default test root is deliberately short
to stay within Windows' executable-path limit for nested npm dependencies.

Supply real portable builds in a tools folder: `vim/**/vim.exe`,
`psmux/psmux.exe`, `node/**/node.exe`, `python/python3.dll`,
`clangd/**/clangd.exe`, `gh/**/gh.exe`, `pwsh/pwsh.exe` and
`prompt/oh-my-posh.exe`. Existing host Git/WinGet/Terminal/Codex are used for
inventory. All writable app, Git and npm locations are redirected to the fixture.

```powershell
.\tests\Test-Deployment.ps1 -ToolsRoot C:\test-tools `
  -RcCandidate C:\dev\rc_files -NotesConfiguration C:\notes
```

This is a real application/configuration deployment test. Host WinGet package
installation, WSL, font registration and credentials are explicitly skipped;
an empty folder does not isolate Windows installers, the registry or services.
`-SkipPackages`, `-PythonRoot`, `-DocumentsPath` and `-EnvironmentPath` are also
available on bootstrap for this purpose. Authentication/SSH permission tests
run separately with fake credentials. A disposable clean Windows VM remains
necessary to validate first-time machine installers, UAC, reboot, actual sign-in,
registry authentication and device pairing. No clean-VM result is implied by
the folder test.

## References

* [psmux native Windows tutorial](https://github.com/psmux/psmux/blob/master/docs/tutorials/getting-started-windows.md)
* [Vim Windows documentation](https://vimhelp.org/os_win32.txt.html)
* [WinGet](https://learn.microsoft.com/windows/package-manager/winget/)
* [WSL installation](https://learn.microsoft.com/windows/wsl/install)
* [MonacoLigaturized Nerd Font latest release](https://github.com/thep0y/monaco-nerd-font/releases/latest)
* [Codex CLI](https://learn.chatgpt.com/docs/codex/cli)
* [Codex authentication](https://learn.chatgpt.com/docs/auth)
* [GitHub CLI authentication](https://cli.github.com/manual/gh_auth_login)
* [Docker registry login](https://docs.docker.com/reference/cli/docker/login/)
* [Windows SSH key management](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_keymanagement)
