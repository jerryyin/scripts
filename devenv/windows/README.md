# Native Windows development environment

A repeatable setup for native PowerShell, terminal Vim, psmux, Windows Terminal, personal configuration, and notes in Obsidian. Setup also installs WSL and the selected Ubuntu distribution when missing; Linux provisioning remains a separate `docker/env/min.sh` run inside Ubuntu.

## First machine setup

Use a Windows account with WinGet, network access to the package feeds and GitHub, and administrator approval for machine changes. Close Obsidian before applying setup because its plugin files may be updated.

Install Git if needed and clone this repository:

```powershell
winget install --id Git.Git --exact --source winget
# Reopen PowerShell if git is not on PATH yet.
git clone https://github.com/jerryyin/scripts.git "$env:USERPROFILE\scripts"
powershell -NoProfile -ExecutionPolicy Bypass -File "$env:USERPROFILE\scripts\devenv\windows\bootstrap.ps1" -Apply
```

The execution policy flag applies to this process. Setup enables Developer Mode for symbolic links, requesting administrator approval when needed; ordinary configuration and credential operations run as the current user. Conflicting local settings require review rather than a copy-based fallback.

Neither `rc_files`, notes nor the native tools need to exist beforehand. Repository URLs, package choices, WSL selection and the font download source are owned by [defaults/environment.json](defaults/environment.json). Existing checkouts are reused without pulling, resetting or discarding local work.

## Plan, apply and check

From this directory:

```powershell
.\bootstrap.ps1                              # show the plan
.\bootstrap.ps1 -Apply                       # install, configure or repair
.\bootstrap.ps1 -Check                       # report missing tools and configuration drift
.\bootstrap.ps1 -Apply -Update               # refresh installed plugins and adapters
.\bootstrap.ps1 -Apply -SkipWsl              # opt out of WSL installation
.\bootstrap.ps1 -Apply -NotesPath C:\notes -RcPath C:\dev\rc_files
```

Use `Get-Help .\bootstrap.ps1 -Full` and the script's parameter block for the complete interface, including redirected locations and optional setup phases. `-Check` checks installed tools, configuration links, font availability and required plugin files. It does not establish successful GUI interaction, authentication or model instruction loading.

Setup installs missing Windows packages, prepares native credentials, clones missing configuration and notes repositories, deploys settings through `rc_files/.setup`, and prepares fonts, Vim, psmux and Obsidian integration. The native package set includes everyday tools such as PowerToys, KDE Connect, Rufus, WizTree and VLC; maintain the actual choices in the defaults file.

## Configuration ownership

- **`scripts`** owns Windows installation, prerequisite discovery and orchestration. The bootstrap and credential entry points are [bootstrap.ps1](bootstrap.ps1) and [priv.ps1](priv.ps1).
- **`rc_files`** owns application settings, shared agent guidance and its self-contained Python setup. Platform subdirectories select native settings. See its [setup guide](https://github.com/jerryyin/rc_files/blob/master/.setup/README.md), [Vim guide](https://github.com/jerryyin/rc_files/blob/master/.setup/vim.md) and [agent guide](https://github.com/jerryyin/rc_files/blob/master/.setup/agents.md).
- **The notes repository** owns `.obsidian/community-plugins.json`, appearance choices and authored plugin settings. Downloads follow those choices; plugin code remains application runtime data.
- **Native credential tools and private storage** own tokens, SSH keys and registry credentials. These stay outside configuration repositories and general setup backups.
- **`%LOCALAPPDATA%\PersonalDevenv`** holds setup backups and local acceptance artifacts. It is not a configuration inventory.

Installed settings link directly to their tracked sources. Edits through those links, including Terminal and PowerToys preferences, reach `rc_files`; review and commit them there. New ignored authored files still require explicit `git add -f`. Application runtime directories remain real directories, with individual settings links or per-skill directory links as appropriate.

Codex's local configuration is seeded from its tracked template only when missing. Existing local preferences are preserved; changing the template does not update an existing local file. Agent instructions and personal skills are deployed from the shared collection. Application-managed skills and downloaded plugin code retain their application ownership.

## Native shell and editor

Windows Terminal opens one visible PowerShell 7 profile using the tracked Gruvbox palette and MonacoLigaturized Nerd Font. The PowerShell profile provides the prompt and familiar command aliases, with native executables found on `PATH`. Commands such as `ls` and `cat` are PowerShell aliases; enter `wsl` to use the separate Linux environment.

`vim` opens terminal Vim using the shared `~/.vimrc`. The bootstrap installs one terminal Vim shortcut and removes standard extra shortcuts from that installation while preserving custom shortcuts. Native clangd supplies C/C++ language services; project build databases and toolchains must match Windows. See the Vim guide for mappings, plugin ownership and maintenance.

`tmux` invokes native psmux, and `dev` attaches or creates its main session. Shared tmux/psmux preferences load directly from linked native configuration, including the backtick prefix and pane navigation. Linux tmux and Windows psmux are separate applications; their native plugin managers and runtime integrations remain distinct.

After changing configuration, reload or restart the owning application. PowerShell reads its profile in a new shell. Reload tmux/psmux configuration directly; no generated translation needs refreshing. Font and color preferences stay in the application's tracked settings.

## WSL and credentials

WSL setup installs the selected distribution without launching it or rebooting automatically. If Windows needs a restart or distribution initialization, bootstrap completes native setup, reports the remaining step and exits with code `3010`. Restart when required, launch Ubuntu to create the Linux account, then rerun `-Apply`. Run `docker/env/min.sh` separately inside Linux. Existing Linux configuration is left to that installer.

The credential phase runs before cloning private notes. It preserves working GitHub and Codex logins, clones the configured private vault when missing, restores the selected Docker registry credentials, and prepares native SSH permissions and agent use. Docker Desktop is not installed by this phase.

```powershell
.\priv.ps1                                  # plan
.\priv.ps1 -Status                          # inspect current credentials
.\priv.ps1 -Apply                           # perform native sign-in and setup
.\priv.ps1 -Apply -SshSource E:\backup\.ssh   # import missing keys from a trusted source
```

Existing SSH keys are preserved. Supply a trusted backup or create and register a key yourself if none exists; setup does not silently replace your identity. Vault updates are explicit Git operations. Browser sign-in, encrypted-key passphrases and service approval may require interaction on the target machine.

## Maintenance and first-use steps

Pull the wanted configuration, notes or scripts checkout yourself, review local edits, and rerun setup. Correct links and completed installations are reused. Missing tools and incomplete plugin installations are repaired. Fresh package installs use the available upstream version; existing Windows packages are retained unless a required prerequisite upgrade is needed.

Ordinary reruns retain installed Vim, mux and Obsidian plugins. `-Apply -Update` requests their refresh along with themes and the Codex ACP adapter. CoC checks the latest tracked and existing extension choices on every setup, preserving local additions. Obsidian updates preserve authored plugin settings; no downloaded plugin code or release hashes belong in the defaults file.

Open the registered notes vault in Obsidian and select Codex Auto-detect in Copilot if the adapter has not been selected on that device. Restart PowerToys to load Awake preferences and pair KDE Connect with your devices. Native sign-in and device pairing remain first-use interactions.

## Verification

See [tests/README.md](tests/README.md) for focused configuration, credentials, keyboard and application checks, plus deployment into an empty redirected home. These checks have different prerequisites and scopes. A disposable clean Windows VM is required to accept first-time installers, UAC, WSL, reboot behavior and real account sign-in.
