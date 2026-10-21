# Native Windows development environment

A repeatable setup for native PowerShell, terminal Vim, psmux, Windows Terminal, personal configuration, and notes in Obsidian. Setup also installs WSL and the selected Ubuntu distribution when missing; Linux provisioning remains a separate `setup/linux/bootstrap.sh` run inside Ubuntu.

## First machine setup

Use a Windows account with WinGet, network access to the package feeds and GitHub, and administrator approval for machine changes. Close Obsidian before applying setup because its plugin files may be updated.

Install Git if needed and clone this repository:

```powershell
winget install --id Git.Git --exact --source winget
# Reopen PowerShell if git is not on PATH yet.
git clone https://github.com/jerryyin/scripts.git "$env:USERPROFILE\scripts"
powershell -NoProfile -ExecutionPolicy Bypass -File "$env:USERPROFILE\scripts\setup\windows\bootstrap.ps1" -Apply
```

The execution policy flag applies to this process. Setup groups machine changes under one administrator approval and requests none when no administrator work is needed. For an existing SSH server, that worker also prepares rc_files prerequisites using their declared installation scopes and creates or repairs configuration links so SSH can read them. Credentials, fonts, plugins and application setup stay in the caller. Approve UAC with the same Windows account that started setup; a different administrator account is rejected before machine changes. Conflicting local settings require review rather than a copy-based fallback.

Neither `rc_files`, notes nor the native tools need to exist beforehand. Repository URLs, machine utilities and first-time WSL selection are owned by [defaults/environment.json](defaults/environment.json). Configured application requirements and the terminal font belong to their packages in `rc_files`. Existing checkouts are reused without pulling, resetting or discarding local work.

## Plan, apply and check

From this directory:

```powershell
.\bootstrap.ps1                              # show the plan
.\bootstrap.ps1 -Apply                       # install, configure or repair
.\bootstrap.ps1 -Check                       # report missing tools and configuration drift
.\bootstrap.ps1 -Apply -Update               # refresh installed plugins and adapters
.\bootstrap.ps1 -Apply -SkipWsl              # opt out of WSL installation
.\bootstrap.ps1 -Apply -SkipSsh              # leave SSH server configuration alone
.\bootstrap.ps1 -Apply -NotesPath C:\notes -RcPath C:\dev\rc_files
```

Use `Get-Help .\bootstrap.ps1 -Full` and the script's parameter block for the complete interface, including redirected locations and optional setup phases. `-Check` checks installed tools, configuration links, font availability and required plugin files. It does not establish successful GUI interaction, authentication or model instruction loading.

Setup installs missing Windows packages, prepares native credentials, clones missing configuration and notes repositories, deploys settings through `rc_files/.setup`, and prepares fonts, Vim, psmux and Obsidian integration. The native package set includes everyday tools such as PowerToys, KDE Connect, Rufus, WizTree and VLC; maintain the actual choices in the defaults file.

Packages marked `"scope": "machine"` run in the grouped administrator step. Other packages install with `--scope user`; Python's launcher is also explicitly installed for the current user. Existing installations are retained regardless of scope. The machine helper logs to `%LOCALAPPDATA%\PersonalDevenv\logs`; its temporary request and result files contain no credentials and are removed after use. Cancelling UAC or a machine-install failure stops setup without retrying elevation.

## Configuration ownership

- **`scripts`** owns Windows installation, prerequisite discovery and orchestration. The bootstrap and credential entry points are [bootstrap.ps1](bootstrap.ps1) and [priv.ps1](priv.ps1).
- **`rc_files`** owns application settings, shared agent guidance and its self-contained Python setup. Platform subdirectories select native settings. See its [setup guide](https://github.com/jerryyin/rc_files/blob/master/.setup/README.md), [Vim guide](https://github.com/jerryyin/rc_files/blob/master/vim/README.md) and [agent guide](https://github.com/jerryyin/rc_files/blob/master/agents/README.md).
- **The notes repository** owns `.obsidian/community-plugins.json`, appearance choices and authored plugin settings. Downloads follow those choices; plugin code remains application runtime data.
- **Native credential tools and private storage** own tokens, SSH keys and registry credentials. These stay outside configuration repositories and general setup backups.
- **`%LOCALAPPDATA%\PersonalDevenv`** holds setup backups and local acceptance artifacts. It is not a configuration inventory.

Within this directory, `Devenv.psm1` provides shared setup helpers; `Credentials.psm1` owns private files and SSH key permissions; `SshSession.psm1` owns SSH shell readiness and blocked-link recovery; and `Obsidian.psm1` installs the plugins and theme selected by the notes repository. `machine.ps1` is the single administrator worker. The entry scripts coordinate these modules.

Installed settings link directly to their tracked sources. Edits through those links, including Terminal preferences, reach `rc_files`; review and commit them there. New ignored authored files still require explicit `git add -f`. Application runtime directories remain real directories, with individual settings links or per-skill directory links as appropriate. Bootstrap installs PowerToys; PowerToys manages its own settings locally.

Claude's credential-bearing settings are generated privately by its `rc_files` package during credential setup. Its preferences come from a tracked template; the rendered file remains local. The installed Claude command runs normally. Rerun credential setup after replacing its vault key, or use `py -3 "$HOME/rc_files/.setup" --package claude`.

Codex's local configuration is seeded from its tracked template only when missing. Existing local preferences are preserved; changing the template does not update an existing local file. Agent instructions and personal skills are deployed from the shared collection. Application-managed skills and downloaded plugin code retain their application ownership.

## Native shell and editor

Windows Terminal opens one visible PowerShell 7 profile using the tracked Gruvbox palette and MonacoLigaturized Nerd Font. The PowerShell profile provides the prompt and familiar command aliases, with native executables found on `PATH`. Commands such as `ls` and `cat` are PowerShell aliases. Choose the single **Ubuntu** profile or enter `wsl` to open the default Linux distribution. That profile uses `wsl.exe --cd ~`; generated WSL profile sources are disabled to prevent duplicate entries. Setup reuses any registered distribution and installs the manifest's selection only when none exists.

`vim` opens terminal Vim using the shared `~/.vimrc`. The `rc_files` Vim package installs one terminal Vim shortcut and removes standard extra shortcuts from that installation while preserving custom shortcuts. Native clangd supplies C/C++ language services; project build databases and toolchains must match Windows. See the Vim guide for mappings, plugin ownership and maintenance.

Bare `psmux` or `tmux` resumes the most recently active native session, creating `main` when none exists. `dev` always attaches or creates `main`; explicit commands such as `tmux new-session -s work` keep their normal behavior. Closing the terminal leaves the session running; reopen PowerShell and run `tmux` to return. Backtick then `d` detaches without closing the terminal. Exiting the last shell ends that session. Shared tmux/psmux preferences load directly from linked native configuration, including the backtick prefix and pane navigation. Linux tmux and Windows psmux are separate applications; their native plugin managers and runtime integrations remain distinct.

After changing configuration, reload or restart the owning application. PowerShell reads its profile in a new shell. Reload tmux/psmux configuration directly; no generated translation needs refreshing. Font and color preferences stay in the application's tracked settings.

## Windows SSH sessions

`bootstrap.ps1 -Apply` is the setup and recovery command. When OpenSSH Server is installed, its grouped administrator step selects PowerShell, repairs matching blocked links, prepares missing native prerequisites, and deploys configuration links with administrator trust metadata. It preserves the original links in the setup backups. The normal user phase then applies credentials and application setup; it reuses the prepared links. SSH key/configuration permissions are reapplied by the credential phase without replacing existing keys or file contents.

`bootstrap.ps1 -Check` checks link traversal in an isolated process with RedirectionGuard enabled where supported, so a desktop check can detect the restriction that SSH applies. It does not change the caller's security settings. Rerun `-Apply` after installation or manual changes cause drift; no separate repair script is needed. Setup does not install or enable SSH Server, change authentication, or restart it. `-SkipSsh` excludes SSH shell and trusted-link provisioning; `-SkipCredentials` excludes key permission setup.

The rc_files PowerShell profile automatically attaches interactive SSH terminals to psmux `main`; command-only SSH requests and nested mux shells stay unaffected. Reconnect after changing the default shell. Client-side fonts and terminal palette remain the SSH client's settings; the prompt and mux status colors come from rc_files.

## WSL and credentials

Any registered WSL distribution satisfies the Windows setup requirement, regardless of its name or Linux flavor. Terminal uses WSL's configured default. The manifest's distribution choice controls first-time installation when no distribution is registered; setup preserves existing distributions and the selected default.

WSL setup installs the selected distribution without launching it or rebooting automatically. If Windows needs a restart or distribution initialization, bootstrap completes native setup, reports the remaining step and exits with code `3010`. Restart when required, launch Ubuntu to create the Linux account, then rerun `-Apply`. Run `setup/linux/bootstrap.sh` separately inside Linux. Existing Linux configuration is left to that installer.

The credential phase runs before cloning private notes. It preserves working GitHub and Codex logins, clones the configured private vault when missing, invokes `rc_files/.setup credentials --vault PATH` for application credentials, and prepares native SSH permissions and agent use. Application rendering belongs to rc_files; the credential phase installs nothing. Docker Desktop is not installed by this phase.

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

Open the registered notes vault in Obsidian and select Codex Auto-detect in Copilot if the adapter has not been selected on that device. Pair KDE Connect with your devices. Native sign-in and device pairing remain first-use interactions.

## Verification

See [tests/README.md](tests/README.md) for focused configuration, credentials, keyboard and application checks, plus deployment into an empty redirected home. These checks have different prerequisites and scopes. A disposable clean Windows VM is required to accept first-time installers, UAC, WSL, reboot behavior and real account sign-in.

## Application ownership

`rc_files` owns the native requirements and setup of Vim, Node, clangd, tmux/psmux, shells, GitHub CLI, Codex and Claude, plus the terminal font. This bootstrap prepares Git/Python, Developer Mode, WSL, credentials, notes/Obsidian and everyday utilities. It asks rc_files for missing machine requirements and includes their installation in its grouped administrator step. It then invokes full rc_files setup as the current user. `-SkipPackages` requires tools supplied on PATH and passes `--no-install`; redirected acceptance runs do not change host package installation, fonts or account settings.
