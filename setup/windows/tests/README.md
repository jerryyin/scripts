# Windows setup checks

Run these commands from `setup/windows`. Inspect each script's parameters before running it. Configuration tests use disposable directories; native application checks additionally require the installed tools they exercise. Developer Mode must already be enabled for tests that create symbolic links; tests do not enable it.

## Focused checks

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tests\Test-Devenv.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\tests\Test-MachineSetup.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\tests\Test-Credentials.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\tests\Test-SshSetup.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\tests\Test-Obsidian.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\tests\Test-TerminalProfile.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\tests\Test-ConfigurationOwnership.ps1
pwsh -NoProfile -File .\tests\Test-VimShortcuts.ps1
pwsh -NoProfile -File .\tests\Test-MuxShortcuts.ps1
pwsh -NoProfile -File .\tests\Test-MuxLifecycle.ps1
pwsh -NoProfile -File .\tests\Test-SshSession.ps1
```

`Test-Devenv.ps1` checks bootstrap behavior and simulated WSL outcomes without installing machine packages. Add `-Network` to exercise current upstream plugin and theme downloads. `Test-Credentials.ps1` uses fake credentials and key fixtures; it does not sign into real accounts.

`Test-MachineSetup.ps1` substitutes the OS process-launch boundary to check one grouped elevation, zero elevation for unchanged or already-elevated callers, cancellation, failure propagation, the credential boundary and request cleanup. It also checks machine Python and launcher registration. Real UAC and first-time package installations still require the disposable VM acceptance below.

`Test-SshSetup.ps1` substitutes service and registry access to check shell selection, missing shells and reruns, then exercises link inspection and recovery against disposable files. The real isolated traversal probe covers Unicode paths and file errors. When already elevated, the test also runs the actual administrator worker and rc_files deployer against a disposable PowerShell profile, checking trusted-link creation and an unchanged rerun. `Test-MachineSetup.ps1` also verifies that SSH-only and combined machine work use the same administrator worker. `Test-Credentials.ps1` exercises repeated credential setup with fake keys and deliberately excessive permissions.

`Test-Obsidian.ps1` checks plugin/theme discovery, updates, failed downloads and preservation of preferences with local fixtures. Add `-Network` for current publisher downloads. `Test-Devenv.ps1` includes these checks as well.

Terminal and ownership checks exercise source links and reruns. Vim and mux shortcut checks drive real applications in isolated console sessions and require their native tools, plugins and configuration to be installed. Supply `-RcRoot` to scripts exposing that parameter when the checkout is outside the default home location.

`Test-MuxLifecycle.ps1` closes and reopens real console clients with isolated psmux data and configuration. It checks that the same shell survives and that bare `psmux` and `tmux` resume an existing numbered session; explicit new-session commands still create a separate session.

`Test-SshSession.ps1` checks automatic attachment to `main`, persistence across terminal closure, and exclusion of command-only and redirected shells using isolated psmux sessions.

The platform-independent configuration setup tests live with the deployer in `rc_files/.setup/tests`; their command is documented in that repository's setup guide.

## Native application integration

```powershell
.\tests\Test-NativeTools.ps1 -VimExe C:\tools\vim\vim.exe `
  -PsmuxExe C:\tools\psmux\psmux.exe -RcRoot C:\dev\rc_files
```

Add `-NodeDirectory C:\tools\node -InstallPlugins` to install real Vim/CoC plugins in the fixture. Add `-ClangdExe C:\tools\clangd\bin\clangd.exe` for native language-server checks. The script uses its own psmux server namespace and preserves diagnostic fixtures when needed.

## Empty-home deployment

```powershell
.\tests\Test-Deployment.ps1 -ToolsRoot C:\test-tools `
  -RcCandidate C:\dev\rc_files -NotesConfiguration C:\notes
```

This runs the real bootstrap twice with redirected application homes, then checks configuration and native application behavior. The candidate checkout supplies tracked configuration; the notes checkout supplies actual plugin/theme choices with fake fixture settings. Current plugin code is downloaded from its publishers. The fixture records transcripts, file snapshots and `verification.json` for inspection.

The tools root must contain real portable builds at `vim/**/vim.exe`, `psmux/psmux.exe`, `node/**/node.exe`, `python/python3.dll`, `clangd/**/clangd.exe`, `gh/**/gh.exe` and `pwsh/pwsh.exe`. Host Git, WinGet, Terminal and Codex are also used for inventory. Read [Test-Deployment.ps1](Test-Deployment.ps1) for the current fixture inputs and [Test-NativeTools.ps1](Test-NativeTools.ps1) for native check options.

An empty home isolates configuration and application data. Machine package installers, font registration, WSL installation, UAC, reboot, real authentication and device pairing require separate acceptance on a disposable clean Windows VM. A folder-based result does not establish those machine-level outcomes.

Native application installer checks live with `rc_files`: `.setup/tests/Test-WindowsPackages.ps1` checks package scopes and grouped elevation, and `terminal/.setup/tests/Test-Font.ps1` checks current-release selection and archive handling. Font registration and actual WinGet installation remain clean-VM checks.
