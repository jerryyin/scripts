#!/bin/bash
# Run only INSIDE a disposable image built with rc_files/base.dockerfile.
set -euo pipefail
[[ -f /.dockerenv && "$HOME" == /root && ! -e /zyin ]] || {
    echo 'Run this check in a disposable root Docker container without host-home mounts.' >&2
    exit 2
}
export PATH="$HOME/.local/bin:$PATH"

test -f "$HOME/scripts/setup/linux/setup.sh"
test -f "$HOME/scripts/setup/linux/packages/aiter.sh"
test -f "$HOME/scripts/containers/docker/pull-base-images.sh"
test ! -e "$HOME/scripts/devenv"
test ! -e "$HOME/scripts/docker"
test ! -e "$HOME/scripts/tools"
echo 'PASS canonical module layout without forwarding entry points'

for binary in git zsh vim nvim tmux node npm gh claude codex gdb ag python3; do
    command -v "$binary" >/dev/null
done
node --version
npm --version
gh --version
"$HOME/.local/bin/codex" --version
claude --version
python3 "$HOME/rc_files/.setup" check
python3 -m pip --version
python3 - <<'PY'
from pathlib import Path
import sys
plugin = Path.home()/'.vim/plugged/LeaderF/autoload/leaderf'
sys.path.insert(0, str(plugin/'python'))
import fuzzyMatchC
assert fuzzyMatchC.__file__.endswith('.so'), fuzzyMatchC.__file__
PY
RC_SETUP_UPDATE=0 GIT_CONFIG_GLOBAL=/dev/null vim -n -E -s -u NONE \
    -S "$HOME/rc_files/vim/.setup/vim-plugins.vim"
test -f "$HOME/.tmux/plugins/tpm/tpm"
test ! -e "$HOME/.ssh/id_rsa"
test ! -e "$HOME/.ssh/id_ed25519"
test ! -e "$HOME/.codex/auth.json"
test ! -e "$HOME/vault"
test "$(readlink -f "$(getent passwd root | cut -d: -f7)")" = "$(readlink -f "$(command -v zsh)")"
echo 'PASS installed tools, dotfiles, editor plugins, login shell, and absence of baked-in credentials'

# All credentials below are generated test data. Network should be disabled by
# the caller, so activation can only use the fixture storage.
mkdir -p /zyin/.ssh /zyin/.codex /zyin/vault
ssh-keygen -q -t ed25519 -N '' -f /zyin/.ssh/id_ed25519
printf '{"test_only":true}\n' > /zyin/.codex/auth.json
printf 'dummy-claude\n' > /zyin/vault/claude_key.txt
printf 'test:dummy-docker' | base64 > /zyin/vault/docker_mkmhub_auth.txt
printf 'dummy-github-personal\n' > /zyin/vault/gh_token_jerryyin.txt
printf 'dummy-github-work\n' > /zyin/vault/gh_token_amdeng.txt
printf 'dummy-artifactory\n' > /zyin/vault/atlartifactory_token.txt
printf '{"projects":{"fixture":{}},"machineID":"fixture"}\n' > "$HOME/.claude.json"
sha256sum "$HOME/.claude.json" > /tmp/claude-state-checksum
export DEVENV_PERSISTENT_ROOT=/zyin
bash "$HOME/scripts/setup/linux/credentials/activate.sh"
test "$(readlink "$HOME/vault")" = /zyin/vault
test "$(readlink "$HOME/.codex")" = /zyin/.codex
cmp "$HOME/.ssh/id_ed25519" /zyin/.ssh/id_ed25519
test "$(stat -c %a "$HOME/.ssh/id_ed25519")" = 600
test "$(stat -c %a "$HOME/.ssh/id_ed25519.pub")" = 644
grep -q dummy-claude "$HOME/.claude/settings.json"
test ! -L "$HOME/.claude/settings.json"
test "$(stat -c %a "$HOME/.claude/settings.json")" = 600
sha256sum --check /tmp/claude-state-checksum
grep -q dummy-github-personal "$HOME/.config/gh/hosts.yml"
grep -q dummy-github-work "$HOME/.config/gh/hosts.yml"
grep -q dummy-artifactory "$HOME/.netrc"
cmp "$HOME/.gist" /zyin/vault/gh_token_jerryyin.txt
python3 - <<'PY'
import base64, json
from pathlib import Path
config = json.loads((Path.home()/'.docker/config.json').read_text())
assert base64.b64decode(config['auths']['mkmhub.amd.com']['auth']) == b'test:dummy-docker'
PY
echo 'PASS real activation: SSH, vault, OAuth symlinks, and all credential templates'

sha256sum "$HOME/.claude.json" "$HOME/.claude/settings.json" "$HOME/.config/gh/hosts.yml" "$HOME/.netrc" \
    "$HOME/.gist" "$HOME/.docker/config.json" > /tmp/credential-checksums
bash "$HOME/scripts/setup/linux/credentials/activate.sh"
sha256sum --check /tmp/credential-checksums
bash "$HOME/scripts/setup/linux/credentials/sync.sh" --status
echo 'PASS repeated canonical activation and credential status preserve configuration'

# Rotation and explicit opt-out must also work through the relocated helpers.
printf 'dummy-github-rotated\n' > /zyin/vault/gh_token_jerryyin.txt
printf 'dummy-claude-rotated\n' > /zyin/vault/claude_key.txt
bash "$HOME/scripts/setup/linux/credentials/activate.sh"
grep -q dummy-github-rotated "$HOME/.config/gh/hosts.yml"
grep -q dummy-claude-rotated "$HOME/.claude/settings.json"
sha256sum --check /tmp/claude-state-checksum
cmp "$HOME/.gist" /zyin/vault/gh_token_jerryyin.txt
DEVENV_PERSISTENT_ROOT='' bash "$HOME/scripts/setup/linux/credentials/sync.sh"
echo 'PASS credential rotation and persistence opt-out'

# Execute the real Compose startup command, then feed Zsh a command over stdin.
python3 - <<'PY'
import json, subprocess
from pathlib import Path
command = json.loads(Path('/validation-startup.json').read_text())
result = subprocess.run(command, input="print -r -- COMPOSE_ZSH_OK\nexit\n", text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
assert result.returncode == 0, result.stdout
assert 'COMPOSE_ZSH_OK' in result.stdout, result.stdout
print('PASS actual Compose command activates credentials and starts Zsh')
PY
