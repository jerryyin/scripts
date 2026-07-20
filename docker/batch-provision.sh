#!/bin/bash
# batch-provision.sh - Get freshly-reserved Conductor SUTs ready to work on
#
# Conductor (https://conductor.amd.com) hands out bare-metal machines from a
# pool; whichever one you land on next has none of your dotfiles, packages,
# or credentials. This script makes any target host as ready as your daily
# driver in one pass:
#
#   1. Seed a GitHub/vault-capable SSH keypair on the target, copied from a
#      known-good source host (default: smci355).
#   2. Clone rc_files + scripts on the target over SSH (using that seeded
#      key) -- or pull latest if they're already there from a prior run.
#   3. Run env/min.sh: installs base packages, stows rc_files (dotfiles),
#      installs the AMD CA cert, and installs the Claude/Codex CLIs.
#   4. Run env/priv.sh --force: SSH key + vault bootstrap, then
#      credential/config runtime patches (vault.sh claude/docker).
#
# This assumes you can ALREADY `ssh user@target` once (i.e. your own key is
# already registered in Conductor's key management for that reservation) --
# this script seeds a git-capable key on the target AFTER that first login,
# it does not bootstrap initial SSH access itself.
#
# Usage:
#   batch-provision.sh [-k SOURCE_KEY_HOST] HOST [HOST ...]
#   batch-provision.sh [-k SOURCE_KEY_HOST] -f hosts_file
#
#   HOST accepts a bare hostname/FQDN (uses $USER@HOST) or an explicit
#   user@host. ~/.ssh/config aliases work too.
#
# Examples:
#   batch-provision.sh zhuoryin@banff-cyxtera-s74-5.ctr.dcgpu zhuoryin@cv350-tnndh2-a08-1.tnn.dcgpu
#   batch-provision.sh -f my-hosts.txt

set -uo pipefail

SOURCE_KEY_HOST="smci355"
HOSTS=()
HOSTS_FILE=""
RC_FILES_REPO="git@github.com:jerryyin/rc_files.git"
SCRIPTS_REPO="git@github.com:jerryyin/scripts.git"

usage() {
    cat <<'EOF'
Usage: batch-provision.sh [-k SOURCE_KEY_HOST] HOST [HOST ...]
       batch-provision.sh [-k SOURCE_KEY_HOST] -f hosts_file

  -k HOST   Source machine to copy a working SSH keypair from (default: smci355)
  -f FILE   Read target hosts from FILE (one per line, '#' comments allowed)

HOST accepts bare hostnames/FQDNs (uses $USER@HOST) or explicit user@host.
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -k)
            SOURCE_KEY_HOST="$2"
            shift 2
            ;;
        -f)
            HOSTS_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            HOSTS+=("$1")
            shift
            ;;
    esac
done

if [[ -n "$HOSTS_FILE" ]]; then
    while IFS= read -r line; do
        line="${line%%#*}"
        line="$(echo "$line" | xargs 2>/dev/null || true)"
        [[ -n "$line" ]] && HOSTS+=("$line")
    done < "$HOSTS_FILE"
fi

if [[ ${#HOSTS[@]} -eq 0 ]]; then
    usage
fi

SSH_OPTS=(-o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o BatchMode=yes)

echo "🔑 Reading SSH keypair from $SOURCE_KEY_HOST..."
KEY_NAME=""
for candidate in id_rsa id_ed25519; do
    if ssh "${SSH_OPTS[@]}" "$SOURCE_KEY_HOST" "test -f ~/.ssh/$candidate" 2>/dev/null; then
        KEY_NAME="$candidate"
        break
    fi
done
if [[ -z "$KEY_NAME" ]]; then
    echo "❌ No id_rsa/id_ed25519 found on $SOURCE_KEY_HOST" >&2
    exit 1
fi

TMP_KEY_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_KEY_DIR"' EXIT
if ! scp "${SSH_OPTS[@]}" "$SOURCE_KEY_HOST:~/.ssh/$KEY_NAME" "$SOURCE_KEY_HOST:~/.ssh/$KEY_NAME.pub" "$TMP_KEY_DIR/" >/dev/null; then
    echo "❌ Failed to copy $KEY_NAME(.pub) from $SOURCE_KEY_HOST" >&2
    exit 1
fi
chmod 600 "$TMP_KEY_DIR/$KEY_NAME"
echo "✓ Got $KEY_NAME from $SOURCE_KEY_HOST"

FAILED=()
OK=()

for host in "${HOSTS[@]}"; do
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Provisioning: $host"
    echo "════════════════════════════════════════════════════════════════"

    if ! ssh "${SSH_OPTS[@]}" "$host" "exit 0" 2>/dev/null; then
        echo "❌ Cannot SSH to $host — skipping (check Conductor reservation/key)"
        FAILED+=("$host")
        continue
    fi

    echo "📤 Seeding SSH keypair ($KEY_NAME)..."
    ssh "${SSH_OPTS[@]}" "$host" '
        set -e
        mkdir -p ~/.ssh && chmod 700 ~/.ssh
        for f in "'"$KEY_NAME"'" "'"$KEY_NAME"'.pub"; do
            if [ -f "$HOME/.ssh/$f" ] && [ ! -L "$HOME/.ssh/$f" ]; then
                mv "$HOME/.ssh/$f" "$HOME/.ssh/$f.bak.$(date +%s)"
            fi
        done
    '
    if ! scp "${SSH_OPTS[@]}" "$TMP_KEY_DIR/$KEY_NAME" "$TMP_KEY_DIR/$KEY_NAME.pub" "$host:~/.ssh/" >/dev/null; then
        echo "❌ Failed to copy keypair to $host"
        FAILED+=("$host")
        continue
    fi
    ssh "${SSH_OPTS[@]}" "$host" "chmod 600 ~/.ssh/$KEY_NAME && chmod 644 ~/.ssh/$KEY_NAME.pub"
    echo "✓ Keypair in place on $host"

    # rc_files' own ~/.ssh/config (which sets StrictHostKeyChecking accept-new
    # for github.com) doesn't exist on the host until rc_files itself is
    # cloned -- chicken-and-egg -- so this bootstrap clone needs its own
    # accept-new override. Every git/ssh call after this point (including
    # priv.sh's vault clone) picks up accept-new for free once rc_files/install.sh
    # stows that config below.
    echo "📥 Cloning/updating rc_files + scripts on $host (via $KEY_NAME)..."
    if ! ssh "${SSH_OPTS[@]}" "$host" '
        set -e
        export GIT_SSH_COMMAND="ssh -i $HOME/.ssh/'"$KEY_NAME"' -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
        if [ -d "$HOME/rc_files/.git" ]; then
            echo "  rc_files already present, pulling latest..."
            git -C "$HOME/rc_files" pull --ff-only
        else
            echo "  cloning rc_files..."
            git clone "'"$RC_FILES_REPO"'" "$HOME/rc_files"
        fi
        if [ -d "$HOME/scripts/.git" ]; then
            echo "  scripts already present, pulling latest..."
            git -C "$HOME/scripts" pull --ff-only
        else
            echo "  cloning scripts..."
            git clone "'"$SCRIPTS_REPO"'" "$HOME/scripts"
        fi
    '; then
        echo "❌ Failed to clone/update rc_files or scripts on $host"
        FAILED+=("$host")
        continue
    fi
    echo "✓ rc_files + scripts up to date on $host"

    echo "🚀 Running env/min.sh on $host (base packages, dotfiles, CLIs)..."
    echo "────────────────────────────────────────────────────────────────"
    if ! ssh -t "${SSH_OPTS[@]}" "$host" "cd ~ && bash ~/scripts/docker/env/min.sh"; then
        echo "────────────────────────────────────────────────────────────────"
        echo "❌ min.sh failed on $host"
        FAILED+=("$host")
        continue
    fi
    echo "────────────────────────────────────────────────────────────────"

    echo "🔧 Running env/priv.sh --force on $host (SSH/vault bootstrap + credential sync)..."
    if ! ssh -t "${SSH_OPTS[@]}" "$host" "bash ~/scripts/docker/env/priv.sh --force"; then
        echo "⚠️  priv.sh reported an issue on $host (continuing; re-run manually if needed)"
    fi

    echo "✅ $host ready"
    OK+=("$host")
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Summary"
echo "════════════════════════════════════════════════════════════════"
echo "  Ready:  ${#OK[@]}/${#HOSTS[@]}  ${OK[*]:-}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  Failed: ${#FAILED[@]}/${#HOSTS[@]}  ${FAILED[*]}"
fi
echo "════════════════════════════════════════════════════════════════"

[[ ${#FAILED[@]} -eq 0 ]]
