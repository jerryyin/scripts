#!/bin/bash
# batch-provision.sh - Get freshly-reserved Conductor SUTs ready to work on
#
# Conductor (https://conductor.amd.com) hands out bare-metal machines from a
# pool; whichever one you land on next has none of your dotfiles, packages,
# or credentials. This script makes any target host as ready as your daily
# driver -- and it provisions every target HOST in parallel: each host's
# work is one call to provision_host(), backgrounded, so N hosts take about
# as long as one (instead of the orchestrator dictating each step to each
# host in turn). Per-host output is prefixed "[host] " and also captured to
# its own log file for later inspection.
#
# Per host, provision_host() does 3 round trips total:
#   1. ssh an inline one-liner that backs up ~/.ssh (whole dir, if it's
#      there -- most freshly-reserved Conductor hosts have nothing to back
#      up at all) and creates a fresh one. Also doubles as the reachability
#      check.
#   2. scp a GitHub/vault-capable SSH keypair in, copied from a known-good
#      source host (default: smci355). If that source host also has an
#      id_enterprise keypair (Enterprise GitHub identity, used as `github-e`
#      in ~/.ssh/config for private AMD-Triton repos etc.), it's seeded too.
#      Everything comes from the one source host, fetched once up front and
#      reused for every target, so all targets end up with a consistent,
#      uniform set of keys.
#   3. scp remote-provision-host.sh over and ssh-run it, which chmod's the
#      seeded keys, clones/updates rc_files + scripts, runs env/min.sh,
#      env/priv.sh --force, and pulls Docker base images -- see that script
#      for the full logic (including why the key backup above hands it an
#      HTTPS-only bootstrap-clone problem to solve).
#
#   remote-provision-host.sh is a real, standalone script, not a heredoc
#   string embedded in here, so it gets its own syntax highlighting/linting
#   and can be tested/run independently. Self-deletes off the target host
#   once it's run.
#
# Pulling base images (part of step 3) grabs the images the triton and triton-mi450
# services build FROM (BASE_IMAGE build arg in rc_files' docker-compose.yml)
# -- e.g. rocm/pytorch:latest and the mkmhub.amd.com gfx1250 image. This does
# NOT build the jeryin/dev:triton(-mi450) service images themselves (those
# are local-only tags built via `dbuild`/`docker compose build`, never
# pushed to a registry) -- just pre-pulls what they'd build FROM, which is
# the expensive, cacheable part. Requires env/priv.sh to have already
# patched ~/.docker/config.json with mkmhub credentials. Resolved via
# `docker compose config --format json` + python3's json module (real
# structured parsing, not YAML text-scraping) -- there's no `docker compose`
# subcommand that pulls a service's Dockerfile FROM image while skipping
# the build; `compose pull` only pulls a service's tagged `image:`, which
# for jeryin/dev:* is a local-only tag that's never been pushed anywhere.
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_PROVISION_SCRIPT="$SCRIPT_DIR/remote-provision-host.sh"

SOURCE_KEY_HOST="smci355"
HOSTS=()
HOSTS_FILE=""

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

if [[ ! -f "$REMOTE_PROVISION_SCRIPT" ]]; then
    echo "❌ Missing companion script: $REMOTE_PROVISION_SCRIPT" >&2
    exit 1
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

# KEY_NAMES holds base names (no .pub) of every keypair to seed on targets.
# KEY_NAME (the git-capable key) is always first/required; id_enterprise is
# appended too if the source host happens to have it.
KEY_NAMES=("$KEY_NAME")
if ssh "${SSH_OPTS[@]}" "$SOURCE_KEY_HOST" "test -f ~/.ssh/id_enterprise && test -f ~/.ssh/id_enterprise.pub" 2>/dev/null; then
    KEY_NAMES+=("id_enterprise")
else
    echo "⚠️  No id_enterprise(.pub) on $SOURCE_KEY_HOST — skipping enterprise-key seeding"
fi

TMP_KEY_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_KEY_DIR"' EXIT
SRC_FILES=()
for name in "${KEY_NAMES[@]}"; do
    SRC_FILES+=("$SOURCE_KEY_HOST:~/.ssh/$name" "$SOURCE_KEY_HOST:~/.ssh/$name.pub")
done
if ! scp "${SSH_OPTS[@]}" "${SRC_FILES[@]}" "$TMP_KEY_DIR/" >/dev/null; then
    echo "❌ Failed to copy ${KEY_NAMES[*]} from $SOURCE_KEY_HOST" >&2
    exit 1
fi
for name in "${KEY_NAMES[@]}"; do
    chmod 600 "$TMP_KEY_DIR/$name"
done
echo "✓ Got ${KEY_NAMES[*]} from $SOURCE_KEY_HOST"

SCP_FILES=()
for name in "${KEY_NAMES[@]}"; do
    SCP_FILES+=("$TMP_KEY_DIR/$name" "$TMP_KEY_DIR/$name.pub")
done

# Backs up ~/.ssh wholesale (not per-key-name) if it's there at all, then
# creates a fresh one. Simple enough to be an inline ssh command rather
# than its own file -- no arguments, nothing host-specific to interpolate.
# Uses a single fixed backup name (~/.ssh.bak), not a timestamped one: most
# provisioned machines start clean so this rarely even triggers, and a
# host that gets re-provisioned repeatedly should still only ever keep the
# one most-recent pre-provision snapshot, not pile up a new directory
# every run.
REMOTE_KEY_BACKUP_CMD='
if [ -d ~/.ssh ] && [ ! -L ~/.ssh ]; then
    rm -rf ~/.ssh.bak
    mv ~/.ssh ~/.ssh.bak
fi
mkdir -p ~/.ssh && chmod 700 ~/.ssh
'

# Provisions a single host end to end. Meant to be run backgrounded (see the
# launch loop below) so every host provisions concurrently; reads the
# SSH_OPTS/KEY_NAMES/SCP_FILES/REMOTE_KEY_BACKUP_CMD/REMOTE_PROVISION_SCRIPT
# globals set up above (shared, read-only from here on -- safe for
# parallel subshells to read).
#
# The heavier remote-side logic lives in a real, standalone,
# shellcheck-able file (remote-provision-host.sh) alongside this script,
# not as a heredoc string embedded in here -- it gets scp'd over and run
# with normal command-line arguments, same as running any other script.
# Self-deletes (via its own EXIT trap) once it's run.
provision_host() {
    local host="$1"

    echo "════════════════════════════════════════════════════════════════"
    echo "  Provisioning: $host"
    echo "════════════════════════════════════════════════════════════════"

    # Round trip 1: back up ~/.ssh if it's there, create a fresh one. Also
    # doubles as the reachability check -- if ssh can't even connect, this
    # fails fast.
    if ! ssh "${SSH_OPTS[@]}" "$host" "$REMOTE_KEY_BACKUP_CMD"; then
        echo "❌ Cannot reach/prepare $host — skipping (check Conductor reservation/key)"
        return 1
    fi

    # Round trip 2: the actual key copy (scp, not ssh -- key material has to
    # leave this machine, so it can't be folded into a remote-only script).
    if ! scp "${SSH_OPTS[@]}" "${SCP_FILES[@]}" "$host:~/.ssh/" >/dev/null; then
        echo "❌ Failed to copy keypair(s) to $host"
        return 1
    fi
    echo "✓ Keypair(s) copied to $host"

    # Round trip 3: ship remote-provision-host.sh over, then run it. It
    # chmod's the keys, clones/updates rc_files + scripts, runs env/min.sh,
    # env/priv.sh --force, and pulls base images -- see that script. -tt
    # forces a real remote pty for min.sh/priv.sh even though this whole
    # function's stdout is piped (not a terminal) on our end, since it's
    # backgrounded for parallel execution.
    if ! scp "${SSH_OPTS[@]}" "$REMOTE_PROVISION_SCRIPT" "$host:~/" >/dev/null; then
        echo "❌ Failed to copy remote-provision-host.sh to $host"
        return 1
    fi
    if ! ssh -tt "${SSH_OPTS[@]}" "$host" bash ~/remote-provision-host.sh "${KEY_NAMES[@]}"; then
        echo "❌ Provisioning steps failed on $host"
        return 1
    fi

    echo "✅ $host ready"
    return 0
}

LOG_DIR=$(mktemp -d)
echo ""
echo "🚀 Provisioning ${#HOSTS[@]} host(s) in parallel — full per-host logs: $LOG_DIR"
echo ""

declare -A PIDS
for host in "${HOSTS[@]}"; do
    logfile="$LOG_DIR/$(printf '%s' "$host" | tr -c 'A-Za-z0-9._-' '_').log"
    (
        # Each backgrounded subshell inherits the EXIT trap above and would
        # otherwise independently delete the shared TMP_KEY_DIR the moment
        # ITS work finishes -- while sibling hosts still mid-flight need it.
        # Only the main script's own exit should clean it up.
        trap - EXIT
        provision_host "$host" 2>&1 | sed -u "s/^/[$host] /" | tee -a "$logfile"
        exit "${PIPESTATUS[0]}"
    ) &
    PIDS["$host"]=$!
done

FAILED=()
OK=()
for host in "${HOSTS[@]}"; do
    if wait "${PIDS[$host]}"; then
        OK+=("$host")
    else
        FAILED+=("$host")
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Summary"
echo "════════════════════════════════════════════════════════════════"
echo "  Ready:  ${#OK[@]}/${#HOSTS[@]}  ${OK[*]:-}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  Failed: ${#FAILED[@]}/${#HOSTS[@]}  ${FAILED[*]}"
fi
echo "  Full per-host logs: $LOG_DIR"
echo "════════════════════════════════════════════════════════════════"

[[ ${#FAILED[@]} -eq 0 ]]
