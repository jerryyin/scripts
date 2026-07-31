#!/bin/bash
# remote-provision-host.sh - scp'd to and run ON the target host by
# batch-provision.sh, AFTER the SSH keypair(s) it needs have already been
# scp'd into ~/.ssh (see the inline key-backup step in batch-provision.sh,
# which runs before that).
#
# chmod's the seeded keypair(s), bootstrap-clones just enough of `scripts`
# to make env/min.sh invokable (min.sh itself clones rc_files, and
# re-clones/no-ops scripts -- see below for why we don't just let it),
# then runs env/min.sh and env/priv.sh --force, and pulls the base images
# the dev services build FROM.
#
# Every step here is idempotent and safe to re-run against an
# already-provisioned host: git clone-or-pull, min.sh/priv.sh are
# explicitly designed to be idempotent (see their own comments), and
# `docker pull` only fetches layers that actually changed.
#
# Self-deletes on exit (success or failure) since it's a transient helper.
#
# Usage: remote-provision-host.sh KEY_NAME [KEY_NAME ...]
#   each KEY_NAME is a base name under ~/.ssh with a matching KEY_NAME.pub
# Env: NO_BASE_PULL=1 skips the base-image pulls (batch-provision.sh -n)

set -e
trap 'rm -f -- "$0"' EXIT

for name in "$@"; do
    chmod 600 "$HOME/.ssh/$name"
    chmod 644 "$HOME/.ssh/$name.pub"
done
echo "✓ Keypair permissions set"

# Just enough to make `bash ~/scripts/docker/env/min.sh` invokable -- min.sh
# lives inside this clone, so something has to put it on disk first (min.sh
# itself will still handle rc_files, and its own already-cloned check for
# scripts will simply no-op once we've done this).
#
# HTTPS-only, never SSH: anything SSH here races its own prerequisite,
# since the key-backup step moves ~/.ssh aside on every run -- taking
# known_hosts and rc_files' stowed ~/.ssh/config, which is what makes
# git@github.com work at all, and isn't re-stowed until min.sh runs after
# this. Staying on HTTPS takes both an explicit URL (origin is an SSH
# remote once min.sh has run) and GIT_CONFIG_GLOBAL=/dev/null, because
# rc_files' .gitconfig rewrites https://github.com/ back to
# git@github.com: -- same bypass lib/git_workspace.sh uses.
SCRIPTS_HTTPS_URL="https://github.com/jerryyin/scripts.git"
if [ -d "$HOME/scripts/.git" ]; then
    echo "📥 scripts already present, pulling latest..."
    GIT_CONFIG_GLOBAL=/dev/null git -C "$HOME/scripts" pull --ff-only "$SCRIPTS_HTTPS_URL"
else
    echo "📥 cloning scripts..."
    GIT_CONFIG_GLOBAL=/dev/null git clone "$SCRIPTS_HTTPS_URL" "$HOME/scripts"
fi

echo "🚀 Running env/min.sh..."
bash "$HOME/scripts/docker/env/min.sh"

echo "🔧 Running env/priv.sh --force..."
bash "$HOME/scripts/docker/env/priv.sh" --force || echo "⚠️  priv.sh reported an issue (continuing)"

if [ -n "${NO_BASE_PULL:-}" ]; then
    echo "🐳 Base-image pulls skipped (-n)"
    exit 0
fi

BASE_PULL_SERVICES="triton triton-mi450"
echo "🐳 Pulling base images ($BASE_PULL_SERVICES)..."
# `docker compose pull <svc>` would fetch the service's own tag
# (jeryin/dev:triton), a local-only tag never pushed anywhere -- not the
# Dockerfile's `FROM $BASE_IMAGE`. No compose subcommand fetches just the
# base image, so read the resolved BASE_IMAGE out of compose's own JSON.
COMPOSE_FILE="$HOME/rc_files/docker/.docker/docker-compose.yml"
if [ ! -f "$COMPOSE_FILE" ]; then
    echo "⚠️  $COMPOSE_FILE not found — skipping base-image pulls"
elif ! docker compose version >/dev/null 2>&1; then
    echo "⚠️  'docker compose' not available — skipping base-image pulls"
elif ! command -v python3 >/dev/null 2>&1; then
    echo "⚠️  python3 not available — skipping base-image pulls"
else
    # Both flags are load-bearing; without either, `config` fails outright
    # and every service below reports an unresolvable BASE_IMAGE.
    #   --profile: a service outside the active profile set isn't in the
    #     rendered model at all, so its BASE_IMAGE isn't there to read.
    #   --no-consistency: the dev services deliberately share one
    #     container_name (only one runs at a time), and rendering several
    #     at once trips the duplicate-name check even for a read-only dump.
    PROFILE_FLAGS=""
    for svc in $BASE_PULL_SERVICES; do
        PROFILE_FLAGS="$PROFILE_FLAGS --profile $svc"
    done
    # shellcheck disable=SC2086
    CONFIG_JSON=$(docker compose -f "$COMPOSE_FILE" $PROFILE_FLAGS config --no-consistency --format json 2>/dev/null)
    for svc in $BASE_PULL_SERVICES; do
        IMAGE=$(printf '%s' "$CONFIG_JSON" | python3 -c '
import json, sys
cfg = json.load(sys.stdin)
print(cfg.get("services", {}).get(sys.argv[1], {}).get("build", {}).get("args", {}).get("BASE_IMAGE", ""))
' "$svc" 2>/dev/null)
        if [ -z "$IMAGE" ]; then
            echo "  ⚠️  Could not resolve BASE_IMAGE for $svc — skipping"
            continue
        fi
        echo "  Pulling $IMAGE (base image for $svc)... (docker pull is incremental -- a no-op fetch if already up to date)"
        docker pull "$IMAGE" || echo "  ⚠️  Failed to pull $IMAGE"
    done
fi
