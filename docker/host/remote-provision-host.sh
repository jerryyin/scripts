#!/bin/bash
# remote-provision-host.sh - scp'd to and run ON the target host by
# batch-provision.sh, AFTER the SSH keypair(s) it needs have already been
# scp'd into ~/.ssh (see the inline key-backup step in batch-provision.sh,
# which runs before that).
#
# chmod's the seeded keypair(s), bootstrap-clones just enough of `scripts`
# to make env/min.sh invokable (min.sh itself clones rc_files, and
# re-clones/no-ops scripts -- see below for why we don't just let it),
# then runs env/min.sh and env/priv.sh --force, and (by default) pulls the
# base images the triton/triton-mi450 services build FROM -- see the
# comment above the image-pulling section for why that's a `docker
# compose config --format json` + python3 step rather than `docker
# compose pull`.
#
# Every step here is idempotent and safe to re-run against an
# already-provisioned host: git clone-or-pull, min.sh/priv.sh are
# explicitly designed to be idempotent (see their own comments), and
# `docker pull` only fetches layers that actually changed. Re-running this
# should just fast-forward a host to latest, not redo expensive work.
#
# Self-deletes on exit (success or failure) since it's a transient helper,
# not something that belongs on the target host afterward.
#
# Usage: remote-provision-host.sh KEY_NAME [KEY_NAME ...]
#   each KEY_NAME is a base name under ~/.ssh with a matching KEY_NAME.pub

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
# Deliberately HTTPS-only, both branches, never switching to an SSH origin:
# the key-backup step above just wiped ~/.ssh/known_hosts, and rc_files'
# own ~/.ssh/config -- which redirects git@github.com through
# ssh.github.com:443 (for networks that block outbound port 22) and
# declares StrictHostKeyChecking accept-new -- isn't stowed until min.sh
# runs, *after* this. An SSH clone/pull here would be racing against its
# own prerequisite. `git pull <url>` (as opposed to plain `git pull`,
# which trusts whatever `origin` is already configured to) pins this to
# HTTPS regardless of what a human may have manually switched the remote
# to on a prior visit to this host.
SCRIPTS_HTTPS_URL="https://github.com/jerryyin/scripts.git"
if [ -d "$HOME/scripts/.git" ]; then
    echo "📥 scripts already present, pulling latest..."
    git -C "$HOME/scripts" pull --ff-only "$SCRIPTS_HTTPS_URL"
else
    echo "📥 cloning scripts..."
    git clone "$SCRIPTS_HTTPS_URL" "$HOME/scripts"
fi

echo "🚀 Running env/min.sh..."
bash "$HOME/scripts/docker/env/min.sh"

echo "🔧 Running env/priv.sh --force..."
bash "$HOME/scripts/docker/env/priv.sh" --force || echo "⚠️  priv.sh reported an issue (continuing)"

echo "🐳 Pulling base images (triton, triton-mi450 services)..."
# `docker compose pull <service>` pulls the service's *tagged* image
# (jeryin/dev:triton here), not the Dockerfile's `FROM $BASE_IMAGE` --
# and jeryin/dev:triton is a local-only tag that's never been pushed
# anywhere, so that would just 404. There's no compose subcommand that
# pulls "the FROM image, skip the build" for a Dockerfile shaped like
# this one. The next best thing, and what's used here: compose's own
# structured `config --format json` export (real JSON, not scraped
# YAML text) piped through python3's json module to pull out each
# service's resolved BASE_IMAGE build arg.
COMPOSE_FILE="$HOME/rc_files/docker/.docker/docker-compose.yml"
if [ ! -f "$COMPOSE_FILE" ]; then
    echo "⚠️  $COMPOSE_FILE not found — skipping base-image pulls"
elif ! docker compose version >/dev/null 2>&1; then
    echo "⚠️  'docker compose' not available — skipping base-image pulls"
elif ! command -v python3 >/dev/null 2>&1; then
    echo "⚠️  python3 not available — skipping base-image pulls"
else
    CONFIG_JSON=$(docker compose -f "$COMPOSE_FILE" config --format json 2>/dev/null)
    for svc in triton triton-mi450; do
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
