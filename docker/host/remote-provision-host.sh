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
#
# Env: PROVISION_SERVICES  space-separated compose services whose base
#      images to pre-pull (default "triton triton-mi450"; "none" skips).

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
# Deliberately HTTPS-only, both branches, never over SSH. Anything SSH
# here would be racing its own prerequisite: batch-provision.sh's key
# backup moves ~/.ssh aside on every run, which takes both known_hosts
# AND rc_files' stowed ~/.ssh/config with it -- and that config is what
# makes git@github.com work at all (it redirects to ssh.github.com:443
# for networks blocking outbound 22, and sets StrictHostKeyChecking
# accept-new). It isn't re-stowed until min.sh runs, *after* this.
#
# Two things are needed to actually stay on HTTPS, and it takes both:
#   - `git pull <url>` rather than plain `git pull`, so we don't inherit
#     whatever `origin` happens to point at (min.sh deliberately sets it
#     to an SSH remote once cloned, and a human may have changed it too).
#   - GIT_CONFIG_GLOBAL=/dev/null, because rc_files' ~/.gitconfig carries
#     `url."git@github.com:".insteadOf https://github.com/`, which
#     silently rewrites this HTTPS URL straight back to SSH. Once
#     rc_files is stowed (i.e. every re-provision of a host), the URL
#     alone is not enough. Same bypass lib/git_workspace.sh uses for its
#     no-SSH clone fallback, for the same reason.
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

# Which services' base images to pre-pull. Defaults to the two general-
# purpose dev services; override per host via batch-provision.sh -s, e.g.
# `-s triton-b0` on a real gfx1250 B0 box, or `-s none` when the base image
# is already on disk and a multi-GB re-pull would be pure waste.
PROVISION_SERVICES="${PROVISION_SERVICES:-triton triton-mi450}"

if [ "$PROVISION_SERVICES" = "none" ]; then
    echo "🐳 Base-image pulls skipped (PROVISION_SERVICES=none)"
    exit 0
fi

echo "🐳 Pulling base images ($PROVISION_SERVICES)..."
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
    # Two flags are needed here, and without them this step silently
    # resolved nothing at all: `config` failed, the 2>/dev/null swallowed
    # the error, and every host just logged "Could not resolve BASE_IMAGE"
    # for each service and pulled none of them.
    #   --profile <svc>   Every dev service is profile-gated, and a service
    #                     outside the active profile set isn't in the
    #                     rendered model -- its BASE_IMAGE isn't there to
    #                     read. So ask for exactly the ones we resolve.
    #   --no-consistency  Every dev service inherits one shared
    #                     `container_name: ${COMPOSE_PROJECT_NAME:-dev}`
    #                     from the x-base-service anchor, by design: only
    #                     one dev container runs at a time, and it's always
    #                     called the same thing. But resolving N services
    #                     means activating N of them at once, and duplicate
    #                     container names fail whole-file validation
    #                     ("container name ... is already in use"), so a
    #                     PROVISION_SERVICES with two entries can't render.
    #                     The check is irrelevant to reading build args --
    #                     we never bring these up -- so skip it.
    # Note `config --services` does NOT run that validation while
    # `--format json` does, so the failure only shows up in this form.
    PROFILE_FLAGS=""
    for svc in $PROVISION_SERVICES; do
        PROFILE_FLAGS="$PROFILE_FLAGS --profile $svc"
    done
    # shellcheck disable=SC2086
    CONFIG_JSON=$(docker compose -f "$COMPOSE_FILE" $PROFILE_FLAGS config --no-consistency --format json 2>/dev/null)
    for svc in $PROVISION_SERVICES; do
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
