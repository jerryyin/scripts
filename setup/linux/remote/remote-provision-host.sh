#!/bin/bash
# remote-provision-host.sh - scp'd to and run ON the target host by
# batch-provision.sh, AFTER the SSH keypair(s) it needs have already been
# scp'd into ~/.ssh (see the inline key-backup step in batch-provision.sh,
# which runs before that).
#
# chmod's the seeded keypair(s), bootstrap-clones just enough of `scripts`
# to make setup/linux/bootstrap.sh invokable (bootstrap.sh itself clones rc_files, and
# re-clones/no-ops scripts -- see below for why we don't just let it),
# then runs setup/linux/bootstrap.sh and credentials/activate.sh --force, and pulls the base images
# the dev services build FROM.
#
# Every step here is idempotent and safe to re-run against an
# already-provisioned host: git clone-or-pull, bootstrap.sh/credentials/activate.sh are
# explicitly designed to be idempotent (see their own comments), and
# `docker pull` only fetches layers that actually changed.
#
# Self-deletes on exit (success or failure) since it's a transient helper.
#
# Usage: remote-provision-host.sh KEY_NAME [KEY_NAME ...]
#   each KEY_NAME is a base name under ~/.ssh with a matching KEY_NAME.pub
# Env: NO_BASE_PULL=1 skips the base-image pulls (batch-provision.sh -n)

set -e
# Delete only a copied bootstrap helper, never a tracked entry point.
REMOTE_HELPER_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ ! -f "$REMOTE_HELPER_DIR/../setup.sh" ]]; then
    trap 'rm -f -- "$0"' EXIT
fi

for name in "$@"; do
    chmod 600 "$HOME/.ssh/$name"
    chmod 644 "$HOME/.ssh/$name.pub"
done
echo "✓ Keypair permissions set"

# Just enough to make `bash ~/scripts/setup/linux/bootstrap.sh` invokable -- bootstrap.sh
# lives inside this clone, so something has to put it on disk first (bootstrap.sh
# itself will still handle rc_files, and its own already-cloned check for
# scripts will simply no-op once we've done this).
#
# HTTPS-only, never SSH: anything SSH here races its own prerequisite,
# since the key-backup step moves ~/.ssh aside on every run -- taking
# known_hosts and rc_files' linked ~/.ssh/config, which is what makes
# git@github.com work at all, and isn't redeployed until bootstrap.sh runs after
# this. Staying on HTTPS takes both an explicit URL (origin is an SSH
# remote once bootstrap.sh has run) and GIT_CONFIG_GLOBAL=/dev/null, because
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

echo "🚀 Running setup/linux/bootstrap.sh..."
bash "$HOME/scripts/setup/linux/bootstrap.sh"

echo "🔧 Running credentials/activate.sh --force..."
bash "$HOME/scripts/setup/linux/credentials/activate.sh" --force || echo "⚠️  credentials/activate.sh reported an issue (continuing)"

if [ -n "${NO_BASE_PULL:-}" ]; then
    echo "🐳 Base-image pulls skipped (-n)"
    exit 0
fi

bash "$HOME/scripts/containers/docker/pull-base-images.sh" triton triton-mi450
