#!/bin/bash
# fetch-am-ffm.sh - Download and extract the AM+FFM-Lite simulator package
# used by the triton-mi450 docker-compose service.
#
# docker-compose.yml mounts this directory straight into the container as
# /am-ffm (see .docker/docker-compose.yml's `triton-mi450` service), and
# run_on_model.sh expects am_env.sh/ffmlite_env.sh directly at its top level
# -- so this always lands the package one level below wherever the tarball's
# own top-level entries are, regardless of whether the tarball wraps its
# contents in a folder.
#
# Usage:
#   fetch-am-ffm.sh [version]        # e.g. fetch-am-ffm.sh 7.13-am+ffmlite-mi400-r6.06
#
# Auth: reads atlartifactory.amd.com credentials from ~/.netrc (kept in sync
# from ~/vault/atlartifactory_token.txt by `vault.sh atlartifactory`).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION="${1:-7.13-am+ffmlite-mi400-r6.06}"
ARTIFACTORY_BASE="https://atlartifactory.amd.com:8443/artifactory/SW-ROCDTIF-MI-DEV-LOCAL/Packages/AM+FFM-LITE/Release"
TARBALL_NAME="rocdtif-${VERSION}.tar.gz"
URL="${ARTIFACTORY_BASE}/${TARBALL_NAME}"

DEST_DIR="${AM_FFM_DIR:-$HOME/rocdtif-${VERSION}}"
CACHE_DIR="$HOME/.cache/am-ffm"
TARBALL="$CACHE_DIR/${TARBALL_NAME}"

if [ ! -f "$HOME/.netrc" ] || ! grep -q "machine atlartifactory.amd.com" "$HOME/.netrc" 2>/dev/null; then
    echo "❌ No atlartifactory.amd.com credentials in ~/.netrc." >&2
    echo "   Run: bash $SCRIPT_DIR/../env/vault.sh atlartifactory" >&2
    exit 1
fi

mkdir -p "$CACHE_DIR"
if [ -f "$TARBALL" ]; then
    echo "✓ Using cached $TARBALL"
else
    echo "📥 Downloading $URL ..."
    curl -fL --netrc --max-time 900 -o "$TARBALL.part" "$URL"
    mv "$TARBALL.part" "$TARBALL"
fi

echo "📦 Extracting to $DEST_DIR ..."
STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT
tar -xzf "$TARBALL" -C "$STAGE_DIR"

# Find the directory that actually holds am_env.sh/ffmlite_env.sh -- either
# the staging root itself (no wrapper folder) or one level under it.
find_pkg_root() {
    local d
    for d in "$STAGE_DIR" "$STAGE_DIR"/*/; do
        [ -d "$d" ] || continue
        if [ -f "$d/am_env.sh" ] || [ -f "$d/ffmlite_env.sh" ]; then
            echo "$d"
            return 0
        fi
    done
    return 1
}

PKG_ROOT="$(find_pkg_root)" || {
    echo "❌ Neither am_env.sh nor ffmlite_env.sh found anywhere in the extracted tarball." >&2
    echo "   Check $TARBALL's layout manually; run_on_model.sh expects one of them at /am-ffm's top level." >&2
    exit 1
}

rm -rf "$DEST_DIR"
mkdir -p "$(dirname "$DEST_DIR")"
mv "$PKG_ROOT" "$DEST_DIR"

echo "✓ AM+FFM package ready at $DEST_DIR"
echo "  Run with: AM_FFM_DIR=$DEST_DIR drun triton-mi450"
