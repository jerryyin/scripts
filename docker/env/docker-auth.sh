#!/bin/bash
# docker-auth.sh - Patch Docker registry auth from vault
#
# Modes:
#   docker-auth.sh                # Patch ~/.docker/config.json from template
#   docker-auth.sh --patch-only   # Same as default, quiet on no-op
#   docker-auth.sh --save         # Save current mkmhub auth from config.json to vault
#   docker-auth.sh --status       # Show non-secret status
#
# The config file (~/.docker/config.json) is generated from
# ~/.docker/config.json.template, deployed by rc_files. The template contains
# __DOCKER_KEY__, which is substituted with a Docker "auth" value from vault.
#
# Docker's plain config auth is base64(username:secret), not encryption. Storing
# that value in the private vault keeps the source of truth out of rc_files while
# still allowing host-side Docker pulls before any container exists.

set -e

DOCKER_REGISTRY="${DOCKER_REGISTRY:-mkmhub.amd.com}"
DOCKER_CONFIG_DIR="${DOCKER_CONFIG:-$HOME/.docker}"
DOCKER_CONFIG_FILE="$DOCKER_CONFIG_DIR/config.json"
DOCKER_TEMPLATE="$DOCKER_CONFIG_DIR/config.json.template"
DOCKER_AUTH_FILE="${DOCKER_AUTH_FILE:-$HOME/vault/docker_mkmhub_auth.txt}"
DOCKER_PLACEHOLDER="${DOCKER_PLACEHOLDER:-__DOCKER_KEY__}"

have_jq() {
    command -v jq >/dev/null 2>&1
}

validate_auth() {
    local auth="$1"
    local decoded username secret

    decoded=$(printf '%s' "$auth" | base64 -d 2>/dev/null || true)
    case "$decoded" in
        *:*) ;;
        *) return 1 ;;
    esac

    username="${decoded%%:*}"
    secret="${decoded#*:}"
    [ -n "$username" ] && [ -n "$secret" ]
}

patch_docker_config() {
    if [ ! -f "$DOCKER_TEMPLATE" ]; then
        [ "${QUIET_NOOP:-0}" = "1" ] || echo "⚠️  $DOCKER_TEMPLATE not found — run rc_files/install.sh first"
        return 0
    fi

    mkdir -p "$DOCKER_CONFIG_DIR"

    if [ ! -f "$DOCKER_CONFIG_FILE" ] || grep -Fq "$DOCKER_PLACEHOLDER" "$DOCKER_CONFIG_FILE"; then
        cp "$DOCKER_TEMPLATE" "$DOCKER_CONFIG_FILE"
        chmod 600 "$DOCKER_CONFIG_FILE" 2>/dev/null || true
        echo "✓ Copied $DOCKER_TEMPLATE → $DOCKER_CONFIG_FILE"
    fi

    if grep -Fq "$DOCKER_PLACEHOLDER" "$DOCKER_CONFIG_FILE"; then
        if [ ! -f "$DOCKER_AUTH_FILE" ]; then
            [ "${QUIET_NOOP:-0}" = "1" ] || {
                echo "⚠️  $DOCKER_AUTH_FILE not found — vault not synced yet"
                echo "   Run docker-auth.sh --save on a logged-in machine, then commit/push vault."
            }
            return 0
        fi

        local auth
        auth=$(tr -d '[:space:]' < "$DOCKER_AUTH_FILE")
        if ! validate_auth "$auth"; then
            echo "⚠️  $DOCKER_AUTH_FILE is not a valid Docker auth value"
            return 0
        fi

        DOCKER_KEY="$auth" DOCKER_PLACEHOLDER="$DOCKER_PLACEHOLDER" \
            perl -0pi -e 's/\Q$ENV{DOCKER_PLACEHOLDER}\E/$ENV{DOCKER_KEY}/g' "$DOCKER_CONFIG_FILE"
        chmod 600 "$DOCKER_CONFIG_FILE" 2>/dev/null || true
        echo "✓ Docker auth patched into $DOCKER_CONFIG_FILE for $DOCKER_REGISTRY"
    fi
}

save_docker_auth() {
    if ! have_jq; then
        echo "❌ jq is required to save Docker auth"
        return 1
    fi
    if [ ! -f "$DOCKER_CONFIG_FILE" ]; then
        echo "❌ $DOCKER_CONFIG_FILE not found — run docker login $DOCKER_REGISTRY first"
        return 1
    fi

    local auth
    auth=$(jq -r --arg registry "$DOCKER_REGISTRY" '.auths[$registry].auth // empty' "$DOCKER_CONFIG_FILE")
    if [ -z "$auth" ]; then
        echo "❌ No auth entry for $DOCKER_REGISTRY in $DOCKER_CONFIG_FILE"
        return 1
    fi
    if ! validate_auth "$auth"; then
        echo "❌ Auth entry for $DOCKER_REGISTRY is not a valid Docker auth value"
        return 1
    fi

    mkdir -p "$(dirname "$DOCKER_AUTH_FILE")"
    umask 077
    printf '%s\n' "$auth" > "$DOCKER_AUTH_FILE"
    chmod 600 "$DOCKER_AUTH_FILE" 2>/dev/null || true
    echo "✓ Saved Docker auth for $DOCKER_REGISTRY to $DOCKER_AUTH_FILE"
}

show_status() {
    local config_state="missing"
    local vault_state="missing"

    if [ -f "$DOCKER_CONFIG_FILE" ]; then
        if grep -Fq "$DOCKER_PLACEHOLDER" "$DOCKER_CONFIG_FILE"; then
            config_state="template-placeholder"
        elif have_jq && jq -e --arg registry "$DOCKER_REGISTRY" '.auths[$registry].auth?' "$DOCKER_CONFIG_FILE" >/dev/null 2>&1; then
            config_state="configured"
        else
            config_state="present"
        fi
    fi
    if [ -f "$DOCKER_AUTH_FILE" ]; then
        vault_state="present"
    fi

    echo "Docker registry: $DOCKER_REGISTRY"
    echo "Config file:     $DOCKER_CONFIG_FILE ($config_state)"
    echo "Template:        $DOCKER_TEMPLATE"
    echo "Vault auth:      $DOCKER_AUTH_FILE ($vault_state)"
}

main() {
    case "${1:-}" in
        "" )
            patch_docker_config
            ;;
        --patch-only)
            QUIET_NOOP=1 patch_docker_config
            ;;
        --save)
            save_docker_auth
            ;;
        --status)
            show_status
            ;;
        *)
            echo "Usage: docker-auth.sh [--patch-only|--save|--status]"
            exit 1
            ;;
    esac
}

main "$@"
