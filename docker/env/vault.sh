#!/bin/bash
# vault.sh - Patch local config files from vault-managed placeholders.
#
# Usage:
#   vault.sh claude [--patch-only|--status]
#   vault.sh docker [--patch-only|--save|--status]
#
# Each profile has the same shape:
#   config.template in rc_files -> config file in $HOME -> placeholder replaced
#   by the matching plaintext secret from ~/vault.

set -e

PROFILE="${1:-}"
if [ -n "$PROFILE" ]; then
    shift
fi
MODE="${1:-}"
if [ -n "$MODE" ]; then
    shift
fi

usage() {
    echo "Usage: vault.sh <claude|docker> [--patch-only|--save|--status]"
    echo "  claude             Patch ~/.claude.json from ~/.claude.json.template"
    echo "  docker             Patch ~/.docker/config.json from ~/.docker/config.json.template"
    echo "  --patch-only       Patch config, quiet when there is nothing to do"
    echo "  --save             Docker only: save current login auth to vault"
    echo "  --status           Show non-secret status"
}

have_jq() {
    command -v jq >/dev/null 2>&1
}

configure_profile() {
    case "$PROFILE" in
        claude)
            CONFIG_FILE="${CLAUDE_CONFIG:-$HOME/.claude.json}"
            TEMPLATE_FILE="${CLAUDE_TEMPLATE:-$HOME/.claude.json.template}"
            SECRET_FILE="${KEY_FILE:-${CLAUDE_KEY_FILE:-$HOME/vault/claude_key.txt}}"
            PLACEHOLDER="${CLAUDE_PLACEHOLDER:-__CLAUDE_SUB_KEY__}"
            DESCRIPTION="Claude subscription key"
            ;;
        docker)
            DOCKER_REGISTRY="${DOCKER_REGISTRY:-mkmhub.amd.com}"
            DOCKER_CONFIG_DIR="${DOCKER_CONFIG:-$HOME/.docker}"
            CONFIG_FILE="$DOCKER_CONFIG_DIR/config.json"
            TEMPLATE_FILE="$DOCKER_CONFIG_DIR/config.json.template"
            SECRET_FILE="${DOCKER_AUTH_FILE:-$HOME/vault/docker_mkmhub_auth.txt}"
            PLACEHOLDER="${DOCKER_PLACEHOLDER:-__DOCKER_KEY__}"
            DESCRIPTION="Docker auth for $DOCKER_REGISTRY"
            ;;
        *)
            usage
            exit 1
            ;;
    esac
}

validate_secret() {
    local secret="$1"
    local decoded username password

    case "$PROFILE" in
        claude)
            [ -n "$secret" ]
            ;;
        docker)
            decoded=$(printf '%s' "$secret" | base64 -d 2>/dev/null || true)
            case "$decoded" in
                *:*) ;;
                *) return 1 ;;
            esac
            username="${decoded%%:*}"
            password="${decoded#*:}"
            [ -n "$username" ] && [ -n "$password" ]
            ;;
    esac
}

patch_config() {
    if [ ! -f "$TEMPLATE_FILE" ]; then
        [ "${QUIET_NOOP:-0}" = "1" ] || echo "Warning: $TEMPLATE_FILE not found; run rc_files/install.sh first"
        return 0
    fi

    mkdir -p "$(dirname "$CONFIG_FILE")"

    if [ ! -f "$CONFIG_FILE" ] || grep -Fq "$PLACEHOLDER" "$CONFIG_FILE"; then
        cp "$TEMPLATE_FILE" "$CONFIG_FILE"
        chmod 600 "$CONFIG_FILE" 2>/dev/null || true
        echo "Copied $TEMPLATE_FILE -> $CONFIG_FILE"
    fi

    if grep -Fq "$PLACEHOLDER" "$CONFIG_FILE"; then
        if [ ! -f "$SECRET_FILE" ]; then
            [ "${QUIET_NOOP:-0}" = "1" ] || {
                echo "Warning: $SECRET_FILE not found; vault not synced yet"
                echo "Run priv.sh to sync vault, then re-run this script."
            }
            return 0
        fi

        local secret
        secret=$(tr -d '[:space:]' < "$SECRET_FILE")
        if ! validate_secret "$secret"; then
            echo "Warning: $SECRET_FILE is not a valid $DESCRIPTION value"
            return 0
        fi

        SECRET_VALUE="$secret" PLACEHOLDER="$PLACEHOLDER" \
            perl -0pi -e 'BEGIN { $p = $ENV{PLACEHOLDER}; $v = $ENV{SECRET_VALUE}; } s/\Q$p\E/$v/g' "$CONFIG_FILE"
        chmod 600 "$CONFIG_FILE" 2>/dev/null || true
        echo "Patched $DESCRIPTION into $CONFIG_FILE"
    fi
}

save_secret() {
    if [ "$PROFILE" != "docker" ]; then
        echo "--save is only supported for docker"
        return 1
    fi
    if ! have_jq; then
        echo "jq is required to save Docker auth"
        return 1
    fi
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "$CONFIG_FILE not found; run docker login $DOCKER_REGISTRY first"
        return 1
    fi

    local secret
    secret=$(jq -r --arg registry "$DOCKER_REGISTRY" '.auths[$registry].auth // empty' "$CONFIG_FILE")
    if [ -z "$secret" ]; then
        echo "No auth entry for $DOCKER_REGISTRY in $CONFIG_FILE"
        return 1
    fi
    if ! validate_secret "$secret"; then
        echo "Auth entry for $DOCKER_REGISTRY is not a valid Docker auth value"
        return 1
    fi

    mkdir -p "$(dirname "$SECRET_FILE")"
    umask 077
    printf '%s\n' "$secret" > "$SECRET_FILE"
    chmod 600 "$SECRET_FILE" 2>/dev/null || true
    echo "Saved $DESCRIPTION to $SECRET_FILE"
}

show_status() {
    local config_state="missing"
    local secret_state="missing"

    if [ -f "$CONFIG_FILE" ]; then
        if grep -Fq "$PLACEHOLDER" "$CONFIG_FILE"; then
            config_state="template-placeholder"
        elif [ "$PROFILE" = "docker" ] && have_jq \
            && jq -e --arg registry "$DOCKER_REGISTRY" '.auths[$registry].auth?' "$CONFIG_FILE" >/dev/null 2>&1; then
            config_state="configured"
        else
            config_state="configured"
        fi
    fi
    if [ -f "$SECRET_FILE" ]; then
        secret_state="present"
    fi

    echo "Profile:      $PROFILE"
    echo "Config file:  $CONFIG_FILE ($config_state)"
    echo "Template:     $TEMPLATE_FILE"
    echo "Vault secret: $SECRET_FILE ($secret_state)"
}

if [ "$#" -ne 0 ]; then
    usage
    exit 1
fi

configure_profile

case "$MODE" in
    "")
        patch_config
        ;;
    --patch-only)
        QUIET_NOOP=1 patch_config
        ;;
    --save)
        save_secret
        ;;
    --status)
        show_status
        ;;
    *)
        usage
        exit 1
        ;;
esac
