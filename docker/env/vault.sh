#!/bin/bash
# vault.sh - Patch local config files from vault-managed secrets.
#
# Usage:
#   vault.sh claude [--status]
#   vault.sh docker [--status]
#   vault.sh atlartifactory [--status]
#
# claude/docker share one shape: config.template in rc_files -> config file in
# $HOME -> placeholder replaced by the matching plaintext secret from ~/vault.
# atlartifactory instead patches a marker-delimited block into ~/.netrc (see
# patch_netrc), since there's no rc_files template for that file.

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
    echo "Usage: vault.sh <claude|docker|atlartifactory> [--status]"
    echo "  claude             Patch ~/.claude.json from ~/.claude.json.template"
    echo "  docker             Patch ~/.docker/config.json from ~/.docker/config.json.template"
    echo "  atlartifactory     Patch ~/.netrc with an atlartifactory.amd.com entry"
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
        atlartifactory)
            NETRC_HOST="${NETRC_HOST:-atlartifactory.amd.com}"
            # The token authenticates via Basic auth regardless of username
            # (it's a JFrog identity token, not a password tied to an
            # account), but curl/wget's .netrc parsing still requires some
            # login value to be present.
            NETRC_LOGIN="${NETRC_LOGIN:-$(id -un 2>/dev/null || whoami)}"
            CONFIG_FILE="$HOME/.netrc"
            SECRET_FILE="${ARTIFACTORY_KEY_FILE:-$HOME/vault/atlartifactory_token.txt}"
            DESCRIPTION="Artifactory identity token for $NETRC_HOST"
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
        claude|atlartifactory)
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
        echo "Warning: $TEMPLATE_FILE not found; run rc_files/install.sh first"
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
            echo "Warning: $SECRET_FILE not found; vault not synced yet"
            echo "Run priv.sh to sync vault, then re-run this script."
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

# Unlike patch_config's claude/docker profiles, there's no rc_files template
# to seed from -- ~/.netrc is a plain credential file a user may already
# have entries in for other hosts, so this only ever touches its own
# marker-delimited block (safe to re-run on secret rotation).
patch_netrc() {
    if [ ! -f "$SECRET_FILE" ]; then
        echo "Warning: $SECRET_FILE not found; vault not synced yet"
        echo "Run priv.sh to sync vault, then re-run this script."
        return 0
    fi

    local secret
    secret=$(tr -d '[:space:]' < "$SECRET_FILE")
    if ! validate_secret "$secret"; then
        echo "Warning: $SECRET_FILE is not a valid $DESCRIPTION value"
        return 0
    fi

    local marker_begin="# >>> vault: $NETRC_HOST >>>"
    local marker_end="# <<< vault: $NETRC_HOST <<<"

    touch "$CONFIG_FILE"
    awk -v b="$marker_begin" -v e="$marker_end" '
        $0 == b { skip=1; next }
        $0 == e { skip=0; next }
        !skip { print }
    ' "$CONFIG_FILE" > "$CONFIG_FILE.tmp"

    {
        cat "$CONFIG_FILE.tmp"
        echo "$marker_begin"
        echo "machine $NETRC_HOST"
        echo "login $NETRC_LOGIN"
        echo "password $secret"
        echo "$marker_end"
    } > "$CONFIG_FILE"
    rm -f "$CONFIG_FILE.tmp"
    chmod 600 "$CONFIG_FILE"
    echo "Patched $DESCRIPTION into $CONFIG_FILE"
}

show_status() {
    local config_state="missing"
    local secret_state="missing"

    if [ "$PROFILE" = "atlartifactory" ]; then
        if [ -f "$CONFIG_FILE" ] && grep -qF "machine $NETRC_HOST" "$CONFIG_FILE" 2>/dev/null; then
            config_state="configured"
        fi
    elif [ -f "$CONFIG_FILE" ]; then
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
    [ "$PROFILE" = "atlartifactory" ] || echo "Template:     $TEMPLATE_FILE"
    echo "Vault secret: $SECRET_FILE ($secret_state)"
}

if [ "$#" -ne 0 ]; then
    usage
    exit 1
fi

configure_profile

case "$MODE" in
    "")
        if [ "$PROFILE" = "atlartifactory" ]; then
            patch_netrc
        else
            patch_config
        fi
        ;;
    --status)
        show_status
        ;;
    *)
        usage
        exit 1
        ;;
esac
