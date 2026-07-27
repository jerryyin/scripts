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
    echo "Usage: vault.sh <claude|docker|gh|atlartifactory> [--status]"
    echo "  claude             Patch ~/.claude.json from ~/.claude.json.template"
    echo "  docker             Patch ~/.docker/config.json from ~/.docker/config.json.template"
    echo "  gh                 Patch ~/.config/gh/hosts.yml from its template (two account tokens)"
    echo "  atlartifactory     Patch ~/.netrc with an atlartifactory.amd.com entry"
    echo "  --status           Show non-secret status"
}

# claude/docker/gh share the template+placeholder shape and use parallel
# PLACEHOLDERS/SECRET_FILES arrays so one config file can carry more than one
# secret (gh holds two account tokens). atlartifactory is handled separately by
# patch_netrc and uses only SECRET_FILES[0].
configure_profile() {
    case "$PROFILE" in
        claude)
            CONFIG_FILE="${CLAUDE_CONFIG:-$HOME/.claude.json}"
            TEMPLATE_FILE="${CLAUDE_TEMPLATE:-$HOME/.claude.json.template}"
            SECRET_FILES=("${KEY_FILE:-${CLAUDE_KEY_FILE:-$HOME/vault/claude_key.txt}}")
            PLACEHOLDERS=("${CLAUDE_PLACEHOLDER:-__CLAUDE_SUB_KEY__}")
            DESCRIPTION="Claude subscription key"
            ;;
        docker)
            DOCKER_REGISTRY="${DOCKER_REGISTRY:-mkmhub.amd.com}"
            DOCKER_CONFIG_DIR="${DOCKER_CONFIG:-$HOME/.docker}"
            CONFIG_FILE="$DOCKER_CONFIG_DIR/config.json"
            TEMPLATE_FILE="$DOCKER_CONFIG_DIR/config.json.template"
            SECRET_FILES=("${DOCKER_AUTH_FILE:-$HOME/vault/docker_mkmhub_auth.txt}")
            PLACEHOLDERS=("${DOCKER_PLACEHOLDER:-__DOCKER_KEY__}")
            DESCRIPTION="Docker auth for $DOCKER_REGISTRY"
            ;;
        gh)
            GH_CONFIG_DIR="${GH_CONFIG_DIR:-$HOME/.config/gh}"
            CONFIG_FILE="$GH_CONFIG_DIR/hosts.yml"
            TEMPLATE_FILE="$GH_CONFIG_DIR/hosts.yml.template"
            # Order matters: each secret file pairs with the placeholder at the
            # same index.
            SECRET_FILES=(
                "${GH_JERRYYIN_KEY_FILE:-$HOME/vault/gh_token_jerryyin.txt}"
                "${GH_AMDENG_KEY_FILE:-$HOME/vault/gh_token_amdeng.txt}"
            )
            PLACEHOLDERS=(
                "${GH_JERRYYIN_PLACEHOLDER:-__GH_TOKEN_JERRYYIN__}"
                "${GH_AMDENG_PLACEHOLDER:-__GH_TOKEN_AMDENG__}"
            )
            DESCRIPTION="GitHub CLI tokens"
            ;;
        atlartifactory)
            NETRC_HOST="${NETRC_HOST:-atlartifactory.amd.com}"
            # The token authenticates via Basic auth regardless of username
            # (it's a JFrog identity token, not a password tied to an
            # account), but curl/wget's .netrc parsing still requires some
            # login value to be present.
            NETRC_LOGIN="${NETRC_LOGIN:-$(id -un 2>/dev/null || whoami)}"
            CONFIG_FILE="$HOME/.netrc"
            SECRET_FILES=("${ARTIFACTORY_KEY_FILE:-$HOME/vault/atlartifactory_token.txt}")
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
        claude|gh|atlartifactory)
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

    # Seed from the template when the config is missing or still carries any
    # placeholder from a prior (partial) run. Once every placeholder has been
    # substituted the config is owned by the tool (e.g. gh rewrites hosts.yml on
    # `gh auth switch`) and is left untouched across restarts.
    local seed=0 placeholder secret_file secret
    if [ ! -f "$CONFIG_FILE" ]; then
        seed=1
    else
        for placeholder in "${PLACEHOLDERS[@]}"; do
            if grep -Fq "$placeholder" "$CONFIG_FILE"; then
                seed=1
                break
            fi
        done
    fi
    if [ "$seed" = 1 ]; then
        cp "$TEMPLATE_FILE" "$CONFIG_FILE"
        chmod 600 "$CONFIG_FILE" 2>/dev/null || true
        echo "Copied $TEMPLATE_FILE -> $CONFIG_FILE"
    fi

    # Substitute each placeholder independently so a missing/invalid secret for
    # one account doesn't block patching the others.
    local i patched=0
    for i in "${!PLACEHOLDERS[@]}"; do
        placeholder="${PLACEHOLDERS[$i]}"
        secret_file="${SECRET_FILES[$i]}"
        grep -Fq "$placeholder" "$CONFIG_FILE" || continue
        if [ ! -f "$secret_file" ]; then
            echo "Warning: $secret_file not found; vault not synced yet"
            echo "Run priv.sh to sync vault, then re-run this script."
            continue
        fi
        secret=$(tr -d '[:space:]' < "$secret_file")
        if ! validate_secret "$secret"; then
            echo "Warning: $secret_file is not a valid $DESCRIPTION value"
            continue
        fi
        SECRET_VALUE="$secret" PLACEHOLDER="$placeholder" \
            perl -0pi -e 'BEGIN { $p = $ENV{PLACEHOLDER}; $v = $ENV{SECRET_VALUE}; } s/\Q$p\E/$v/g' "$CONFIG_FILE"
        patched=1
    done

    if [ "$patched" = 1 ]; then
        chmod 600 "$CONFIG_FILE" 2>/dev/null || true
        echo "Patched $DESCRIPTION into $CONFIG_FILE"
    fi
}

# Unlike patch_config's claude/docker profiles, there's no rc_files template
# to seed from -- ~/.netrc is a plain credential file a user may already
# have entries in for other hosts, so this only ever touches its own
# marker-delimited block (safe to re-run on secret rotation).
patch_netrc() {
    local secret_file="${SECRET_FILES[0]}"
    if [ ! -f "$secret_file" ]; then
        echo "Warning: $secret_file not found; vault not synced yet"
        echo "Run priv.sh to sync vault, then re-run this script."
        return 0
    fi

    local secret
    secret=$(tr -d '[:space:]' < "$secret_file")
    if ! validate_secret "$secret"; then
        echo "Warning: $secret_file is not a valid $DESCRIPTION value"
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

    if [ "$PROFILE" = "atlartifactory" ]; then
        if [ -f "$CONFIG_FILE" ] && grep -qF "machine $NETRC_HOST" "$CONFIG_FILE" 2>/dev/null; then
            config_state="configured"
        fi
    elif [ -f "$CONFIG_FILE" ]; then
        config_state="configured"
        local placeholder
        for placeholder in "${PLACEHOLDERS[@]}"; do
            if grep -Fq "$placeholder" "$CONFIG_FILE"; then
                config_state="template-placeholder"
                break
            fi
        done
    fi

    echo "Profile:      $PROFILE"
    echo "Config file:  $CONFIG_FILE ($config_state)"
    [ "$PROFILE" = "atlartifactory" ] || echo "Template:     $TEMPLATE_FILE"
    local secret_file
    for secret_file in "${SECRET_FILES[@]}"; do
        if [ -f "$secret_file" ]; then
            echo "Vault secret: $secret_file (present)"
        else
            echo "Vault secret: $secret_file (missing)"
        fi
    done
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
