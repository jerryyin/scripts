#!/bin/bash
# vault.sh - Patch local config files from vault-managed secrets.
#
# Usage:
#   vault.sh claude [--status|--force]
#   vault.sh docker [--status|--force]
#   vault.sh atlartifactory [--status]
#
# claude/docker share one shape: config.template in rc_files -> config file in
# $HOME -> placeholder replaced by the matching plaintext secret from ~/vault.
# atlartifactory instead patches a marker-delimited block into ~/.netrc (see
# patch_netrc), since there's no rc_files template for that file.
#
# Re-runnable: a rotated secret is re-applied automatically when the live config
# is still its template apart from the secret values, so re-seeding cannot lose
# anything. A config the tool itself writes into (~/.claude.json accumulates
# projects/machineID/userID) fails that test and is left alone until --force.

set -e

PROFILE="${1:-}"
if [ -n "$PROFILE" ]; then
    shift
fi
MODE="${1:-}"
if [ -n "$MODE" ]; then
    shift
fi

# Only patch_config needs this: patch_netrc rewrites its marker block every run
# and patch_rawfile compares content, so both already pick up a rotated secret.
FORCE=0
if [ "$MODE" = "--force" ]; then
    FORCE=1
fi

# The substitution below renders the secret into a temp file. `set -e` means a
# failure mid-render would otherwise leave a credential sitting in $TMPDIR.
VAULT_TMP=""
cleanup_tmp() {
    if [ -n "$VAULT_TMP" ]; then
        rm -f "$VAULT_TMP"
    fi
}
trap cleanup_tmp EXIT

usage() {
    echo "Usage: vault.sh <claude|docker|gh|gist|atlartifactory> [--status|--force]"
    echo "  claude             Patch ~/.claude.json from ~/.claude.json.template"
    echo "  docker             Patch ~/.docker/config.json from ~/.docker/config.json.template"
    echo "  gh                 Patch ~/.config/gh/hosts.yml from its template (two account tokens)"
    echo "  gist               Write ~/.gist (reuses the gh jerryyin PAT; needs gist scope)"
    echo "  atlartifactory     Patch ~/.netrc with an atlartifactory.amd.com entry"
    echo "  --status           Show non-secret status"
    echo "  --force            Re-seed from the template and re-apply the vault secret,"
    echo "                     even when the config holds state of its own. Rotations are"
    echo "                     applied automatically when that is lossless, so this is"
    echo "                     only needed for a config the tool writes into (e.g."
    echo "                     ~/.claude.json). Discards local edits -- for gh, that"
    echo "                     includes the active account set by \`gh auth switch\`."
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
        gist)
            # gist-paste reads a bare token from ~/.gist. Reuse the gh jerryyin
            # PAT (it carries the `gist` scope), so there's no separate secret
            # file to maintain. No template/placeholder -- written raw by
            # patch_rawfile. Gists are created under the jerryyin identity.
            CONFIG_FILE="${GIST_CONFIG:-$HOME/.gist}"
            SECRET_FILES=("${GIST_KEY_FILE:-${GH_JERRYYIN_KEY_FILE:-$HOME/vault/gh_token_jerryyin.txt}}")
            DESCRIPTION="gist-paste token (gh jerryyin PAT)"
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
        claude|gh|gist|atlartifactory)
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

# True when $secret_file's current value already appears in CONFIG_FILE. The
# value is piped to grep rather than passed as an argument to keep it out of the
# process table.
secret_in_config() {
    local secret
    [ -f "$CONFIG_FILE" ] || return 1
    [ -f "$1" ] || return 1
    secret=$(tr -d '[:space:]' < "$1")
    validate_secret "$secret" || return 1
    printf '%s\n' "$secret" | grep -Fqf - "$CONFIG_FILE" 2>/dev/null
}

# True when every vault secret for this profile is already in the live config.
# A false here after the placeholders are gone is exactly the rotation case.
secrets_in_sync() {
    local secret_file
    for secret_file in "${SECRET_FILES[@]}"; do
        secret_in_config "$secret_file" || return 1
    done
    return 0
}

# True when the live config is the template modulo secret values -- i.e. it
# carries no state of its own, so re-seeding provably loses nothing.
#
# Compares line by line, positionally. A template line holding a placeholder
# matches the live line at the same index when the text on either side of the
# placeholder still brackets it, so only the secret itself may differ; every
# other line must be equal. Positional matching is what keeps gh honest: its
# two tokens sit on lines with an identical `oauth_token: ` prefix, so matching
# by prefix alone would let either value satisfy either line. Anything else --
# an extra key, a reordering, a differing line count -- counts as drift, which
# is the conservative direction: at worst we decline to auto-apply and ask for
# --force. Sets TEMPLATE_DRIFT_LINES for the caller's message.
TEMPLATE_DRIFT_LINES=0
config_is_template_shaped() {
    local out
    out=$(TPL="$TEMPLATE_FILE" LIVE="$CONFIG_FILE" perl -e '
        open my $tf, "<", $ENV{TPL}  or exit 2;
        open my $lf, "<", $ENV{LIVE} or exit 2;
        my @t = <$tf>; my @l = <$lf>;
        chomp @t; chomp @l;
        if (@t != @l) { print abs(@t - @l) || 1, "\n"; exit 1 }
        my $drift = 0;
        for my $i (0 .. $#t) {
            if ($t[$i] =~ /^(.*?)__[A-Z_]+__(.*)$/) {
                my ($p, $s) = ($1, $2);
                $drift++ unless $l[$i] =~ /^\Q$p\E.*\Q$s\E$/;
            } else {
                $drift++ unless $t[$i] eq $l[$i];
            }
        }
        print "$drift\n";
        exit($drift ? 1 : 0);
    ') || true
    TEMPLATE_DRIFT_LINES="${out:-1}"
    [ "$TEMPLATE_DRIFT_LINES" = "0" ]
}

patch_config() {
    if [ ! -f "$TEMPLATE_FILE" ]; then
        echo "Warning: $TEMPLATE_FILE not found; run rc_files/install.sh first"
        return 0
    fi

    mkdir -p "$(dirname "$CONFIG_FILE")"

    # On a freshly provisioned machine the config path is a symlink into
    # rc_files whose target does not exist yet -- the rendered config is
    # gitignored, so a clone ships the link but not the file. cp refuses to
    # write through a dangling symlink ("not writing through dangling
    # symlink"), so materialize the target before seeding.
    if [ -L "$CONFIG_FILE" ] && [ ! -e "$CONFIG_FILE" ]; then
        local link_target
        link_target=$(readlink -f "$CONFIG_FILE")
        mkdir -p "$(dirname "$link_target")"
        # Seed it from the template rather than touching it empty: an empty file
        # is "present with no placeholder", which the logic below reads as
        # already-patched and skips.
        cp "$TEMPLATE_FILE" "$link_target"
        chmod 600 "$link_target"
        echo "Created missing symlink target $link_target"
    fi

    # Seed from the template when the config is missing or still carries any
    # placeholder from a prior (partial) run. Once every placeholder has been
    # substituted the config is owned by the tool (e.g. gh rewrites hosts.yml on
    # `gh auth switch`) and is left untouched across restarts.
    #
    # That ownership rule would make a plain re-run a no-op after a rotation:
    # the config holds the *old* secret and no placeholder, so nothing matches.
    # So when the secrets have drifted, re-seed anyway -- but only once the
    # config is shown to carry no state of its own, which makes re-seeding
    # lossless by construction. A config the tool has written into (e.g.
    # ~/.claude.json, which accumulates projects/machineID/userID) fails that
    # test and is left alone until --force says to discard it.
    local seed=0 placeholder secret_file secret rotation_blocked=0
    if [ ! -f "$CONFIG_FILE" ] || [ "$FORCE" = 1 ]; then
        seed=1
    else
        for placeholder in "${PLACEHOLDERS[@]}"; do
            if grep -Fq "$placeholder" "$CONFIG_FILE"; then
                seed=1
                break
            fi
        done
        if [ "$seed" = 0 ] && ! secrets_in_sync; then
            if config_is_template_shaped; then
                echo "Vault secret changed; config matches the template apart from secrets."
                seed=1
            else
                echo "Vault secret changed, but $CONFIG_FILE holds local state not in"
                echo "the template ($TEMPLATE_DRIFT_LINES lines); refusing to re-seed."
                echo "Re-run with --force to discard that state and re-apply the secret."
                rotation_blocked=1
            fi
        fi
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
        # Substitute into a temp file, then write the result *through*
        # $CONFIG_FILE. `perl -i` renames a new file over the target, which
        # replaces a symlink with a regular file -- and these configs are
        # symlinks into rc_files, so in-place editing strands the real file
        # holding the placeholder while the secret lands on the link path.
        VAULT_TMP=$(mktemp)
        SECRET_VALUE="$secret" PLACEHOLDER="$placeholder" \
            perl -0pe 'BEGIN { $p = $ENV{PLACEHOLDER}; $v = $ENV{SECRET_VALUE}; } s/\Q$p\E/$v/g' \
            "$CONFIG_FILE" > "$VAULT_TMP"
        cat "$VAULT_TMP" > "$CONFIG_FILE"
        rm -f "$VAULT_TMP"
        VAULT_TMP=""
        patched=1
    done

    if [ "$patched" = 1 ]; then
        chmod 600 "$CONFIG_FILE" 2>/dev/null || true
        echo "Patched $DESCRIPTION into $CONFIG_FILE"
    elif [ "$rotation_blocked" = 0 ]; then
        echo "$DESCRIPTION already up to date in $CONFIG_FILE"
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

# Write a secret verbatim into CONFIG_FILE (no template, no placeholder). Used
# by the gist profile: ~/.gist is just a bare token. Idempotent -- skips the
# write (and its log line) when the file already matches.
patch_rawfile() {
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

    if [ -f "$CONFIG_FILE" ] && [ "$(tr -d '[:space:]' < "$CONFIG_FILE")" = "$secret" ]; then
        return 0
    fi

    mkdir -p "$(dirname "$CONFIG_FILE")"
    printf '%s\n' "$secret" > "$CONFIG_FILE"
    chmod 600 "$CONFIG_FILE"
    echo "Wrote $DESCRIPTION to $CONFIG_FILE"
}

show_status() {
    local config_state="missing"

    if [ "$PROFILE" = "gist" ]; then
        [ -f "$CONFIG_FILE" ] && config_state="configured"
    elif [ "$PROFILE" = "atlartifactory" ]; then
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
    case "$PROFILE" in atlartifactory|gist) ;; *) echo "Template:     $TEMPLATE_FILE" ;; esac
    local secret_file
    for secret_file in "${SECRET_FILES[@]}"; do
        if [ -f "$secret_file" ]; then
            echo "Vault secret: $secret_file (present, $(vault_sync_state "$secret_file"))"
        else
            echo "Vault secret: $secret_file (missing)"
        fi
    done
}

# Report whether the live config already carries the vault's current secret, so
# a rotation that has not been applied is visible without dumping the secret.
# The value is piped to grep rather than passed as an argument to keep it out of
# the process table.
vault_sync_state() {
    local secret_file="$1"
    [ -f "$CONFIG_FILE" ] || { echo "config missing"; return 0; }
    if secret_in_config "$secret_file"; then
        echo "in sync with config"
        return 0
    fi
    # Out of sync: say whether a plain re-run will fix it or --force is needed.
    case "$PROFILE" in
        gist|atlartifactory)
            echo "rotated -- next run re-applies it"
            ;;
        *)
            if config_is_template_shaped; then
                echo "rotated -- next run re-applies it"
            else
                echo "rotated -- config has local state, needs --force"
            fi
            ;;
    esac
}

if [ "$#" -ne 0 ]; then
    usage
    exit 1
fi

configure_profile

case "$MODE" in
    ""|--force)
        case "$PROFILE" in
            atlartifactory) patch_netrc ;;
            gist) patch_rawfile ;;
            *) patch_config ;;
        esac
        ;;
    --status)
        show_status
        ;;
    *)
        usage
        exit 1
        ;;
esac
