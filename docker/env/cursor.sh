#!/bin/bash
# cursor.sh - Cursor AI assistant rules setup
#
# Owns Cursor rule linking as its own concern (like claude.sh/codex.sh own
# their CLIs), rather than it being a hidden side effect of project workspace
# creation. min.sh is the only caller -- it is the universal entry point that
# every service's setup runs through, so this is the one place rule-linking
# happens. Workspace scripts (workspace/triton.sh, workspace/iree.sh) do NOT
# call setup_cursor_rules themselves; a workspace cloned after this run's
# min.sh step picks up rules on the next cursor.sh/min.sh run.
#
# Usage:
#   cursor.sh   # Refresh rules for every project workspace that already
#                 exists under $HOME (safe no-op if none do yet)
#
# Rules live in rc_files (stow-managed), so the source directory below is
# populated by rc_files/install.sh, not by this script.

set -e

CURSOR_RULES_SOURCE="$HOME/rc_files/cursor/.cursor/rules"

# Link Cursor rules into a workspace based on their globs pattern:
# **/* = universal, otherwise match project name.
# Uses hardlinks for better compatibility with Cursor's file watching.
setup_cursor_rules() {
    local pattern="$1"
    local workspace_dir="$2"

    if [ -z "$pattern" ] || [ -z "$workspace_dir" ]; then
        return 1
    fi

    if [ ! -d "$CURSOR_RULES_SOURCE" ]; then
        echo "   ℹ️  No Cursor rules source found"
        return 0
    fi

    local rules_dest="$workspace_dir/.cursor/rules"
    mkdir -p "$rules_dest"

    # Extract base project name (first segment before hyphen)
    # e.g., "triton-mi450" -> "triton", "iree-turbine" -> "iree"
    local base_pattern="${pattern%%-*}"

    local count=0
    for rule in "$CURSOR_RULES_SOURCE"/*.mdc; do
        [ -e "$rule" ] || continue
        local rulename
        rulename=$(basename "$rule")

        # Check if rule applies:
        # 1. Universal glob (**/*) matches all projects
        # 2. Rule filename contains project name (e.g., "iree" in "iree-turbine.mdc")
        # 3. Rule filename contains base project name (e.g., "triton" matches "triton-ffm-development.mdc")
        if grep -q '^globs:.*\*\*/\*' "$rule" || \
           [[ "$rulename" == *"$pattern"* ]] || \
           [[ "$rulename" == *"$base_pattern"* ]]; then
            # Use hardlink for better Cursor compatibility (removes existing first)
            rm -f "$rules_dest/$rulename"
            ln "$rule" "$rules_dest/$rulename"
            echo "   Linked Cursor rule: $rulename"
            count=$((count + 1))
        fi
    done

    if [ "$count" -eq 0 ]; then
        echo "   ℹ️  No applicable Cursor rules found for '$pattern'"
    fi
}

# Refresh rules for every project workspace directory that already exists
# under $HOME. This is what makes running cursor.sh directly (e.g. from
# min.sh, before any workspace/*.sh has run) worthwhile: on a fresh container
# there's nothing to do yet, but on a reused persistent $HOME (existing pod
# restart, re-provisioned host), it re-syncs rules for whatever is already
# cloned instead of requiring a re-clone to pick up rule changes.
refresh_all_workspaces() {
    local found=0
    local dir name
    for dir in "$HOME"/*/; do
        dir="${dir%/}"
        name=$(basename "$dir")
        case "$name" in
            rc_files|scripts|vault) continue ;;
        esac
        [ -d "$dir/.git" ] || continue

        echo "   Refreshing rules for ~/$name"
        setup_cursor_rules "$name" "$dir"
        found=1
    done

    if [ "$found" -eq 0 ]; then
        echo "   ℹ️  No existing project workspaces under \$HOME yet"
    fi
}

main() {
    case "${1:-}" in
        "") ;;
        *)
            echo "Usage: cursor.sh"
            echo "  Refresh Cursor rules for any project workspaces that already exist under \$HOME"
            exit 1
            ;;
    esac

    echo ""
    echo "📐 Cursor Rules Setup"
    echo "─────────────────────"
    refresh_all_workspaces
    echo ""
}

main "$@"
