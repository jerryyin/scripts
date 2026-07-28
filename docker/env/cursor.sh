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
#
# Rules are authored in the Claude set (claude/.claude/rules/*.md) now; the old
# cursor/.cursor/rules/*.mdc tree was migrated into it. Cursor only reads `.mdc`
# from a project's .cursor/rules/, so rules are linked in under that extension --
# verbatim, since the frontmatter both tools read (globs / description) already
# means the same thing to each.

set -e

CURSOR_RULES_SOURCE="$HOME/rc_files/claude/.claude/rules"

# Does this rule apply to a workspace? Decided from its frontmatter `globs:`, not
# its filename. Note the sense is inverted from the old .mdc convention, where
# "globs: **/*" was what marked a rule universal:
#   no globs:      general guidance (code review, style, workflow) -> everywhere
#   file-type glob e.g. "**/*.py" -> Cursor already scopes it per matched file
#   project glob   e.g. "**/triton*/**" -> only where the directory segment
#                  matches this workspace (or its base name, so a rule for
#                  "iree" still reaches the "iree-turbine" workspace)
rule_applies() {
    local rule="$1" name="$2" globs core

    globs=$(sed -n '/^---$/,/^---$/s/^globs:[[:space:]]*//p' "$rule" | head -1 | tr -d "\"'")
    [ -z "$globs" ] && return 0

    core="${globs#\*\*/}"
    core="${core%/\*\*}"

    # A pattern still carrying an extension is a file-type rule, not a
    # project-directory rule; let Cursor decide per file.
    case "$core" in
        *.*) return 0 ;;
    esac

    # shellcheck disable=SC2053  # $core is a glob here, intentionally unquoted
    [[ "$name" == $core || "${name%%-*}" == $core ]]
}

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

    local count=0
    for rule in "$CURSOR_RULES_SOURCE"/*.md; do
        [ -e "$rule" ] || continue
        rule_applies "$rule" "$pattern" || continue

        local rulename
        rulename="$(basename "$rule" .md).mdc"

        # Hardlink, so editing either path keeps the rule in sync with rc_files.
        rm -f "$rules_dest/$rulename"
        ln "$rule" "$rules_dest/$rulename"
        echo "   Linked Cursor rule: $rulename"
        count=$((count + 1))
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
