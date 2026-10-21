#!/bin/bash
# sync_checkout DIR [URL] - Clone a missing checkout; otherwise fast-forward it to
# its upstream (or the remote default branch when the upstream is gone), carrying
# local edits across in a stash. Upstream wins a conflict and the edits stay in
# `git stash list`. Failures to reach the remote only warn.

sync_checkout() {
    local dir="$1" url="${2:-}" target stashed=0
    if [ ! -d "$dir/.git" ]; then
        [ -n "$url" ] || return 0
        if [ -e "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
            echo "Refusing to replace non-repository directory: $dir" >&2
            return 1
        fi
        git clone "$url" "$dir"
        return
    fi
    if ! git -C "$dir" fetch --quiet --prune origin; then
        echo "WARN $dir: cannot reach origin; using it as is"
        return 0
    fi
    target=$(git -C "$dir" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null) || target=
    if [ -z "$target" ] || ! git -C "$dir" rev-parse --verify --quiet "$target" >/dev/null; then
        git -C "$dir" remote set-head origin --auto >/dev/null 2>&1 || true
        target=$(git -C "$dir" symbolic-ref --quiet --short refs/remotes/origin/HEAD) || target=
    fi
    if [ -z "$target" ]; then
        echo "WARN $dir: no upstream branch; using it as is"
        return 0
    fi
    if [ -n "$(git -C "$dir" status --porcelain)" ]; then
        git -C "$dir" stash push --include-untracked --quiet -m 'bootstrap sync'
        stashed=1
    fi
    if git -C "$dir" merge-base --is-ancestor HEAD "$target"; then
        git -C "$dir" checkout --quiet -B "${target#*/}" --track "$target"
    else
        echo "WARN $dir: local commits are not on $target; branch left as is"
    fi
    if [ "$stashed" = 1 ] && ! git -C "$dir" stash pop --quiet; then
        git -C "$dir" reset --hard --quiet
        echo "WARN $dir: local edits conflict with $target; kept in git stash list"
    fi
    echo "OK $dir at $target"
}
