#!/bin/bash
# npm_global_bin.sh - Put npm's global bin directory on PATH
#
# Source this, then call use_npm_global_bin. The CLI installers (env/claude.sh,
# env/codex.sh) set npm's prefix themselves and then verify the install by
# running the binary they just placed. That verification resolves the command
# through PATH, but min.sh invokes them from a non-interactive shell whose PATH
# is the system default: the rc files that add npm's bin directory are
# login/interactive-only, and Ubuntu's stock ~/.bashrc returns early when not
# interactive. So on a fresh host a perfectly good install reported "may have
# silently failed" -- the binary was there, the bare `claude --version` that
# checked it was not resolvable.
#
# Ask npm where it installs rather than assuming ~/.local/bin, so this still
# holds if the prefix is ever configured somewhere else.
#
# Deliberately not gated on the directory existing: callers run this *before*
# `npm install -g`, and on a fresh host npm only creates the bin dir during that
# install -- gating here left PATH unchanged and the post-install check failed.
use_npm_global_bin() {
    local prefix bin
    prefix="$(npm prefix -g 2>/dev/null)"
    [ -n "$prefix" ] || return 0
    bin="$prefix/bin"

    case ":$PATH:" in
        *":$bin:"*) ;;
        *) PATH="$bin:$PATH"; export PATH ;;
    esac
}
