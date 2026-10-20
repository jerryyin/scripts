#!/bin/bash
# find_persistent_root.sh - Locate mounted persistent storage (host home or PVC)
#
# Source this, then call find_persistent_root; echoes the root path, or
# nothing if none is found. Shared by credentials/activate.sh and
# lib/git_workspace.sh so all three agree on where persistent storage lives
# instead of drifting out of sync as separate copies.
find_persistent_root() {
    # Launchers provide the storage location. An explicitly empty value disables
    # persistence (image builds and ephemeral pods); omission keeps legacy discovery.
    if [[ ${DEVENV_PERSISTENT_ROOT+x} ]]; then
        if [[ -n "$DEVENV_PERSISTENT_ROOT" && ! -d "$DEVENV_PERSISTENT_ROOT" ]]; then
            echo "Persistent storage does not exist: $DEVENV_PERSISTENT_ROOT" >&2
            return 1
        fi
        printf '%s\n' "$DEVENV_PERSISTENT_ROOT"
        return 0
    fi
    local username
    username=$(id -un 2>/dev/null || whoami 2>/dev/null || echo "")

    # Strategy 1: Try username-based paths (Kubernetes pattern: /{username})
    if [ -n "$username" ] && [ -d "/$username" ] && [ "/$username" != "/root" ] && [ "/$username" != "/home" ]; then
        echo "/$username"
        return 0
    fi

    # Strategy 2: Check for common Docker mount patterns
    for candidate in /zyin /host_home /persistent; do
        if [ -d "$candidate/.ssh" ] || [ -d "$candidate/rc_files" ]; then
            echo "$candidate"
            return 0
        fi
    done

    # Strategy 3: Scan root-level directories for home-like markers
    # (mounted home dirs typically have .bashrc, .ssh, rc_files, scripts, etc.)
    for candidate in /*/; do
        candidate="${candidate%/}"
        case "$candidate" in
            /bin|/boot|/dev|/etc|/home|/lib*|/media|/mnt|/opt|/proc|/root|/run|/sbin|/srv|/sys|/tmp|/usr|/var|/data|/ffm|/tools|/snap)
                continue
                ;;
        esac
        if [ -d "$candidate/.ssh" ] || [ -d "$candidate/rc_files" ] || [ -d "$candidate/scripts" ]; then
            echo "$candidate"
            return 0
        fi
    done

    echo ""
}
