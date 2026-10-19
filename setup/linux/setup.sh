#!/bin/bash
# Public Linux setup interface. Profiles compose installers; launchers select phases.
set -euo pipefail

DEVENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE=base
PHASE=all
PLAN=false
PROFILE_SET=false

usage() {
    cat <<'EOF'
Usage: setup.sh [PROFILE] [--phase all|build|install|activate|workspace] [--plan]
       setup.sh --list

  all        Bootstrap, activate credentials, install project dependencies, prepare workspaces.
  build      Install and prepare workspaces, with persistence and activation disabled.
  install    Install base tools and the selected project's dependencies.
  activate   Prepare credentials only; no package installation.
  workspace  Prepare the selected project's workspaces only.
  --plan     Print the selected commands without executing them.

DEVENV_PERSISTENT_ROOT: persistent storage location; empty disables discovery.
DEVENV_WORKSPACE_ROOT: checkout parent directory (default: HOME).
EOF
}

while (( $# )); do
    case "$1" in
        --phase)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            PHASE="$2"; shift 2 ;;
        --plan|--dry-run) PLAN=true; shift ;;
        --list)
            printf '%s\n' base
            for profile_file in "$DEVENV_DIR"/profiles/*/profile.sh; do
                [[ -f "$profile_file" ]] || continue
                profile_name="${profile_file%/profile.sh}"
                printf '%s\n' "${profile_name##*/}"
            done
            exit 0 ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "Unknown option: $1" >&2; exit 2 ;;
        *)
            if "$PROFILE_SET"; then usage >&2; exit 2; fi
            PROFILE="$1"; PROFILE_SET=true; shift ;;
    esac
done

case "$PHASE" in
    all|build|install|activate|workspace) ;;
    *) echo "Unknown phase: $PHASE" >&2; exit 2 ;;
esac
if [[ ! "$PROFILE" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] ||
   [[ "$PROFILE" != base && ! -f "$DEVENV_DIR/profiles/$PROFILE/profile.sh" ]]; then
    echo "Unknown profile: $PROFILE (use --list)" >&2
    exit 2
fi

INSTALL_STEPS=()
WORKSPACE_STEPS=()
# Base uses only the shared phases. Environment plans live beside their scripts.
if [[ "$PROFILE" != base ]]; then
    # shellcheck disable=SC1090
    source "$DEVENV_DIR/profiles/$PROFILE/profile.sh"
fi
STEPS=()
case "$PHASE" in
    install|all|build) STEPS+=(bootstrap.sh) ;;
esac
case "$PHASE" in
    activate|all) STEPS+=(credentials/activate.sh) ;;
esac
case "$PHASE" in
    install|all|build) STEPS+=("${INSTALL_STEPS[@]}") ;;
esac
case "$PHASE" in
    workspace|all|build) STEPS+=("${WORKSPACE_STEPS[@]}") ;;
esac

# Validate the entire plan before the first installer makes changes.
for step in "${STEPS[@]}"; do
    [[ -f "$DEVENV_DIR/$step" ]] || { echo "Missing setup step: $step" >&2; exit 1; }
done
if [[ "$PHASE" == build ]]; then
    export DEVENV_PERSISTENT_ROOT=""
fi
for step in "${STEPS[@]}"; do
    if "$PLAN"; then
        printf 'bash %q\n' "$DEVENV_DIR/$step"
    else
        bash "$DEVENV_DIR/$step"
    fi
done
