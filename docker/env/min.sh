#!/bin/bash
set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# On a freshly-imaged/rebooted bare-metal host, systemd's automatic update
# timers (unattended-upgrades, apt-daily) can be mid-run and hold the dpkg
# lock for a long time. apt-get's own "Waiting for cache lock" retry is easy
# to miss when not attached to a tty (e.g. a non-interactive SSH invocation),
# which makes a transient lock look identical to a genuine hang. Containers
# have no systemd (command -v systemctl / /run/systemd/system both absent),
# so this is a no-op there -- it only kicks in on real hosts.
if command -v systemctl >/dev/null 2>&1 && [ -d /run/systemd/system ]; then
    sudo systemctl stop apt-daily.service apt-daily-upgrade.service \
        apt-daily.timer apt-daily-upgrade.timer unattended-upgrades.service \
        2>/dev/null || true
fi

# Wait (with visible progress + a bounded timeout) for any in-progress apt/dpkg
# run to release the lock, instead of relying on apt-get's own silent,
# unbounded retry. Safe/instant no-op when nothing holds the lock (the normal
# case for fresh containers).
wait_for_dpkg_lock() {
    local waited=0
    local max_wait=600
    while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
        if [ "$waited" -ge "$max_wait" ]; then
            echo "⚠️  dpkg lock still held after ${max_wait}s -- proceeding anyway, apt may fail" >&2
            return 1
        fi
        echo "⏳ Waiting for dpkg lock (held by another process, e.g. a background apt/unattended-upgrade run)... ${waited}s elapsed"
        sleep 10
        waited=$((waited + 10))
    done
    return 0
}

# Delete rocm sources if any, they tend to cause problem with apt update
#find /etc/apt \( -name "*amdgpu*" -o -name "*rocm*" \) -delete
# Re-import gpg key to not have warnings all over the place
# --batch --no-tty: non-interactive invocations (e.g. this script run from an
# IDE/CI agent with no controlling terminal) otherwise hit
# "gpg: cannot open '/dev/tty': No such device or address" as gpg tries to
# open a tty for its default (interactive) status/prompt path.
curl -fsSL --max-time 30 https://repo.radeon.com/rocm/rocm.gpg.key | sudo gpg --batch --yes --no-tty --dearmor -o /etc/apt/trusted.gpg.d/rocm.gpg

wait_for_dpkg_lock
# Deliberately not `update && install`: a fetch error from some unrelated
# repo (e.g. a stale AMD mirror pin) can make `apt-get update` exit non-zero
# even though the indices this install actually needs are fine. Chaining on
# that exit code would silently skip installing sudo/curl/etc. Run update
# best-effort, then always attempt the install against whatever cache we do
# have -- same pattern as every other apt call below.
sudo apt-get update
sudo apt-get -y install sudo software-properties-common apt-utils curl

# Fixing /etc/host file, refer to https://askubuntu.com/questions/59458/error-message-sudo-unable-to-resolve-host-none
if ! grep -q "$HOSTNAME" /etc/hosts; then
    echo $(hostname -I | cut -d\  -f1) $(hostname) | sudo tee -a /etc/hosts
fi

shopt -s expand_aliases
add_ppa_if_available() {
    local ppa="$1"
    local codename
    codename=$(lsb_release -cs 2>/dev/null || { . /etc/os-release && echo "$VERSION_CODENAME"; })
    local ppa_path=${ppa#ppa:}
    local url="https://ppa.launchpadcontent.net/${ppa_path}/ubuntu/dists/${codename}/Release"
    if timeout 15 wget -q --spider "$url" 2>/dev/null; then
        wait_for_dpkg_lock
        sudo add-apt-repository -y "$ppa"
    else
        echo "Warning: $ppa not available for ${codename}, skipping..." >&2
    fi
}

wait_for_dpkg_lock
sudo apt-get update --allow-insecure-repositories

# Get Ubuntu codename (focal, jammy, noble, etc.)
CODENAME=$(lsb_release -sc 2>/dev/null || . /etc/os-release && echo "$VERSION_CODENAME")

add_ppa_if_available ppa:jonathonf/vim
add_ppa_if_available ppa:neovim-ppa/stable

wait_for_dpkg_lock
# Install misc pkgs (For macos: the_silver_searcher)
sudo DEBIAN_FRONTEND=noninteractive apt-get install -f -y  \
     git zsh fonts-powerline tmux silversearcher-ag less stow neovim vim wget \
     python-is-python3 gdb gist openssh-client

# rc files. Clone once, but ALWAYS re-run install.sh: it is idempotent (stow -R,
# backs up real-file conflicts) and is what re-heals a partial prior setup — e.g.
# a pod reusing persistent $HOME where rc_files/ exists but a previous install.sh
# died midway. Gating install.sh on the directory would skip that repair forever.
if [ ! -d rc_files ]; then
    git clone https://github.com/jerryyin/rc_files.git
    git -C rc_files remote set-url origin git@github.com:jerryyin/rc_files.git
fi

# Everything Node-related (NodeSource repo setup, version guard, the
# apt-fetch-error shielding workaround, and the final version verification)
# lives in node.sh so it can be reasoned about/tested on its own. Must run
# before rc_files/install.sh below: its coc.nvim extensions step checks
# `command -v npm` and silently skips (soft warning, not a failure) if node
# isn't installed yet.
bash "$SCRIPT_DIR/node.sh"

# install.sh, claude.sh, and codex.sh each self-sufficiently source
# rc_files/lib/node-ca-cert.sh (Node's corporate-TLS-proxy CA workaround)
# right before they run npm, so nothing needs to be pre-sourced here.
bash rc_files/install.sh

# Clone scripts
if [ ! -d scripts ]; then
    git clone https://github.com/jerryyin/scripts.git
    git -C scripts remote set-url origin git@github.com:jerryyin/scripts.git
fi

# Only download AMD_CA.crt if not already present (PVC persists across pods)
if [ ! -f /usr/local/share/ca-certificates/AMD_CA.crt ]; then
    wget -q --timeout=30 -O /tmp/AMD_CA.crt https://gist.githubusercontent.com/jerryyin/8da8f21024b5d4ef853b171771def28c/raw/d769419709807acfeff1a3c7a7f4acbc44b76b28/AMD_CA.crt
    sudo cp /tmp/AMD_CA.crt /usr/local/share/ca-certificates
    sudo update-ca-certificates
    rm -f /tmp/AMD_CA.crt  # Clean up downloaded file
else
    echo "AMD_CA.crt already installed, skipping"
fi

wait_for_dpkg_lock
sudo apt-get install -y locales && sudo locale-gen en_US.UTF-8

# Sibling scripts in the same env/ dir (SCRIPT_DIR set above): install CLIs
# and best-effort config. priv.sh handles runtime credential/config sync
# once host storage is mounted.
bash "$SCRIPT_DIR/claude.sh"
bash "$SCRIPT_DIR/codex.sh"
bash "$SCRIPT_DIR/gh.sh"

# Refresh Cursor AI assistant rules for any project workspaces that already
# exist under $HOME (e.g. reused persistent storage). min.sh is the only
# caller of cursor.sh -- see env/cursor.sh for why.
bash "$SCRIPT_DIR/cursor.sh"
