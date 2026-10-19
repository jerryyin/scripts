#!/bin/bash
set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

. "$SCRIPT_DIR/lib/wait_for_dpkg_lock.sh"
. "$SCRIPT_DIR/lib/fix_hostname.sh"

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

fix_hostname

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

add_ppa_if_available ppa:jonathonf/vim
add_ppa_if_available ppa:neovim-ppa/stable

# Repository probes above are best-effort. From here onward every installer is
# required: a failed dotfile/CLI setup must fail the enclosing Docker build too.
set -e

wait_for_dpkg_lock
# Install misc pkgs (For macos: the_silver_searcher)
# gcc + Python headers/setuptools: LeaderF's vim-plug `do` hook (run by
# python3 ~/rc_files/.setup) compiles its fuzzyMatch C extension. Fresh Ubuntu
# also needs pip for the project installers; ROCm/PyTorch images can mask this
# missing prerequisite by providing it already.
sudo DEBIAN_FRONTEND=noninteractive apt-get install -f -y  \
     git zsh fonts-powerline tmux silversearcher-ag less neovim vim wget \
     python-is-python3 gdb gist openssh-client gcc python3-dev python3-pip python3-setuptools

# Clone once, then rerun the repository's self-contained installer to repair
# incomplete deployment. Correct links are left intact; local conflicts stop setup.
if [ ! -d "$HOME/rc_files" ]; then
    git clone https://github.com/jerryyin/rc_files.git "$HOME/rc_files"
    git -C "$HOME/rc_files" remote set-url origin git@github.com:jerryyin/rc_files.git
fi

# Everything Node-related (NodeSource repo setup, version guard, the
# apt-fetch-error shielding workaround, and the final version verification)
# lives in node.sh so it can be reasoned about/tested on its own. Must run
# before python3 ~/rc_files/.setup below: CoC extension setup requires npm.
bash "$SCRIPT_DIR/packages/node.sh"

# Vim's package setup uses the system CA bundle for npm. CLI installers source
# ~/.config/node/ca-cert.sh themselves when they need it.
python3 "$HOME/rc_files/.setup"
. "$SCRIPT_DIR/lib/login_shell.sh"
configure_login_shell

# Clone scripts
if [ ! -d "$HOME/scripts" ]; then
    git clone https://github.com/jerryyin/scripts.git "$HOME/scripts"
    git -C "$HOME/scripts" remote set-url origin git@github.com:jerryyin/scripts.git
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

# Install CLIs and best-effort config. credentials/activate.sh handles sync
# once host storage is mounted.
bash "$SCRIPT_DIR/packages/claude.sh"
bash "$SCRIPT_DIR/packages/codex.sh"
bash "$SCRIPT_DIR/packages/gh.sh"
