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

# Machine and project prerequisites. Configured applications belong to rc_files.
set -e
wait_for_dpkg_lock
sudo DEBIAN_FRONTEND=noninteractive apt-get install -f -y \
     git python3 python-is-python3 less wget gist python3-pip

# Clone once, then rerun the repository's self-contained installer to repair
# incomplete deployment. Correct links are left intact; local conflicts stop setup.
if [ ! -d "$HOME/rc_files" ]; then
    git clone https://github.com/jerryyin/rc_files.git "$HOME/rc_files"
    git -C "$HOME/rc_files" remote set-url origin git@github.com:jerryyin/rc_files.git
fi

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

# rc_files owns application dependencies, configuration and plugins.
python3 "$HOME/rc_files/.setup"
