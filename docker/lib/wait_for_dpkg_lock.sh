#!/bin/bash
# wait_for_dpkg_lock.sh - Block while another process holds the dpkg lock
#
# Source this, then call wait_for_dpkg_lock. Waits with visible progress and a
# bounded timeout rather than relying on apt-get's own silent, unbounded retry,
# whose "Waiting for cache lock" notice is easy to miss without a tty (e.g. a
# non-interactive SSH invocation) -- which makes a transient lock look identical
# to a genuine hang. Instant no-op when nothing holds the lock (the normal case
# for fresh containers), and also when fuser itself is missing, since a non-zero
# exit reads as "not held" and so fails open rather than stalling.
#
# Shared by env/min.sh and env/node.sh so the two can't drift apart as separate
# copies. Sourcing keeps both independently runnable: each resolves this path
# from its own location, the same way env/credentials.sh picks up
# lib/find_persistent_root.sh while still working standalone.
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
