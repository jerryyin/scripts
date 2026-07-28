#!/bin/bash
# fix_hostname.sh - Make $HOSTNAME resolvable so sudo stops warning
#
# Source this, then call fix_hostname. Without a matching /etc/hosts entry every
# sudo call prints "unable to resolve host"
# (https://askubuntu.com/questions/59458/error-message-sudo-unable-to-resolve-host-none).
# Shared by env/min.sh and env/priv.sh so both agree instead of drifting as
# separate copies -- in particular on skipping the append when no IP is assigned
# yet, which would otherwise write an address-less entry and break the very
# lookup this exists to fix.
fix_hostname() {
    if [ -z "${HOSTNAME:-}" ]; then
        return 0
    fi

    if grep -q "$HOSTNAME" /etc/hosts 2>/dev/null; then
        return 0
    fi

    local ip
    ip=$(hostname -I 2>/dev/null | cut -d' ' -f1)
    if [ -z "$ip" ]; then
        return 0
    fi

    echo "🔧 Adding $HOSTNAME to /etc/hosts..."
    # Tolerate failure: a container start (priv.sh) must not abort just because
    # /etc/hosts isn't writable here.
    echo "$ip $HOSTNAME" | sudo tee -a /etc/hosts >/dev/null 2>&1 || true
}
