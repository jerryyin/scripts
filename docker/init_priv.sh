#!/bin/sh
set -x

# clone the configuration to root
if [ ! -f ~/.ssh/id_rsa ]; then
    rm -rf ~/.ssh
    # Note: Docker needs to mount home directory to /zyin
    cp -r /zyin/.ssh ~/.ssh
    chmod 700 ~/.ssh 2>/dev/null || true
    chmod 600 ~/.ssh/id_rsa 2>/dev/null || true
    ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts || true
fi

# Fixing /etc/host file, refer to https://askubuntu.com/questions/59458/error-message-sudo-unable-to-resolve-host-none
if ! grep -q "$HOSTNAME" /etc/hosts; then
    echo $(hostname -I | cut -d\  -f1) $(hostname) | sudo -h 127.0.0.1 tee -a /etc/hosts
fi

exit 0
