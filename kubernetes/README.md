# Kubernetes Configuration for IREE Development

Self-contained Kubernetes configuration for connecting to TensorWave clusters. All files organized in one place, ready for version control.

## Directory Structure

```
kubernetes/
├── kube-configs/                   # Kubernetes configurations
│   ├── switch-config.yaml          # kubeswitch configuration
│   └── tw-tus1-bm-private-sso.conf # Cluster config with Okta SSO
├── pvc/                            # PersistentVolumeClaim
│   └── iree-dev-zyin-pvc.yaml      
├── interactive/                    # Interactive pod scripts
│   ├── config.json                 # Your configuration
│   ├── connect.sh                  # Connect to pod
│   ├── stop.sh                     # Cleanup pods
│   ├── pod-ssh.yml                 # Pod template (SSH)
│   ├── pod-web.yml                 # Pod template (web)
│   └── ssh-config.txt              # SSH config reference
└── setup-symlinks.sh               # Setup script for new systems
```

---

## Use Case 1: Setup Connection (First Time)

### Prerequisites

Install required tools (WSL Ubuntu or Linux):

```bash
# kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# krew (kubectl plugin manager)
(
  set -x; cd "$(mktemp -d)" &&
  OS="$(uname | tr '[:upper:]' '[:lower:]')" &&
  ARCH="$(uname -m | sed -e 's/x86_64/amd64/' -e 's/\(arm\)\(64\)\?.*/\1\2/' -e 's/aarch64$/arm64/')" &&
  KREW="krew-${OS}_${ARCH}" &&
  curl -fsSLO "https://github.com/kubernetes-sigs/krew/releases/latest/download/${KREW}.tar.gz" &&
  tar zxvf "${KREW}.tar.gz" &&
  ./"${KREW}" install krew
)

# Add to ~/.bashrc or ~/.zshrc:
export PATH="${KREW_ROOT:-$HOME/.krew}/bin:$PATH"

# oidc-login (Okta SSO authentication)
kubectl krew install oidc-login

# kubeswitch (cluster switching)
sudo curl -L -o /usr/local/bin/switcher https://github.com/danielfoehrKn/kubeswitch/releases/latest/download/switcher_linux_amd64
sudo chmod a+x /usr/local/bin/switcher

# Add to ~/.bashrc or ~/.zshrc:
source <(switcher init bash)  # or 'zsh'
```

### Setup Steps

```bash
# 1. Run setup script (creates symlinks)
cd ~/scripts/kubernetes
./setup-symlinks.sh

# 2. Create PVC (first time only)
kubectl apply -f ~/scripts/kubernetes/pvc/iree-dev-zyin-pvc.yaml

# 3. Connect! (handles everything automatically)
~/scripts/kubernetes/interactive/connect.sh
# If not authenticated, it will automatically:
# - Open browser at http://localhost:8000 for Okta login
# - Wait for you to complete sign-in
# - Continue with pod connection
```

**That's it!** The script handles kubeconfig, authentication, everything automatically.

**Cluster Selection:** The script uses `tw-tus1-bm-private-sso.conf` by default (set in `config.json`). To use a different cluster, either:
- Edit `config.json` to change the default
- Run `switch` before `connect.sh` (script respects your selection)

**Note:** The setup script creates these symlinks:
- `~/.kube/switch-config.yaml` → `~/scripts/kubernetes/kube-configs/switch-config.yaml`
- `~/.kube/configs/tw-tus1-bm-private-sso.conf` → `~/scripts/kubernetes/kube-configs/tw-tus1-bm-private-sso.conf`

**SSH Config:** The setup script adds this to `~/.ssh/config`:
```
Host ossci
  HostName 127.0.0.1
  User ossci
  Port 2222
  IdentityFile ~/.ssh/id_rsa
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
```

### Remote Machine Setup

If running on a remote machine with browser on local machine:

```bash
# On local machine (for Okta redirect)
ssh -L 8000:localhost:8000 <remote-host>

# Then run kubectl get ns on remote machine
```

---

## Use Case 2: Setup on New System

Copy configuration and run setup script:

```bash
# Copy from old system
scp -r old-machine:~/scripts/kubernetes ~/scripts/

# Or clone from git (if you pushed it)
git clone <your-repo> ~/scripts

# Run setup
cd ~/scripts/kubernetes
./setup-symlinks.sh

# Create PVC (first time only)
kubectl apply -f ~/scripts/kubernetes/pvc/iree-dev-zyin-pvc.yaml

# Connect (handles everything automatically)
~/scripts/kubernetes/interactive/connect.sh
# First time: browser opens for Okta login
# After that: direct connection

# When done, cleanup
~/scripts/kubernetes/interactive/stop.sh
```

---

## Daily Workflow

```bash
# 1. Connect to interactive pod (ONE command does everything!)
~/scripts/kubernetes/interactive/connect.sh
# Script automatically:
# - Sets kubeconfig
# - Authenticates (opens browser if needed)
# - Attaches to existing pod if available
# - Creates new pod if none exist
# - SSH's you into the pod

# To start a second/third pod (for parallel work):
~/scripts/kubernetes/interactive/connect.sh --new

# 2. Work in the pod
# (you're already connected)

# 3. Exit when done (pod keeps running)
exit

# 4. To reconnect later
ssh ossci

# 5. To stop and cleanup (when completely done)
~/scripts/kubernetes/interactive/stop.sh
```

**Notes:** 
- The script handles everything automatically (cluster selection, authentication, pod creation)
- Default cluster: `tw-tus1-bm-private-sso.conf` (configurable in `config.json`)
- To use different cluster: run `switch` first, or edit `config.json`

---

## Configuration

### Cluster Selection

The `connect.sh` script automatically handles cluster selection with smart defaults:

**Default Cluster:**
- Set in `config.json` as `"default_cluster": "tw-tus1-bm-private-sso.conf"`
- Used automatically when you run `connect.sh`

**How to Use a Different Cluster:**

```bash
# Option 1: Edit config.json (permanent change)
vim ~/scripts/kubernetes/interactive/config.json
# Change: "default_cluster": "tw-tus1-bm-private-sso.conf"
# To:     "default_cluster": "your-other-cluster.conf"

# Option 2: Use switch (one-time override)
switch  # Select different cluster
~/scripts/kubernetes/interactive/connect.sh
# Script respects your KUBECONFIG selection

# Option 3: Set KUBECONFIG manually
export KUBECONFIG=~/.kube/configs/other-cluster.conf
~/scripts/kubernetes/interactive/connect.sh
```

**Smart Behavior:**
- If `KUBECONFIG` is already set (from `switch` or manual export) → uses that
- If not set → uses `default_cluster` from `config.json`
- You can still use `switch` if you want, but you don't have to!

### Edit Your Settings

```bash
# Change namespace, PVC name, SSH key path, or default cluster
vim ~/scripts/kubernetes/interactive/config.json

# Modify PVC size or storage class
vim ~/scripts/kubernetes/pvc/iree-dev-zyin-pvc.yaml
kubectl apply -f ~/scripts/kubernetes/pvc/iree-dev-zyin-pvc.yaml

# Customize pod resources (GPU count, image, etc.)
vim ~/scripts/kubernetes/interactive/pod-ssh.yml
```

### Interactive Pod Scripts

**connect.sh** - One-command connection (truly automatic):
- Automatically sets kubeconfig
- Authenticates via Okta (opens browser if needed)
- Attaches to existing pod if available
- Creates new pod if none exist
- Starts port-forward in background
- SSH's you into the pod

```bash
# SSH mode (default) - just run this!
~/scripts/kubernetes/interactive/connect.sh
# Attaches to existing pod if available

# Create a NEW pod (even if others exist)
~/scripts/kubernetes/interactive/connect.sh --new
# Useful for running multiple pods simultaneously

# Web mode (browser-based code-server)
~/scripts/kubernetes/interactive/connect.sh web
# Access: http://localhost:8000
```

**Multiple Pods:**
- By default, script attaches to existing pod (reuses your environment)
- Use `--new` flag to create additional pods for parallel work
- Each pod gets unique name: `${USER}-iree-YYYYMMDD-HHMMSS`

**Reconnect:**
```bash
ssh ossci
```

**stop.sh** - Selective cleanup:
```bash
~/scripts/kubernetes/interactive/stop.sh
# Interactive menu to choose which pods to delete:
# - a: Delete all pods
# - 1,2,3: Delete specific pods by number
# - q: Quit without deleting
```

**Example:**
```
📦 Found 3 pod(s):
  1. zyin-iree-20251114-100000
  2. zyin-iree-20251114-110000
  3. zyin-iree-20251114-120000

Options:
  a     - Delete ALL pods
  1-3   - Delete specific pod
  1,3   - Delete multiple pods
  q     - Quit

Choose pods to delete: 2
```

---

## Common Commands

**Note:** `connect.sh` automatically sets up kubeconfig (default: `tw-tus1-bm-private-sso.conf`), so you can just run it directly!

```bash
# Connect to pod (handles everything, uses default cluster)
~/scripts/kubernetes/interactive/connect.sh

# To use different cluster temporarily:
switch  # Select your cluster
~/scripts/kubernetes/interactive/connect.sh

# If you need to run kubectl commands manually:
export KUBECONFIG=~/.kube/configs/tw-tus1-bm-private-sso.conf
kubectl get ns

# View resources in your namespace
kubectl get all -n iree-dev

# Check PVC status
kubectl get pvc -n iree-dev

# List running pods
kubectl get pods -n iree-dev

# View pod logs
kubectl logs <pod-name> -n iree-dev

# Check current context
kubectl config current-context
```

---

## Troubleshooting

### kubectl Commands Not Working

If kubectl commands fail with "connection refused" or can't find the server:

```bash
# Make sure you've selected the cluster first!
switch
# Select: tw-tus1-bm-private-sso.conf

# Then try kubectl again
kubectl get ns
```

### Authentication Issues

```bash
# Select cluster first
switch

# Clear cached tokens
rm -rf ~/.kube/cache/oidc-login

# Re-authenticate
kubectl get ns
```

### Pod Issues

```bash
# Check pod status
kubectl get pods -n iree-dev

# View pod details
kubectl describe pod <pod-name> -n iree-dev

# Check pod logs
kubectl logs <pod-name> -n iree-dev
```

---

## File Locations Reference

| Description | Path |
|-------------|------|
| Kubeconfig | `~/.kube/configs/tw-tus1-bm-private-sso.conf` (symlink) |
| PVC definition | `~/scripts/kubernetes/pvc/iree-dev-zyin-pvc.yaml` |
| Connect script | `~/scripts/kubernetes/interactive/connect.sh` |
| Stop script | `~/scripts/kubernetes/interactive/stop.sh` |
| Configuration | `~/scripts/kubernetes/interactive/config.json` |
| SSH config | `~/.ssh/config` (has `ossci` host entry) |

---

## References

- Kubeswitch: https://github.com/danielfoehrKn/kubeswitch
- OIDC Login: https://github.com/int128/kubelogin
