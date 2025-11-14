# Workspace Isolation for Multiple Pods

## Overview

Each pod gets its own **isolated IREE workspace** to prevent conflicts when running multiple pods simultaneously. Your dotfiles (`rc_files`, `scripts`) remain shared, but IREE development happens in pod-specific directories.

## Directory Structure

```
/home/ossci/                          ← Shared PVC
├── .iree-reference/                  ← Clean reference (never touch!)
│   └── iree/                         ← Up-to-date IREE + submodules
├── workspace-pod1/                   ← Pod 1's isolated workspace
│   └── iree/                         ← Pod 1's IREE (can have changes)
├── workspace-pod2/                   ← Pod 2's isolated workspace
│   └── iree/                         ← Pod 2's IREE (different branch, etc.)
├── iree -> workspace-pod1/iree       ← Symlink to current pod's workspace
├── rc_files/                         ← Shared (stable)
└── scripts/                          ← Shared (stable)
```

## How It Works

### First Pod Ever (Initial Setup)
```bash
~/scripts/kubernetes/interactive/connect.sh

1. Runs init_min.sh (installs git, zsh, tmux, neovim, etc.)
2. Runs init_iree.sh (installs cmake, python packages)
3. Clones repos: rc_files, scripts (to PVC)
4. Clones IREE to ~/.iree-reference/iree/     (one-time, ~10min)
5. Copies to ~/workspace-<pod>/iree/
6. Updates to latest (git reset --hard origin/main)
7. Creates symlink: ~/iree -> workspace-<pod>/iree
8. SSHs into pod
Total time: ~15-20 minutes
```

### Subsequent New Pods (Fresh Container)
```bash
~/scripts/kubernetes/interactive/connect.sh --new

1. Runs init_min.sh (installs system packages)
2. Runs init_iree.sh (installs python packages)
   - Repos already exist on PVC, so cloning is skipped
3. Copies from ~/.iree-reference/iree/        (fast, ~1-2min)
4. Creates ~/workspace-<new-pod>/iree/
5. Updates to latest
6. Creates symlink: ~/iree -> workspace-<new-pod>/iree
7. SSHs into pod
Total time: ~10-12 minutes
```

### Existing Pod
```bash
~/scripts/kubernetes/interactive/connect.sh

1. Verifies ~/workspace-<pod>/iree/ exists
2. Updates symlink: ~/iree -> workspace-<pod>/iree
3. SSHs into pod
```

### Pod Deletion
```bash
~/scripts/kubernetes/interactive/stop.sh

1. SSH into pod
2. Removes ~/workspace-<pod>/                 (cleans up workspace)
3. Removes ~/iree symlink if pointing to this workspace
4. Deletes pod
```

## What Persists vs What Gets Reinstalled

### Persists (PVC - `/home/ossci/`)
- ✅ `rc_files/`, `scripts/` (git repos)
- ✅ `.iree-reference/` (reference IREE)
- ✅ `workspace-*/` (your work)
- ✅ Dotfiles (`.zshrc`, `.tmux.conf`, etc.)
- ✅ Build artifacts, local changes

### Gets Reinstalled Every New Pod (Container Filesystem)
- 🔄 System packages (`git`, `zsh`, `tmux`, `neovim`, `cmake`)
- 🔄 Python packages (`numpy`, `pytest`, `pandas`, etc.)
- 🔄 `/usr/bin/*`, `/usr/local/*`, `/etc/*`

**Why?** When you delete Pod 1 and create Pod 2:
- Pod 2 gets a **fresh container** from `rocm/dev-ubuntu-24.04:latest`
- The **PVC mounts** to `/home/ossci/` with all your existing work
- But the **system packages** need to be reinstalled (they live in the container, not PVC)

**Result:** Init scripts run every time to set up the container, but skip cloning repos since they already exist on the PVC.

## Benefits

### ✅ Isolation
- Each pod has its own IREE directory
- Work on different branches simultaneously
- No git conflicts between pods

### ✅ Fresh Start
- New pods get updated IREE (latest main)
- No stale state from previous work

### ✅ Automatic Cleanup
- Workspaces deleted when pod is deleted
- No accumulation of old workspaces

### ✅ Shared Dotfiles
- `rc_files/`, `scripts/` remain shared
- Consistent environment across all pods

### ✅ Fast Workspace Creation
- Reference copy (~1-2 min) vs clone (~8-10 min)
- Submodules already initialized

## Usage Examples

### Scenario 1: Working on Different Features

**Pod 1:** Working on feature-A
```bash
~/scripts/kubernetes/interactive/connect.sh
# Pod: zyin-iree-20251114-140000

ssh ossci
cd ~/iree  # -> ~/workspace-zyin-iree-20251114-140000/iree/
git checkout -b feature-A
# Make changes, commit, push
```

**Pod 2:** Working on feature-B (simultaneously)
```bash
~/scripts/kubernetes/interactive/connect.sh --new
# Pod: zyin-iree-20251114-143000

ssh ossci
cd ~/iree  # -> ~/workspace-zyin-iree-20251114-143000/iree/
git checkout -b feature-B
# Make changes, commit, push
# Pod 1's work is unaffected!
```

### Scenario 2: Testing Different Builds

**Pod 1:** Testing ROCm build
```bash
cd ~/iree
cmake -B build-rocm -DIREE_HAL_DRIVER_ROCM=ON
ninja -C build-rocm
```

**Pod 2:** Testing CPU-only build
```bash
cd ~/iree
cmake -B build-cpu
ninja -C build-cpu
# Completely independent from Pod 1
```

### Scenario 3: Cleanup

```bash
~/scripts/kubernetes/interactive/stop.sh

📦 Found 2 pod(s):
  1. zyin-iree-20251114-140000
  2. zyin-iree-20251114-143000

Choose pods to delete: 1

  Cleaning workspace for zyin-iree-20251114-140000...
    Removing workspace: /home/ossci/workspace-zyin-iree-20251114-140000
  Deleting zyin-iree-20251114-140000...
✅ Deleted 1 pod(s)

# Pod 2 continues working, unaffected
```

## Reference IREE Maintenance

The `.iree-reference/iree/` is a **clean, never-modified** copy:
- Created once on first pod launch
- Updated automatically when new pods are created
- Never has local changes
- Safe to delete and recreate if needed

**To update reference manually:**
```bash
ssh ossci  # (any pod)
cd ~/.iree-reference/iree
git fetch origin
git reset --hard origin/main
git submodule update --init
```

## Troubleshooting

### Issue: Symlink points to wrong workspace

**Symptoms:** `cd ~/iree` goes to old pod's workspace

**Fix:**
```bash
ssh ossci
POD_NAME=$(hostname)
ln -sf ~/workspace-$POD_NAME/iree ~/iree
```

### Issue: Workspace not cleaned up

**Symptoms:** Old workspaces still exist after pod deletion

**Manual cleanup:**
```bash
ssh ossci  # (any active pod)
ls -d ~/workspace-*
rm -rf ~/workspace-<old-pod-name>
```

### Issue: Reference IREE corrupted

**Fix:**
```bash
ssh ossci
rm -rf ~/.iree-reference
# Next pod creation will recreate it
```

## Technical Details

### Why Copy Instead of Clone?
- **Cloning** from GitHub: ~8-10 minutes (large repo + submodules)
- **Copying** from reference: ~1-2 minutes (local filesystem)
- Reference is kept up-to-date automatically

### Why Symlink ~/iree?
- **Compatibility:** Existing scripts/docs reference `~/iree`
- **Convenience:** Type `cd ~/iree` instead of `cd ~/workspace-<long-pod-name>/iree`
- **Clarity:** Symlink always points to current pod's workspace

### Disk Space Considerations
- Reference: ~10GB
- Each workspace: ~10GB
- 3 pods = ~40GB total
- PVC size should be 50GB+ for comfortable multi-pod usage
