# Workspace storage in SSH pods

Both SSH pod templates give `/home/ossci` a pod-local `emptyDir`. A regular pod
also mounts the configured PVC at `/<local-user>`; an ephemeral pod has no PVC.

```text
/home/ossci/                      Pod-local home (emptyDir)
├── scripts/                     Copied scripts checkout
├── rc_files/                    Configuration checkout
├── iree/ or triton/              Independent working checkout
└── ...

/<local-user>/                   PVC, regular pods only
├── .iree-reference/iree/         Reusable reference checkout
├── .triton-reference/triton/     Reusable reference checkout
├── vault/                       Persistent credentials, when provided
└── ...
```

`interactive/connect.sh` chooses an image and setup profile from `config.json`.
When setup is needed, it copies the setup code and runs
`setup/linux/setup.sh PROFILE` inside the pod. This bootstraps tools, activates
available credentials, installs project dependencies, and prepares the workspace.

The connector supplies `DEVENV_PERSISTENT_ROOT=/<local-user>` for a regular pod
and an empty value for an ephemeral pod. With persistence, the workspace helper
creates or refreshes a reference checkout on the PVC and copies it into the
pod-local home. Without persistence, it clones the public upstream directly.
Existing working checkouts are reused without resetting local edits.

Each pod therefore has an independent working checkout. These are ordinary
checkout directories, not `workspace-<pod>` directories or shared `~/iree` symlinks.
The shared reference cache is managed by the helper; keep personal edits in the
working checkout.

Deleting a pod deletes its `emptyDir`, including unpushed work in `/home/ossci`.
The PVC and reference caches survive. Commit and push work, or copy it to durable
storage, before stopping a pod. The connector does not back up work automatically.

For manual setup, `DEVENV_WORKSPACE_ROOT` can redirect IREE, Triton, and TokenSpeed
checkouts. The default remains `$HOME`. See [Linux setup](../../setup/linux/README.md)
for phases and storage settings. The web pod has its own template and does not run
this SSH setup sequence.
