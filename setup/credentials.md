# Separate work and homelab credentials

`dbuild` and `drun` select `work` by default. `VAULT_SCOPE=homelab` selects the home environment. Windows host setup defaults to homelab; Linux development setup defaults to work. The selected repository is `jerryyin/vault-work` or `jerryyin/vault-homelab`, cloned to `~/vault-work` or `~/vault-homelab`. Both repositories are private and start with separate histories. The old mixed repository is retired.

## Ownership and constraints

Scripts selects and makes the vault available; rc_files owns application credential formats. Each vault contains `.vault-scope` with exactly `work` or `homelab`. Activation rejects a mismatched or unmarked existing vault. No caller falls back to the mixed `~/vault` repository. SSH bootstrap may run before the selected vault exists, using already selected local keys.

Work contains the AMD GitHub, AMD Claude gateway, AtlArtifactory and MKMHub Docker credentials and work SSH inventory. Homelab contains personal GitHub and home service credentials, Samsung signing material and home SSH inventory. Private SSH keys remain outside both repositories. A key used for music-server, homelab or desktop must not be copied into the work environment, even when a work host also accepts that key.

Containers mount a dedicated role directory at `/zyin` and only the selected vault, read-only, at `/zyin/vault-<scope>`. They do not mount the host home, host SSH directory, SSH agent socket or all of `/data`. The role directory is `~/.local/share/devenv/<scope>` by default and holds that environment's workspace and explicitly imported keys/logins. A directory already used by another role cannot be reused. Builds use a separate Docker credential configuration for the selected role; vault files never enter the image build context or image layers.

The host may retain both checkouts and existing interactive logins. This separation controls what a build/container receives; it is not a security boundary against a host administrator or a user explicitly supplying additional mounts. Homelab services migrate their secret paths without rotating values or restarting unrelated services. An old `~/vault` compatibility link is acceptable only when it points to the homelab-only checkout.

Existing development images must be rebuilt after scripts and rc_files are updated. Compose refuses their old credential startup code before it can clone the mixed vault. `drun` still never builds implicitly. Existing running containers retain their old mounts until stopped and recreated.

## Validation

Tests must show one GitHub account per selected vault, refusal of a wrong or mixed vault, correct mounts and Docker credential directory, no host-home bind, and separate persistent roots. Existing SSH, credential rotation, argument handling and image-build isolation checks remain applicable. Migration compares secret bytes without printing them. Each new GitHub repository must be verified private before any upload.

## Alternatives Considered

Two folders or branches in the old vault would expose both roles through its Git history. Fresh repositories provide the requested separation. Merely changing the vault path while retaining the host-home bind would still expose the other vault, SSH keys and application logins. Dedicated container storage removes that accidental access. A new general secret manager would add deployment and recovery complexity without being required for this split.
