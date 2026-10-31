"""Run Docker with one vault and a separate persistent directory for its role."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def prepare() -> dict[str, str]:
    environment = dict(os.environ)
    scope = environment.get('VAULT_SCOPE', 'work')
    if scope not in ('work', 'homelab'):
        raise ValueError('VAULT_SCOPE must be work or homelab')
    home = Path.home().resolve()
    vault = Path(environment.get('VAULT_DIR', str(home / f'vault-{scope}'))).expanduser().resolve()
    marker = vault / '.vault-scope'
    if not marker.is_file() or marker.read_text().strip() != scope:
        raise ValueError(f'Vault scope mismatch or missing .vault-scope: {vault}')
    root = Path(environment.get('DEVENV_ROLE_ROOT', str(home / '.local/share/devenv' / scope))).expanduser().resolve()
    if home.is_relative_to(root) or vault.is_relative_to(root) or root.is_relative_to(vault):
        raise ValueError('The role directory cannot mount the host home or contain a vault checkout')
    marker = root / '.vault-scope'
    if root.exists() and any(root.iterdir()) and not marker.is_file():
        raise ValueError('Use a new role directory; existing unclassified storage cannot be imported')
    if marker.exists() and marker.read_text().strip() != scope:
        raise ValueError('Persistent directory belongs to another vault scope')
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    if not marker.exists():
        marker.write_text(scope + '\n')
    environment.update(VAULT_SCOPE=scope, DEVENV_ROLE_ROOT=str(root), DEVENV_VAULT_DIR=str(vault),
                       DOCKER_CONFIG=str(root / '.docker'))
    for name in ('SSH_AUTH_SOCK', 'GH_TOKEN', 'GITHUB_TOKEN', 'DOCKER_AUTH_CONFIG', 'DOCKER_AUTH_FILE'):
        environment.pop(name, None)
    rc = Path(environment.get('RC_FILES_DIR', str(home / 'rc_files')))
    if not (rc / '.setup/__main__.py').is_file():
        raise ValueError(f'Missing rc_files credential setup: {rc}')
    subprocess.run([sys.executable, '-B', str(rc / '.setup'), 'credentials', '--package', 'docker',
                    '--vault', str(vault), '--vault-scope', scope, '--home', str(root)],
                   env=environment, check=True, stdout=sys.stderr)
    return environment


def main() -> None:
    environment = prepare()
    arguments = sys.argv[1:]
    if arguments[0] == 'run':
        boundary = arguments.index('--')
        options, caller = arguments[:boundary], arguments[boundary + 1:]
        arguments = ['sudo', 'docker', '--config', environment['DOCKER_CONFIG'], *options,
                     '-v', environment['DEVENV_ROLE_ROOT'] + ':/zyin',
                     '-v', environment['DEVENV_VAULT_DIR'] + ':/zyin/vault-' + environment['VAULT_SCOPE'] + ':ro',
                     '-e', 'VAULT_SCOPE=' + environment['VAULT_SCOPE'],
                     '-e', 'DEVENV_PERSISTENT_ROOT=/zyin', *caller]
    else:
        arguments = ['docker', *arguments]
    os.execvpe(arguments[0], arguments, environment)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as error:
        print(f'Docker credentials: {error}', file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
