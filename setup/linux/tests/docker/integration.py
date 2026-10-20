#!/usr/bin/env python3
"""Opt-in Docker/Compose contract tests; installers record calls inside containers.

This checks real Docker builds, Compose startup/restart, argument handling, and
canonical setup paths. It does not claim to validate third-party installers.
See README.md for the separate full-install validation procedure.
"""

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[4]


def snapshot(source, destination):
    files = subprocess.check_output(
        ['git', '-C', str(source), 'ls-files', '-c', '-o', '--exclude-standard', '-z'])
    for name in filter(None, files.split(b'\0')):
        relative = Path(os.fsdecode(name))
        if relative.name.startswith('.') and relative.suffix in ('.swp', '.swo', '.swn'):
            continue
        original = source / relative
        if not original.exists() and not original.is_symlink():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target, follow_symlinks=False)


def commit(directory):
    for args in (['init', '-q', '-b', 'validation'], ['add', '--force', '.'],
                 ['-c', 'user.name=Local validation', '-c', 'user.email=validation@localhost',
                  'commit', '-qm', 'Disposable validation snapshot']):
        subprocess.run(['git', '-C', str(directory), *args], check=True,
                       stdout=subprocess.DEVNULL)


def record(script, name):
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text('#!/bin/bash\nset -e\n'
                      f"printf '%s|%s|%s\\n' '{name}' "
                      '"${DEVENV_PERSISTENT_ROOT-unset}" "$*" >> /validation-calls\n'
                      f'[[ "${{TEST_FAIL_STEP:-}}" != "{name}" ]] || exit 23\n')
    script.chmod(0o755)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rc-files', type=Path, default=ROOT.parent / 'rc_files')
    parser.add_argument('--docker-command', default='docker', help="e.g. 'sudo -n docker'")
    parser.add_argument('--logs', type=Path, required=True)
    args = parser.parse_args()
    args.logs.mkdir(parents=True, exist_ok=True)
    docker = shlex.split(args.docker_command)
    configuration = ROOT / 'containers/docker'
    project = f'scripts-validation-{os.getpid()}'
    tags = []
    checks = []

    def run(command, name, *, expected=0):
        logfile = args.logs / f'{name}.log'
        with logfile.open('w') as output:
            result = subprocess.run(list(map(str, command)), stdin=subprocess.DEVNULL,
                                    text=True, stdout=output, stderr=subprocess.STDOUT,
                                    timeout=900)
        output = logfile.read_text()
        if result.returncode != expected:
            raise RuntimeError(f'{name}: exit {result.returncode}\n' + output[-5000:])
        return output

    def passed(name):
        checks.append(name)
        print(f'PASS {name}', flush=True)

    with tempfile.TemporaryDirectory(prefix='scripts docker validation ') as temporary:
        work = Path(temporary)
        sources = work / 'sources'
        source = sources / 'scripts'
        snapshot(ROOT, source)
        for script in (source / 'setup/linux').rglob('*.sh'):
            relative = script.relative_to(source / 'setup/linux')
            is_plan = relative.parts[0] == 'profiles' and relative.name == 'profile.sh'
            if relative.parts[0] not in ('lib', 'tests') and relative != Path('setup.sh') and not is_plan:
                record(script, str(relative))
        commit(source)
        compose = None
        try:
            seed = f'{project}:seed'
            tags.append(seed)
            (work / 'Dockerfile').write_text('''FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y git ca-certificates sudo wget curl gpg lsb-release zsh
COPY sources /validation-sources
ENV DEVENV_PERSISTENT_ROOT=/must-not-be-used-during-build
ENTRYPOINT ["/bin/false"]
''')
            run([*docker, 'build', '-t', seed, work], 'seed-build')
            model = json.loads(run([*docker, 'compose', '-f', configuration / 'docker-compose.yml',
                                   '--profile', '*', 'config', '--no-consistency', '--format', 'json'],
                                  'compose-model'))
            services = {name: value for name, value in model['services'].items()
                        if value.get('build', {}).get('dockerfile', '').endswith('base.dockerfile')}
            overrides = ['services:']
            for name in services:
                tag = f'{project}:{name}'
                tags.append(tag)
                overrides.extend([f'  {name}:', f'    image: {tag}',
                                  f'    container_name: {project}-{name}',
                                  '    devices: !reset []', '    volumes: !reset []',
                                  '    tmpfs: !reset []', '    group_add: !reset []',
                                  '    cap_add: !reset []', '    security_opt: !reset []',
                                  '    network_mode: none', '    build:',
                                  '      args:', f'        BASE_IMAGE: {seed}',
                                  '        SCRIPTS_REPOSITORY: file:///validation-sources/scripts',
                                  '        SCRIPTS_REF: validation'])
            override = work / 'override.yml'
            override.write_text('\n'.join(overrides) + '\n')
            compose = [*docker, 'compose', '-p', project, '-f', configuration / 'docker-compose.yml',
                       '-f', override]

            def calls(service):
                return run([*compose, 'exec', '-T', service, 'cat', '/validation-calls'],
                           f'{service}-calls').splitlines()

            # Exercise the actual rc_files shell functions through a per-file
            # symlink, as deployed by rc_files/.setup. The merged model points
            # only at our disposable image tags and service settings.
            merged = run([*compose, '--profile', '*', 'config', '--format', 'json'], 'dbuild-model')
            merged_path = work / 'entrypoint-compose.json'
            merged_path.write_text(merged)
            shell_home = work / 'shell-home'
            (shell_home / '.docker').mkdir(parents=True)
            (shell_home / '.docker/docker-compose.yml').symlink_to(merged_path)
            binaries = work / 'bin'
            binaries.mkdir()
            wrapper = binaries / 'docker'
            wrapper.write_text('#!/bin/sh\nexec ' + shlex.join(docker) + ' "$@"\n')
            wrapper.chmod(0o755)
            shell = (args.rc_files / 'zsh/.linux/.zshrc').read_text()
            definitions = shell[shell.index('function set_compose_project_name()'):shell.index('export CLICOLOR_FORCE=1')]
            invocation = work / 'dbuild.zsh'

            def dbuild(arguments, name):
                invocation.write_text(definitions + '\ndbuild ' + shlex.join(arguments) + '\n')
                run(['env', f'HOME={shell_home}', f'PATH={binaries}:{os.environ["PATH"]}',
                     'zsh', '-f', invocation], name)

            for name, service in services.items():
                dbuild([name], f'{name}-dbuild')
                profile = service['build']['args']['SERVICE_NAME']
                expected = ['bootstrap.sh||']
                if profile == 'triton':
                    expected += ['profiles/triton/deps.sh||', 'packages/tokenspeed.sh||',
                                 'profiles/triton/workspace.sh||']
                elif profile == 'iree':
                    expected += ['profiles/iree/deps.sh||', 'profiles/iree/workspace.sh||']
                built = run([*docker, 'run', '--rm', f'{project}:{name}',
                             'cat', '/validation-calls'], f'{name}-image-calls').splitlines()
                assert built == expected, (name, built, expected)
                passed(f'{name}: actual dbuild routes profile and disables activation/persistence')
                run([*compose, 'up', '-d', '--no-build', '--no-deps', name], f'{name}-start')
                for _ in range(100):
                    actual = calls(name)
                    if len(actual) == len(expected) + 1:
                        break
                    time.sleep(.1)
                assert actual == expected + ['credentials/activate.sh|/zyin|'], actual
                run([*compose, 'restart', '-t', '1', name], f'{name}-restart')
                for _ in range(100):
                    actual = calls(name)
                    if len(actual) == len(expected) + 2:
                        break
                    time.sleep(.1)
                assert actual == expected + ['credentials/activate.sh|/zyin|'] * 2, actual
                passed(f'{name}: Compose starts Zsh and reactivates credentials on restart')

            base_image = f'{project}:base'
            run([*docker, 'run', '--rm', '-e', 'TEST_FAIL_STEP=profiles/triton/deps.sh',
                          base_image, 'bash', '/root/scripts/setup/linux/setup.sh',
                          'triton', '--phase', 'build'], 'installer-failure', expected=23)
            passed('installer failure makes the container command fail')

            def direct_build(name, repository='scripts', *extra, expected=0):
                tag = f'{project}:{name}'
                tags.append(tag)
                run([*docker, 'build', '-t', tag,
                     '--build-arg', f'BASE_IMAGE={seed}',
                     '--build-arg', f'SCRIPTS_REPOSITORY=file:///validation-sources/{repository}',
                     *extra, '-f', configuration / 'base.dockerfile', configuration],
                    f'{name}-build', expected=expected)
                return tag

            image = direct_build('precedence', 'scripts', '--build-arg', 'SERVICE_NAME=base',
                                 '--build-arg', 'SETUP_PROFILE=triton')
            output = run([*docker, 'run', '--rm', image, 'cat', '/validation-calls'], 'precedence-calls')
            assert 'profiles/triton/workspace.sh||' in output, output
            passed('SETUP_PROFILE overrides SERVICE_NAME')
            dbuild(['--build-arg', 'SERVICE_NAME=iree', 'base'], 'dbuild-override')
            output = run([*docker, 'run', '--rm', base_image, 'cat', '/validation-calls'], 'dbuild-calls')
            assert 'profiles/iree/workspace.sh||' in output, output
            assert 'profiles/triton/deps.sh' not in output, output
            passed('actual dbuild function resolves its symlink and honors SERVICE_NAME overrides')
            direct_build('invalid-profile', 'scripts', '--build-arg', 'SERVICE_NAME=does-not-exist',
                         expected=1)
            direct_build('missing-source', 'missing', expected=1)
            passed('unknown profiles and clone failures fail real image builds')
        finally:
            if compose:
                run([*compose, '--profile', '*', 'down', '--remove-orphans', '-t', '1'], 'cleanup-compose')
            subprocess.run([*docker, 'image', 'rm', *tags], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
    (args.logs / 'summary.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(f'{len(checks)} Docker integration checks passed. Logs: {args.logs}')


if __name__ == '__main__':
    main()
