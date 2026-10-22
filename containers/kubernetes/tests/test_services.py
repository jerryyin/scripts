import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
INTERACTIVE = ROOT / 'containers/kubernetes/interactive'
spec = importlib.util.spec_from_file_location('kube_services', INTERACTIVE / 'services.py')
services = importlib.util.module_from_spec(spec)
spec.loader.exec_module(services)


class ServiceTests(unittest.TestCase):
    def test_list_and_stop_use_the_selected_configuration(self):
        with tempfile.TemporaryDirectory(prefix='kube-config-') as temporary:
            root = Path(temporary)
            config = root / 'alternate config.json'
            config.write_text(json.dumps({'namespace': 'selected-namespace', 'default_cluster': 'selected.conf'}))
            log = root / 'calls.jsonl'
            kubectl = root / 'kubectl'
            kubectl.write_text('''#!/usr/bin/env python3
import json, os, sys
with open(os.environ['CALL_LOG'], 'a') as stream:
    stream.write(json.dumps([sys.argv[1:], os.environ.get('KUBECONFIG')]) + '\\n')
print('{"items": []}')
''')
            jq = root / 'jq'
            jq.write_text('''#!/usr/bin/env python3
import json, sys
if sys.argv[2] in ('.namespace', '.default_cluster'):
    with open(sys.argv[3]) as stream:
        print(json.load(stream)[sys.argv[2][1:]])
''')
            kubectl.chmod(0o755)
            jq.chmod(0o755)
            environment = dict(os.environ, HOME=temporary, USER=root.name, KUBECONFIG='',
                               KUBE_DEV_CONFIG=str(config), CALL_LOG=str(log), PATH=temporary + ':' + os.environ['PATH'])
            for command in ('list-pods.sh', 'stop.sh'):
                with self.subTest(command=command):
                    log.unlink(missing_ok=True)
                    result = subprocess.run(['bash', str(INTERACTIVE / command)], env=environment,
                                            input='', text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    calls = [json.loads(line) for line in log.read_text().splitlines()]
                    self.assertTrue(calls)
                    for arguments, kubeconfig in calls:
                        self.assertIn('selected-namespace', arguments)
                        self.assertEqual(kubeconfig, str(root / '.kube/configs/selected.conf'))

    def test_existing_pod_refresh_copies_the_relocated_tools(self):
        source = (INTERACTIVE / 'connect.sh').read_text()
        start = source.index('                for directory in ')
        end = source.index('\n                done', start) + len('\n                done')
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / 'copies'
            kubectl = Path(temporary) / 'kubectl'
            kubectl.write_text('#!/bin/sh\ntest -d "$2" || exit 9\nprintf "%s\\n" "$2" >> "$COPY_LOG"\n')
            kubectl.chmod(0o755)
            result = subprocess.run(['bash', '-eu', '-c', source[start:end]], text=True, capture_output=True,
                                    env=dict(os.environ, PATH=temporary + ':' + os.environ['PATH'],
                                             SCRIPTS_ROOT=str(ROOT), NAMESPACE='fixture', POD_NAME='fixture', COPY_LOG=str(log)))
            self.assertEqual(result.returncode, 0, result.stderr)
            copied = {Path(path).name for path in log.read_text().splitlines()}
            self.assertIn('tools', copied)
            self.assertNotIn('gdb', copied)

    def test_all_services_select_existing_profiles(self):
        configured = services.load_services(INTERACTIVE / 'config.json')
        self.assertEqual(set(configured), {'base', 'rocmlir', 'ck', 'triton', 'triton-mi450', 'triton-b0', 'iree'})
        for name, (_, profile) in configured.items():
            with self.subTest(service=name):
                result = subprocess.run(['bash', str(ROOT / 'setup/linux/setup.sh'), profile, '--plan'],
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(configured['triton-mi450'][1], 'triton')
        self.assertEqual(configured['triton-b0'][1], 'triton')
        self.assertEqual(configured['ck'][1], 'base')
        self.assertEqual(configured['rocmlir'][1], 'base')

    def test_image_override_preserves_setup_profile(self):
        with patch.dict(os.environ, TRITON_MI450_BASE_IMAGE='example.invalid/custom:tag'):
            configured = services.load_services(INTERACTIVE / 'config.json')
        self.assertEqual(configured['triton-mi450'], ('example.invalid/custom:tag', 'triton'))

    def test_listing_does_not_need_docker_compose_or_cluster_access(self):
        with tempfile.TemporaryDirectory() as home:
            result = subprocess.run(['bash', str(INTERACTIVE / 'connect.sh'), '--list-services'],
                                    cwd=home, env=dict(os.environ, HOME=home, COMPOSE_FILE='/missing'),
                                    text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('triton-mi450\t', result.stdout)
        self.assertNotIn('Auto-setup', result.stdout)

    def test_unknown_service_fails(self):
        result = subprocess.run(['python3', str(INTERACTIVE / 'services.py'), 'unknown'],
                                text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Unknown service', result.stderr)

    def test_invalid_remote_profile_and_image_are_rejected(self):
        for entry in ({'image': 'ubuntu:24.04', 'setup_profile': 'triton; false'},
                      {'image': 'ubuntu:24.04; false', 'setup_profile': 'triton'}):
            with self.subTest(entry=entry), tempfile.TemporaryDirectory() as temporary:
                config = Path(temporary) / 'config.json'
                config.write_text(json.dumps({'services': {'fixture': entry}}))
                with self.assertRaises(ValueError):
                    services.load_services(config)


if __name__ == '__main__':
    unittest.main()
