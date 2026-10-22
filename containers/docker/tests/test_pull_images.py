import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PullImagesTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix='image pull ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.script = Path(shutil.copy2(ROOT / 'pull-base-images.sh', self.root))
        self.compose = self.root / 'docker-compose.yml'
        self.compose.write_text('services: {}\n')
        self.log = self.root / 'calls'
        docker = self.root / 'docker'
        docker.write_text('''#!/usr/bin/env python3
import json, os, sys
with open(os.environ['CALL_LOG'], 'a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
if os.environ.get('FAIL_COMMAND') in sys.argv[1:]:
    sys.exit(23)
if 'config' in sys.argv:
    print(json.dumps({'services': {'triton': {'build': {'args': {'BASE_IMAGE': 'fixture:image'}}}}}))
''')
        docker.chmod(0o755)
        self.env = dict(os.environ, HOME=str(self.root), PATH=str(self.root) + ':' + os.environ['PATH'], CALL_LOG=str(self.log))
        for name in ('COMPOSE_FILE', 'FAIL_COMMAND'):
            self.env.pop(name, None)

    def invoke(self, service: str = 'triton') -> subprocess.CompletedProcess[str]:
        return subprocess.run(['bash', str(self.script), service], cwd=self.root,
                              env=self.env, text=True, capture_output=True)

    def test_default_compose_follows_relocated_script(self) -> None:
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertIn(['pull', 'fixture:image'], calls)
        self.assertTrue(any(str(self.compose) in call for call in calls))

    def test_failures_are_nonzero(self) -> None:
        self.env['COMPOSE_FILE'] = str(self.compose)
        for command in ('version', 'config', 'pull'):
            with self.subTest(command=command):
                self.env['FAIL_COMMAND'] = command
                result = self.invoke()
                self.assertNotEqual(result.returncode, 0, result.stdout)
        self.env.pop('FAIL_COMMAND')
        self.assertNotEqual(self.invoke('unknown').returncode, 0)

    def test_missing_model_never_calls_docker(self) -> None:
        self.env['COMPOSE_FILE'] = 'missing.yml'
        self.assertNotEqual(self.invoke().returncode, 0)
        self.assertFalse(self.log.exists())


if __name__ == '__main__':
    unittest.main()
