"""The simulator reducers find their shared decoder after a checkout is moved."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DecoderPathTests(unittest.TestCase):
    def test_decoder_discovery_in_relocated_and_standalone_scripts(self):
        for source in ('sibling', 'scripts_root', 'home', 'explicit'):
            for name in ('parse_am_itrace.py', 'compare_att_itrace.py'):
                with self.subTest(source=source, script=name), tempfile.TemporaryDirectory() as temporary:
                    work = Path(temporary)
                    home = work / 'home'
                    home.mkdir()
                    repo = home / 'scripts' if source == 'home' else work / 'relocated checkout'
                    decoder = repo / 'profiling/att_analyze.py'
                    decoder.parent.mkdir(parents=True)
                    shutil.copy2(ROOT / 'profiling/att_analyze.py', decoder)
                    script = repo / 'am' / name if source in ('sibling', 'explicit') else work / name
                    script.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ROOT / 'am' / name, script)
                    env = dict(os.environ, HOME=str(home), PYTHONDONTWRITEBYTECODE='1')
                    env.pop('ATT_ANALYZE', None)
                    env.pop('SCRIPTS_ROOT', None)
                    if source == 'scripts_root':
                        env['SCRIPTS_ROOT'] = str(repo)
                    elif source == 'explicit':
                        override = work / 'explicit decoder.py'
                        shutil.copy2(decoder, override)
                        decoder = override
                        env['ATT_ANALYZE'] = str(decoder)
                    result = subprocess.run(['python3', '-B', '-c', '''
import runpy, sys
from pathlib import Path
namespace = runpy.run_path(sys.argv[1])
if 'load_categorizer' in namespace:
    categorize, _, decoder, _ = namespace['load_categorizer'](None)
else:
    decoder = namespace['DEFAULT_ATT_ANALYZE']
    categorize = namespace['load_categorize'](decoder)
assert decoder == Path(sys.argv[2]), (decoder, sys.argv[2])
assert categorize('s_endpgm') == 's_endpgm'
''', str(script), str(decoder)], cwd=work, env=env, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
