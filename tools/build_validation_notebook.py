"""Explicitly opted-in validation notebook, separate from the normal interactive app."""
import json
import uuid
from pathlib import Path

root = Path(__file__).resolve().parents[1]
nb = json.loads((root/'private_3d_studio.ipynb').read_text())
for cell in nb['cells']:
    text = ''.join(cell['source'])
    if cell['cell_type'] == 'code':
        text = text.replace('ALLOW_DOWNLOAD = False', 'ALLOW_DOWNLOAD = True')
        text = text.replace('RUN_GPU_TEST = False', 'RUN_GPU_TEST = True')
        cell['source'] = text.splitlines(True)
intro = nb['cells'][0]
intro['source'] = ['# PRIVATE 3D STUDIO — actual GPU validation\n',
    '\nPrivate validation notebook. Run All explicitly downloads model checkpoints and executes real two-image geometry/PBR inference. This consumes GPU quota. Keep it private.\n']
nb['cells'].insert(6, dict(cell_type='code', id=uuid.uuid4().hex[:8], metadata={}, execution_count=None, outputs=[], source='''import unittest
suite = unittest.defaultTestLoader.discover(str(APP/'tests'))
cpu_result = unittest.TextTestRunner(verbosity=2).run(suite)
assert cpu_result.wasSuccessful() and not cpu_result.skipped
'''.splitlines(True)))
target = root/'private_3d_studio_validation.ipynb'
target.write_text(json.dumps(nb, indent=1), encoding='utf-8')
print(target)
