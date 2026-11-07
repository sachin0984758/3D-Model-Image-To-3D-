"""Build self-contained notebooks from the delivered source files."""
import base64
import json
import uuid
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return dict(cell_type='markdown', id=uuid.uuid4().hex[:8], metadata={}, source=text.splitlines(True))


def code(text):
    return dict(cell_type='code', id=uuid.uuid4().hex[:8], metadata={}, execution_count=None, outputs=[], source=text.splitlines(True))


def bundle():
    paths = [p for folder in ['studio', 'workers', 'tests'] for p in (ROOT/folder).glob('*') if p.is_file()]
    paths += [ROOT/p for p in ('LICENSE', 'TENCENT_LICENSE.txt', 'NOTICE', 'requirements.txt')]
    data = {str(p.relative_to(ROOT)).replace('\\', '/'): p.read_text(encoding='utf-8-sig') for p in paths}
    packed = base64.b85encode(zlib.compress(json.dumps(data).encode())).decode()
    return """# Embedded source: original MIT notice and Tencent terms are included.
import base64, zlib, json, sys
from pathlib import Path
APP = Path('/kaggle/working/private_3d_studio/app')
APP.mkdir(parents=True, exist_ok=True)
payload = json.loads(zlib.decompress(base64.b85decode(%r)))
for rel, text in payload.items():
    path = APP / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
print('PRIVATE 3D STUDIO source ready:', APP)
""" % packed


PREFLIGHT = """import sys, subprocess, shutil, json
from studio.runner import gpu_info
print('Python:', sys.version)
subprocess.run([sys.executable, '-c', 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, "CUDA available:", torch.cuda.is_available())'], check=True)
print('GPUs:', json.dumps(gpu_info(), indent=2))
print('nvcc:', shutil.which('nvcc'), 'C++:', shutil.which('c++'))
print('Free temporary GiB:', shutil.disk_usage('/tmp').free / 2**30)
if not gpu_info():
    raise RuntimeError('Select Settings > Accelerator > GPU T4 x2 and start the session.')
"""

INSTALL = """import os, subprocess, urllib.request
# Reuses the upstream compatibility fixes and Kaggle's existing torch installation.
env = dict(os.environ, BUILD_CUDA='1' if len(gpu_info()) >= 2 else '0')
subprocess.run(['bash', str(APP/'workers/install.sh')], env=env, check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', 'ipywidgets>=8,<9'], check=True)
(APP/'assets').mkdir(exist_ok=True)
# Fetch viewer code during setup, then render locally without outbound network access.
urllib.request.urlretrieve('https://cdn.jsdelivr.net/npm/@google/model-viewer@4.0.0/dist/model-viewer.min.js', APP/'assets/model-viewer.min.js')
urllib.request.urlretrieve('https://cdn.jsdelivr.net/npm/@google/model-viewer@4.0.0/LICENSE', APP/'assets/model-viewer-LICENSE.txt')
print('Setup completed. Restart the kernel if Kaggle asks, then rerun source/preflight cells; setup need not be repeated in this session.')
"""

WEIGHTS = """from studio.weights import prepare
DATASET = None  # Optional explicit root containing tencent/, facebook/, u2net/.
ALLOW_DOWNLOAD = False  # Explicitly opt in only if no private dataset is available.
WEIGHTS = prepare(APP, dataset=DATASET, allow_download=ALLOW_DOWNLOAD, textures=len(gpu_info()) >= 2)
print('Validated local weights:', WEIGHTS)
"""

SAVE = """from IPython.display import display
from studio.storage import export_all
from studio.ui import download_link
if studio.jobs():
    path = export_all(studio.root)
    display(download_link(path))
    print('In Kaggle Output, the ZIP is under private_3d_studio / exports / generated_models.zip.')
    print('Actual per-stage outcomes:')
    for item in studio.jobs():
        print(item['name'], item['shape'], item['paint'])
else:
    print('Generate a model in Studio, then run this cell to download the outputs.')
"""

main = [md('# PRIVATE 3D STUDIO\n\nPrivate notebook application for image-to-3D generation. Keep this notebook and all inputs **Private**.\n\nSelect **GPU T4 x2**, enable Internet for installation, then run cells in order. Attach your private weights dataset before the weights cell. Read the included Tencent license before using the model.\n\nTwo example inputs passed geometry and PBR export validation on T4 x2. Use the optional smoke test to check a changed runtime.'),
        code(bundle()), md('## Environment'), code(PREFLIGHT), md('## Install and compile'), code(INSTALL),
        md('## Locate model weights\nRun the weights builder once if you have no weights dataset. Inference workers operate offline.'), code(WEIGHTS),
        md('## Studio\nUpload images through the widget. Generate runs asynchronously; use Refresh dashboard to inspect progress. High settings may OOM on a T4. Geometry is retained when paint fails.'),
        code("""from studio.runner import Studio
from studio.ui import show
if 'studio' in globals():
    studio.stop()
studio = Studio('/kaggle/working/private_3d_studio', APP, WEIGHTS)
show(studio)
"""),
        md('## Optional reproducible GPU smoke test\nRun manually after setup. Uses an official example and the same worker runner as the interface. Two images exercise batch scheduling. Examine metadata and logs; file existence alone is not a pass.'),
        code("""# Change to True to execute REAL inference; this consumes Kaggle GPU quota.
RUN_GPU_TEST = False
if RUN_GPU_TEST:
    import urllib.request
    from studio.storage import prepare_image, atomic_json, export_all
    samples = []
    for filename in ['004.png', '052.png']:
        # The repository listing should be checked if the example path changes.
        url = 'https://raw.githubusercontent.com/Tencent-Hunyuan/Hunyuan3D-2.1/82920d643c0dc2f7bfd7255f45f62d386edfe60c/assets/example_images/'+filename
        try:
            data = urllib.request.urlopen(url).read()
            samples.append(prepare_image(data, filename, studio.root/'uploads'))
        except Exception as e:
            print('Example download failed:', filename, str(e))
    if not samples:
        raise RuntimeError('No official example downloaded. Use an image from the official assets directory.')
    report = studio.run(samples, dict(textures=True))
    atomic_json(studio.root/'gpu_test_results.json', report)
    print(json.dumps(report, indent=2))
    print(export_all(studio.root))
"""),
        md('## Save your work\nAfter generation, run the cell below and click `generated_models.zip`. You can also use Download All in Studio. In Kaggle Output, find `private_3d_studio/exports/generated_models.zip`. Save a **private** notebook version to retain outputs. To resume in a new session, attach the saved output or a private dataset and copy its `outputs/` folder into `/kaggle/working/private_3d_studio/outputs/` before generating. Upload the same references; completed jobs with identical settings are reused.\n\nFor unattended batch commits, replace interactive uploads with `prepare_image` calls over an explicitly selected private image folder and call `studio.run` directly. Widgets require an interactive session.'),
        code(SAVE)]

builder = [md('# PRIVATE 3D STUDIO — weights builder\n\nKeep **Private**. Use **no accelerator** and enable Internet. Run this notebook once, save a completed version, then create a **Private** dataset from the `private-3d-weights` output folder. You must create/confirm dataset visibility in Kaggle; the notebook cannot enforce it. Requires roughly 18 GB; actual size is printed.'),
           code(bundle()), code("""import subprocess
subprocess.run([sys.executable, '-m', 'pip', 'install', 'huggingface-hub==0.30.2', 'hf_transfer'], check=True)
root = Path('/kaggle/working/private-3d-weights')
subprocess.run([sys.executable, str(APP/'workers/prefetch.py'), str(root)], check=True)
from studio.weights import build_manifest, validate
build_manifest(root)
validate(root, verify_hashes=False)  # Just fingerprinted; future sessions verify hashes on read.
print('Validated bytes:', sum(p.stat().st_size for p in root.rglob('*') if p.is_file()))
print('Save Version > Save & Run All. Once successful, create a Private dataset from this output.')
""")]

for filename, cells, gpu in [('private_3d_studio.ipynb', main, True), ('private_3d_weights_builder.ipynb', builder, False)]:
    nb = dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python', version='3.13'), kaggle=dict(accelerator='nvidiaTeslaT4' if gpu else 'none', isInternetEnabled=True, isGpuEnabled=gpu)))
    (ROOT/filename).write_text(json.dumps(nb, indent=1), encoding='utf-8')
    print('Built', filename)
