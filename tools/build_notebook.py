"""Build the single self-contained notebook for Kaggle and Google Colab."""
import base64
import hashlib
import json
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return dict(cell_type='markdown', id=hashlib.sha256(('md:'+text).encode()).hexdigest()[:12], metadata={}, source=text.splitlines(True))


def code(text):
    return dict(cell_type='code', id=hashlib.sha256(('code:'+text).encode()).hexdigest()[:12], metadata={}, execution_count=None, outputs=[], source=text.splitlines(True))


def bundle():
    paths = [p for folder in ['studio', 'workers', 'tests'] for p in sorted((ROOT/folder).glob('*')) if p.is_file()]
    paths += [ROOT/p for p in ('LICENSE', 'TENCENT_LICENSE.txt', 'NOTICE', 'requirements.txt')]
    data = {str(p.relative_to(ROOT)).replace('\\', '/'): p.read_text(encoding='utf-8-sig') for p in paths}
    packed = base64.b85encode(zlib.compress(json.dumps(data).encode())).decode()
    return """# Embedded source: original MIT notice and Tencent terms are included.
import base64, zlib, json, sys, importlib
from pathlib import Path
if Path('/kaggle/working').is_dir():
    WORK_ROOT = Path('/kaggle/working')
elif Path('/content').is_dir():
    WORK_ROOT = Path('/content')
else:
    raise RuntimeError('Open this notebook in a Kaggle or Google Colab GPU runtime.')
APP = WORK_ROOT / 'private_3d_studio' / 'app'
if 'studio' in globals():
    studio.stop()
    if not studio.lock.acquire(timeout=30):
        raise RuntimeError('The previous batch is still stopping. Run this cell again once it finishes.')
    studio.lock.release()
    del studio
APP.mkdir(parents=True, exist_ok=True)
payload = json.loads(zlib.decompress(base64.b85decode(%r)))
for rel, text in payload.items():
    path = APP / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
for module_name in list(sys.modules):
    if module_name == 'studio' or module_name.startswith('studio.'):
        del sys.modules[module_name]
importlib.invalidate_caches()
print('Private 3D Studio source ready:', APP)
""" % packed


PREFLIGHT = """import sys, subprocess, shutil, json
from studio.runner import gpu_info
print('Python:', sys.version)
subprocess.run([sys.executable, '-c', 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, "CUDA available:", torch.cuda.is_available())'], check=True)
print('GPUs:', json.dumps(gpu_info(), indent=2))
print('nvcc:', shutil.which('nvcc'), 'C++:', shutil.which('c++'))
print('Free working storage GiB:', shutil.disk_usage(WORK_ROOT).free / 2**30)
if not gpu_info():
    raise RuntimeError('Enable a GPU runtime. Use T4 x2 on Kaggle or T4 on Colab.')
"""

INSTALL = """import os, subprocess, urllib.request
# Reuses the compatibility fixes and this runtime's existing PyTorch installation.
env = dict(os.environ, BUILD_CUDA='1' if gpu_info() else '0')
subprocess.run(['bash', str(APP/'workers/install.sh')], env=env, check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', 'ipywidgets>=8,<9'], check=True)
(APP/'assets').mkdir(exist_ok=True)
# Fetch viewer code during setup, then render locally without outbound network access.
urllib.request.urlretrieve('https://cdn.jsdelivr.net/npm/@google/model-viewer@4.0.0/dist/model-viewer.min.js', APP/'assets/model-viewer.min.js')
urllib.request.urlretrieve('https://cdn.jsdelivr.net/npm/@google/model-viewer@4.0.0/LICENSE', APP/'assets/model-viewer-LICENSE.txt')
print('Setup completed. If the runtime asks for a restart, restart it and run all again. Cached weights will be reused.')
"""

WEIGHTS = """from studio.weights import prepare
DATASET = None  # Attached weights datasets and saved notebook outputs are detected automatically.
ALLOW_DOWNLOAD = True  # Download missing weights as part of this notebook's setup.
WEIGHTS_DIR = WORK_ROOT / 'private-3d-weights'
WEIGHTS = prepare(APP, destination=WEIGHTS_DIR, dataset=DATASET,
                  allow_download=ALLOW_DOWNLOAD, textures=True)
print('Validated local weights:', WEIGHTS)
print('Downloaded weights remain in this runtime. Save or attach weights to reuse them across sessions.')
"""

SAVE = """from IPython.display import display
from studio.storage import export_all
from studio.ui import download_file
if studio.jobs():
    path = export_all(studio.root)
    display(download_file(path))
    print('Model ZIP:', path)
    print('Actual per-stage outcomes:')
    for item in studio.jobs():
        print(item['name'], item['shape'], item['paint'])
else:
    print('Generate a model in Studio, then run this cell to download the outputs.')
"""

main = [md('# Private 3D Studio\n\nOne notebook for image-to-3D generation on **Kaggle or Google Colab**. It installs the runtime, downloads weights, opens the studio and exports your models. Keep your notebook and images private.\n\n**Kaggle** — select **GPU T4 x2**, enable **Internet**, then click **Run All**.\n\n**Colab** — choose **Runtime > Change runtime type > T4 GPU**, then **Runtime > Run all**. The single GPU generates geometry first and paints it afterward.\n\nWhen the studio appears, upload an image or choose an example. Select **Generate Shape** or **Generate Textured Shape**. The centre panel previews completed models. Advanced options and export settings are on the left.\n\nThe first download needs roughly 18 GiB. Weights are cached in `/kaggle/working/private-3d-weights` on Kaggle or `/content/private-3d-weights` on Colab. Runtime storage is temporary until you save it. Model use follows the included Tencent license.\n\nAn earlier two-GPU Kaggle run validated two textured models. The current interface and one-GPU scheduling have CPU fixture checks. A live Colab GPU run remains unverified.'),
        code(bundle()), md('## Environment'), code(PREFLIGHT), md('## Install and compile'), code(INSTALL),
        md('## Model weights\nThis cell downloads missing checkpoints and writes their SHA-256 manifest. Completed downloads are reused on later runs. Existing attached weights datasets or saved notebook outputs are detected automatically. Inference workers load these local files.'), code(WEIGHTS),
        md('## Studio\nUpload your reference on the left or choose an example on the right. Use Medium settings for your first run. High settings may exceed T4 memory.'),
        code("""from studio.runner import Studio
from studio.ui import show
try:
    from google.colab import output
except ImportError:
    pass
else:
    output.enable_custom_widget_manager()
if 'studio' in globals():
    studio.stop()
studio = Studio(WORK_ROOT / 'private_3d_studio', APP, WEIGHTS)
show(studio)
"""),
        md('## Optional reproducible GPU smoke test\nRun manually after setup. Uses an official example and the same worker runner as the interface. Two images exercise batch scheduling. Examine metadata and logs; file existence alone is not a pass.'),
        code("""# Change to True to execute real inference; this consumes runtime GPU quota.
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
        md('## Save your work\nUse **Prepare export > Download** for GLB, OBJ, PLY or STL. Textured OBJ exports include their material and image files in a ZIP. **Download all / checkpoint** saves the generation outputs and recovery records.\n\nThe cell below downloads the full model ZIP. Colab uses its file-download helper. Kaggle displays a relative file link, and the ZIP is also available under `private_3d_studio/exports/generated_models.zip` in Output.\n\nOn Kaggle, save a private notebook version with outputs to retain the downloaded weights folder, then attach that output through **Add Input** in a later session. Colab runtime files disappear when the runtime is cleared. Save your generated ZIP before disconnecting.\n\nTo resume models, restore the saved `outputs/` directory under this runtime\'s `private_3d_studio/` folder and use the same references and settings.'),
        code(SAVE)]

filename = 'private_3d_studio.ipynb'
nb = dict(nbformat=4, nbformat_minor=5, cells=main,
          metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                        language_info=dict(name='python'), accelerator='GPU',
                        colab=dict(name='Private 3D Studio', gpuType='T4', provenance=[]),
                        kaggle=dict(accelerator='nvidiaTeslaT4', isInternetEnabled=True, isGpuEnabled=True)))
(ROOT/filename).write_text(json.dumps(nb, indent=1), encoding='utf-8')
print('Built', filename)
