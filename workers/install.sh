#!/usr/bin/env bash
# Installs Hunyuan3D 2.1 using the notebook runtime's existing PyTorch.
set -euo pipefail
PY=${PY:-python}
REPO=${REPO:-/tmp/Hunyuan3D-2.1}
BUILD_CUDA=${BUILD_CUDA:-1}
mkdir -p /tmp/hy3d

EXPECTED_REV=82920d643c0dc2f7bfd7255f45f62d386edfe60c
# Fetch the tested commit directly, including when the default branch has advanced.
if [ ! -d "$REPO" ]; then
  git init -q "$REPO"
  git -C "$REPO" remote add origin https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
fi
[ -e "$REPO/.git" ] || { echo "Runtime folder is not a Git checkout: $REPO"; exit 1; }
cd "$REPO"
if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
  git fetch -q --depth 1 origin "$EXPECTED_REV"
  git checkout -q --detach "$EXPECTED_REV"
fi
[ "$(git rev-parse HEAD)" = "$EXPECTED_REV" ] || { echo "Unexpected runtime revision. Expected $EXPECTED_REV; use a fresh checkout at that commit."; exit 1; }
"$PY" -c "import sys, torch; print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.version.cuda)"

# Some Colab images omit the shared libraries used by MeshLab's plugins.
if ! "$PY" - <<'PYSYSTEM'
import ctypes
import sys
missing = []
for library in ('libGL.so.1', 'libOpenGL.so.0', 'libEGL.so.1'):
    try:
        ctypes.CDLL(library)
    except OSError:
        missing.append(library)
if missing:
    print('Installing missing graphics libraries:', ', '.join(missing))
sys.exit(bool(missing))
PYSYSTEM
then
  [ "$(id -u)" = 0 ] && command -v apt-get >/dev/null || { echo "Missing graphics libraries; install libgl1 libopengl0 libegl1 in this runtime."; exit 1; }
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libgl1 libopengl0 libegl1
fi

# numpy 1.26 has no wheels for Python 3.13+, where numpy 2.2 is tested to work.
"$PY" -c "import sys; print('numpy==1.26.4' if sys.version_info < (3, 13) else 'numpy<2.3')" > /tmp/hy3d/constraints.txt
PIP="$PY -m pip install -q --root-user-action=ignore -c /tmp/hy3d/constraints.txt"

echo "Installing Python packages..."
$PIP numpy \
  diffusers==0.30.0 transformers==4.46.0 accelerate==1.1.1 huggingface-hub==0.30.2 hf_transfer \
  pytorch-lightning==1.9.5 safetensors einops==0.8.0 omegaconf==2.3.0 pyyaml tqdm psutil configargparse \
  opencv-python==4.10.0.84 opencv-python-headless==4.10.0.84 scikit-image imageio \
  trimesh==4.4.7 "pymeshlab>=2022.2.post3" pygltflib==1.16.3 xatlas \
  rembg==2.0.65 onnxruntime timm torchdiffeq ninja pybind11 spandrel

# Memory-map zip checkpoints so the single-GPU Colab loader can reclaim mapped pages.
"$PY" - <<'PYLOAD'
from pathlib import Path
path = Path('hy3dshape/hy3dshape/pipelines.py')
source = path.read_text()
old = "ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)"
new = "ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True, mmap=zipfile.is_zipfile(ckpt_path))"
if old in source:
    source = source.replace('import torch\n', 'import torch\nimport zipfile\n', 1).replace(old, new)
    path.write_text('# Private 3D Studio modification: memory-mapped checkpoint loading.\n' + source)
elif new not in source:
    raise RuntimeError('The expected shape checkpoint loader was not found.')
PYLOAD

# bpy (Blender) has no wheel for this Python and is only used by an OBJ->GLB helper we replace.
# Mark modified vendor files while retaining their original headers.
sed -i 's/^import bpy$/try:\n    import bpy\nexcept ImportError:\n    bpy = None/' hy3dpaint/DifferentiableRenderer/mesh_utils.py
if ! head -1 hy3dpaint/DifferentiableRenderer/mesh_utils.py | grep -q 'Private 3D Studio modification'; then
  sed -i '1i# Private 3D Studio modification: optional bpy import; original notices below.' hy3dpaint/DifferentiableRenderer/mesh_utils.py
fi
# basicsr/realesrgan (the texture upscaler) fail to install on Python 3.13; swap in spandrel.
# Private 3D Studio: retain vendor comment/license header when replacing this helper.
"$PY" - <<'PYNOTICE'
from pathlib import Path
import subprocess
source = subprocess.check_output(['git', 'show', 'HEAD:hy3dpaint/utils/image_super_utils.py']).decode()
lines = []
for line in source.splitlines(True):
    if line.startswith('#') or not line.strip():
        lines.append(line)
    else:
        break
Path('/tmp/hy3d/upscaler-header.txt').write_text(''.join(lines))
PYNOTICE
cat > hy3dpaint/utils/image_super_utils.py <<'PYEOF'
# Replaces the basicsr/realesrgan upscaler: neither package installs on Python 3.13+.
# spandrel loads the same RealESRGAN_x4plus.pth weights.
import numpy as np
import torch
from PIL import Image
from spandrel import ImageModelDescriptor, ModelLoader


class imageSuperNet:
    def __init__(self, config) -> None:
        model = ModelLoader().load_from_file(config.realesrgan_ckpt_path)
        assert isinstance(model, ImageModelDescriptor)
        device = torch.device(config.device)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.model = model.to(device).to(dtype).eval()
        self.device, self.dtype = device, dtype

    @torch.no_grad()
    def __call__(self, image):
        x = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1)[None]
        y = self.model(x.to(self.device, self.dtype) / 255.0).clamp(0, 1)
        return Image.fromarray((y[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8))
PYEOF
"$PY" - <<'PYNOTICE'
from pathlib import Path
p = Path('hy3dpaint/utils/image_super_utils.py')
p.write_text('# Private 3D Studio modification: spandrel compatibility adapter.\n' +
             Path('/tmp/hy3d/upscaler-header.txt').read_text() + p.read_text())
PYNOTICE

EXT=$("$PY" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
echo "Building mesh inpainter..."
(cd hy3dpaint/DifferentiableRenderer && \
  c++ -O3 -shared -std=c++11 -fPIC $("$PY" -m pybind11 --includes) mesh_inpaint_processor.cpp -o "mesh_inpaint_processor$EXT")

if [ "$BUILD_CUDA" = 1 ]; then
  export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
  export PATH="$CUDA_HOME/bin:$PATH"
  command -v nvcc >/dev/null || { echo "nvcc not found in $CUDA_HOME/bin - is a GPU accelerator enabled?"; exit 1; }
  # Compile only for the GPUs in this session (T4 = 7.5, P100 = 6.0): much faster than all archs.
  export TORCH_CUDA_ARCH_LIST=$("$PY" -c "import torch; print(';'.join(sorted({'%d.%d' % torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})))")
  echo "Building custom rasterizer for CUDA arch $TORCH_CUDA_ARCH_LIST (takes a few minutes)..."
  (cd hy3dpaint/custom_rasterizer && $PY -m pip install -q --root-user-action=ignore --no-build-isolation .)
fi

echo "Checking imports..."
"$PY" - <<'EOF'
import os, sys
sys.path[:0] = ["hy3dshape", "hy3dpaint"]
from DifferentiableRenderer import mesh_inpaint_processor
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
if os.environ.get("BUILD_CUDA", "1") == "1":
    import custom_rasterizer_kernel
    from textureGenPipeline import Hunyuan3DPaintPipeline
from convert_utils import create_glb_with_pbr_materials
import numpy
import pymeshlab
meshes = pymeshlab.MeshSet()
assert hasattr(meshes, 'meshing_decimation_quadric_edge_collapse'), 'MeshLab geometry plugin is unavailable'
assert hasattr(meshes, 'meshing_decimation_quadric_edge_collapse_with_texture'), 'MeshLab texture plugin is unavailable'
print("Install OK (numpy %s)" % numpy.__version__)
EOF
