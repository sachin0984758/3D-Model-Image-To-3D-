# Adapted for Private 3D Studio; original copyright Gaurav Kumar (MIT).
"""Downloads every weight file (~17.8 GB) into one folder, in parallel.

Usage: python prefetch.py <weights_dir>

Layout (the workers and the Kaggle weights dataset both use it):
  <weights_dir>/tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/       shape model      6.9 GB
  <weights_dir>/tencent/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1/  texture model    6.4 GB
  <weights_dir>/facebook/dinov2-giant/                          texture encoder  4.2 GB
  <weights_dir>/RealESRGAN_x4plus.pth                           upscaler
  <weights_dir>/u2net/u2net.onnx                                background removal
"""
import os, shutil, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

try:
    import hf_transfer  # noqa: F401  (several times faster on Kaggle)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
except ImportError:
    pass
from huggingface_hub import snapshot_download

W = sys.argv[1]
DIT = "tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/model.fp16.ckpt"
PAINT = "tencent/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin"
DINO = "facebook/dinov2-giant/model.safetensors"
ESRGAN = "RealESRGAN_x4plus.pth"
U2NET = "u2net/u2net.onnx"


def have(rel):
    return os.path.exists(os.path.join(W, rel))


def hub(repo_id, patterns):
    # One call per repo: parallel calls sharing a local_dir clobber each other's temp files.
    if patterns:
        snapshot_download(repo_id=repo_id, allow_patterns=patterns, local_dir=os.path.join(W, repo_id))
    return repo_id


def url(link, rel):
    dest = os.path.join(W, rel)
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        urllib.request.urlretrieve(link, dest + ".part")
        os.replace(dest + ".part", dest)
    return rel


# Anything already present (e.g. linked in from a partial dataset) is skipped.
tasks = [
    lambda: hub("tencent/Hunyuan3D-2.1", ["hunyuan3d-dit-v2-1/*", "hunyuan3d-paintpbr-v2-1/*", "LICENSE", "NOTICE"]),
    lambda: hub("facebook/dinov2-giant", ["*.json", "*.safetensors", "*.txt", "LICENSE"]),
    lambda: url("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth", ESRGAN),
    lambda: url("https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx", U2NET),
]
with ThreadPoolExecutor(len(tasks)) as pool:
    for done in pool.map(lambda task: task(), tasks):
        print("ready:", done)

for repo_id in ("tencent/Hunyuan3D-2.1", "facebook/dinov2-giant"):  # download bookkeeping only
    pass  # Retain local download metadata to avoid repeat transfers.

missing = [rel for rel in (DIT, PAINT, DINO, ESRGAN, U2NET) if not have(rel)]
if missing:
    sys.exit(f"Download incomplete, missing: {missing}")
size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(W) for f in fs)
print(f"All weights present: {size / 2**30:.1f} GB in {W}")
