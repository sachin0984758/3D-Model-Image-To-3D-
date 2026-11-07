"""Detect, validate, stage and fingerprint local model files."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from .storage import atomic_json

CRITICAL = [
    'tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/model.fp16.ckpt',
    'tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/config.yaml',
    'u2net/u2net.onnx',
    'tencent/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin',
    'facebook/dinov2-giant/model.safetensors',
    'RealESRGAN_x4plus.pth',
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def validate(root, textures=True, verify_hashes=True):
    root = Path(root)
    required = CRITICAL if textures else CRITICAL[:3]
    missing = [p for p in required if not (root/p).is_file() or (root/p).stat().st_size == 0]
    if missing:
        raise FileNotFoundError('Missing/empty checkpoints: ' + ', '.join(missing))
    for rel in required:
        p = root/rel
        if p.suffix in {'.ckpt', '.bin', '.pth', '.onnx', '.safetensors'}:
            with p.open('rb') as f:
                head = f.read(128)
            if p.stat().st_size < 1024 or head.startswith((b'version https://git-lfs', b'<!DOCTYPE', b'<html')):
                raise ValueError('Checkpoint is truncated or is a download error: ' + rel)
    manifest = root/'weights_manifest.json'
    if manifest.exists() and verify_hashes:
        for rel, info in json.loads(manifest.read_text())['files'].items():
            p = root/rel
            print('Verifying checkpoint:', rel, flush=True)
            if not p.is_file() or p.stat().st_size != info['bytes'] or digest(p) != info['sha256']:
                raise ValueError('Checkpoint checksum mismatch: ' + rel)
    elif not manifest.exists():
        print('No checksum manifest: basic file checks only.')
    return root


def build_manifest(root):
    root = Path(root)
    validate(root, verify_hashes=False)
    files = {}
    for p in root.rglob('*'):
        if p.is_file() and p.name != 'weights_manifest.json' and '.cache' not in p.parts:
            rel = str(p.relative_to(root)).replace('\\', '/')
            print('Fingerprinting checkpoint:', rel, flush=True)
            files[rel] = dict(bytes=p.stat().st_size, sha256=digest(p))
    atomic_json(root/'weights_manifest.json', dict(files=files, note='Local SHA-256 integrity manifest; not vendor signatures.'))


def detect(input_root='/kaggle/input'):
    return sorted({p.parent.parent for p in Path(input_root).glob('**/tencent/Hunyuan3D-2.1')})


def prepare(app_root, destination='/tmp/private-3d-weights', dataset=None, allow_download=False, textures=True):
    app_root, destination = Path(app_root), Path(destination)
    candidates = detect() if dataset is None else [Path(dataset)]
    if len(candidates) > 1:
        raise ValueError('Several weight datasets found; set DATASET explicitly: ' + str(candidates))
    source = candidates[0] if candidates else None
    if source:
        # Dataset reads are slow: check headers now, hash after parallel local staging.
        validate(source, textures=textures, verify_hashes=False)
        if source.resolve() == destination.resolve():
            return validate(source, textures=textures)
        destination.parent.mkdir(parents=True, exist_ok=True)
        manifest = source/'weights_manifest.json'
        provenance = dict(source=str(source.resolve()), manifest_sha256=digest(manifest) if manifest.exists() else None,
                          critical={rel: [int((source/rel).stat().st_size), int((source/rel).stat().st_mtime_ns)]
                                    for rel in (CRITICAL if textures else CRITICAL[:3])})
        if shutil.disk_usage(destination.parent).free > sum(p.stat().st_size for p in source.rglob('*') if p.is_file()) + (5 << 30):
            provenance_path = destination/'stage_provenance.json'
            prior = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
            if not (destination/'_STAGED').exists() or prior != provenance:
                subprocess.run([sys.executable, str(app_root/'workers/stage_weights.py'), str(source), str(destination)], check=True)
                atomic_json(provenance_path, provenance)
            try:
                validate(destination, textures=textures)
            except Exception:
                (destination/'_STAGED').unlink(missing_ok=True)
                raise
            return destination
        print('Insufficient staging disk; loading directly from attached dataset.')
        return validate(source, textures=textures)
    try:
        return validate(destination, textures=textures)
    except FileNotFoundError:
        if not allow_download:
            raise FileNotFoundError('Attach your private weights dataset, or explicitly enable ALLOW_DOWNLOAD.')
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(app_root/'workers/prefetch.py'), str(destination)], check=True)
    build_manifest(destination)
    # The manifest was just computed from these local bytes; do not reread 18 GiB immediately.
    return validate(destination, textures=textures, verify_hashes=False)
