"""Atomic state, image validation and export validation. No model imports."""
import hashlib
import io
import json
import os
import struct
import threading
import uuid
import zipfile
from pathlib import Path

MODEL = "tencent/Hunyuan3D-2.1"
_export_lock = threading.Lock()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def update_stage(out, stage, **values):
    # Each worker owns a different record, avoiding concurrent read-modify-write loss.
    path = Path(out) / (stage + '_status.json')
    previous = json.loads(path.read_text()) if values.get('resumed') and path.exists() else {}
    atomic_json(path, dict(previous, stage=stage, **values))


def prepare_image(data, name, directory):
    from PIL import Image, ImageOps
    if Path(name).suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
        raise ValueError('Choose PNG, JPG, JPEG or WebP.')
    if len(data) > 40 * 1024**2:
        raise ValueError('Image exceeds 40 MiB.')
    with Image.open(io.BytesIO(data)) as check:
        check.verify()
    with Image.open(io.BytesIO(data)) as original:
        image = ImageOps.exif_transpose(original).convert('RGBA')
    if min(image.size) < 32 or image.width * image.height > 32_000_000:
        raise ValueError('Image must be at least 32 pixels per side and under 32 megapixels.')
    image.thumbnail((2048, 2048))
    identity = hashlib.sha256(data).hexdigest()
    directory = Path(directory) / identity[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'source.png'
    image.save(path)
    return dict(path=str(path), filename=Path(name).name, sha256=identity)


def center_image(image):
    from PIL import Image
    image = image.convert('RGBA')
    box = image.getchannel('A').getbbox()
    if box is None:
        raise ValueError('Image is completely transparent.')
    obj = image.crop(box)
    side = max(obj.size)
    canvas = Image.new('RGBA', (round(side * 1.2),) * 2, (255, 255, 255, 0))
    canvas.alpha_composite(obj, ((canvas.width - obj.width)//2, (canvas.height - obj.height)//2))
    return canvas


def validate_glb(path, textured=False):
    """Check container, buffers, actual triangles, UVs and embedded PBR assignments."""
    path = Path(path)
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError('Truncated GLB')
    magic, version, length = struct.unpack_from('<4sII', data)
    if magic != b'glTF' or version != 2 or length != len(data):
        raise ValueError('Invalid GLB header')
    n, kind = struct.unpack_from('<II', data, 12)
    if kind != 0x4E4F534A or 20+n > len(data):
        raise ValueError('Invalid GLB JSON chunk')
    doc = json.loads(data[20:20+n])
    binary = data[20+n:]
    if len(binary) < 8:
        raise ValueError('GLB has no binary chunk')
    bin_size, bin_type = struct.unpack_from('<II', binary)
    if bin_type != 0x004E4942 or bin_size != len(binary)-8:
        raise ValueError('Invalid binary chunk')
    for view in doc.get('bufferViews', []):
        if view.get('byteOffset', 0) + view['byteLength'] > bin_size:
            raise ValueError('Buffer view exceeds binary chunk')
    if any(b.get('uri') for b in doc.get('buffers', [])):
        raise ValueError('GLB references external buffers')
    primitives = [p for m in doc.get('meshes', []) for p in m.get('primitives', [])]
    if not primitives:
        raise ValueError('No mesh primitives')
    if textured:
        images = doc.get('images', [])
        if not images or any(i.get('bufferView') is None or i.get('mimeType') not in {'image/jpeg', 'image/png'} for i in images):
            raise ValueError('PBR images are missing or not embedded')
        for p in primitives:
            if 'TEXCOORD_0' not in p.get('attributes', {}):
                raise ValueError('Textured primitive has no UV coordinates')
            mat = doc.get('materials', [])[p.get('material', -1)] if 'material' in p else {}
            pbr = mat.get('pbrMetallicRoughness', {})
            if 'baseColorTexture' not in pbr or 'metallicRoughnessTexture' not in pbr:
                raise ValueError('Missing albedo or metallic/roughness texture')
    import trimesh
    scene = trimesh.load(path, force='scene', process=False)
    faces = sum(len(g.faces) for g in scene.geometry.values() if hasattr(g, 'faces'))
    if not faces:
        raise ValueError('No triangles')
    return dict(bytes=len(data), faces=faces, embedded_images=len(doc.get('images', [])), pbr_verified=textured)


def metadata(out):
    out = Path(out)
    doc = json.loads((out/'job.json').read_text())
    for stage in ('shape', 'paint'):
        path = out/(stage+'_status.json')
        doc[stage] = json.loads(path.read_text()) if path.exists() else {'status': 'pending'}
    doc['outputs'] = [str(p.relative_to(out)) for p in out.rglob('*')
                      if p.is_file() and p.name != 'metadata.json' and not p.name.startswith('.')]
    atomic_json(out/'metadata.json', doc)
    return doc


def export_all(root):
    with _export_lock:
        root = Path(root)
        target = root/'exports'/'generated_models.zip'
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name('.' + target.name + '.' + uuid.uuid4().hex + '.tmp')
        with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
            for out in sorted((root/'outputs').glob('model_*')):
                if (out/'job.json').exists():
                    metadata(out)
                    # Include failed records/intermediates as well, for checkpoint recovery.
                    for p in out.rglob('*'):
                        if p.is_file() and not p.name.startswith('.'):
                            z.write(p, p.relative_to(root))
            for p in (root/'logs').glob('*'):
                if p.is_file():
                    z.write(p, p.relative_to(root))
        os.replace(tmp, target)
        return target
