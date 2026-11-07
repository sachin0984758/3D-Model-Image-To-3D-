"""CPU-only tests. Fixtures are synthetic and never represented as model inference."""
import io
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
try:
    import trimesh
except ImportError:
    trimesh = None
from PIL import Image
from studio.storage import atomic_json, prepare_image, center_image, validate_glb, update_stage, metadata, export_all
from studio.weights import CRITICAL, validate, build_manifest, detect, prepare
from studio.runner import Studio, DEFAULTS


class StorageTests(unittest.TestCase):
    def setUp(self):
        temp_root = Path(__file__).resolve().parents[1]/'.test-tmp'
        temp_root.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=temp_root)
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_images(self):
        im = Image.new('RGBA', (100, 200), (0, 0, 0, 0))
        im.paste((255, 0, 0, 255), (20, 30, 80, 180))
        data = io.BytesIO()
        im.save(data, format='PNG')
        a = prepare_image(data.getvalue(), '../../tower.png', self.root)
        b = prepare_image(data.getvalue(), 'other.png', self.root)
        self.assertEqual(a['sha256'], b['sha256'])
        self.assertEqual(a['filename'], 'tower.png')
        centered = center_image(im)
        box = centered.getchannel('A').getbbox()
        self.assertEqual(box[2]-box[0], 60)
        self.assertEqual(box[3]-box[1], 150)
        self.assertEqual(centered.width, centered.height)
        with self.assertRaises(ValueError):
            prepare_image(b'fake', 'bad.exe', self.root)
        with self.assertRaises(Exception):
            prepare_image(b'fake', 'bad.png', self.root)
        with self.assertRaises(ValueError):
            center_image(Image.new('RGBA', (50, 50)))

    @unittest.skipIf(trimesh is None, 'trimesh unavailable; package download blocked in local sandbox')
    def test_glb_validation(self):
        mesh = trimesh.creation.box()
        path = self.root/'geometry.glb'
        mesh.export(path)
        self.assertEqual(validate_glb(path)['faces'], 12)
        with self.assertRaises(ValueError):
            validate_glb(path, textured=True)
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaises(ValueError):
            validate_glb(path)

    @unittest.skipIf(trimesh is None, 'trimesh unavailable; package download blocked in local sandbox')
    def test_pbr_validation(self):
        mesh = trimesh.creation.box()
        image = Image.new('RGB', (16, 16), 'red')
        material = trimesh.visual.material.PBRMaterial(baseColorTexture=image, metallicRoughnessTexture=image)
        mesh.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(mesh.vertices), 2)), material=material)
        path = self.root/'fixture.glb'
        mesh.export(path)
        report = validate_glb(path, textured=True)
        self.assertTrue(report['pbr_verified'])
        self.assertGreaterEqual(report['embedded_images'], 1)

    def test_independent_stage_records_and_checkpoint(self):
        out = self.root/'outputs/model_001'
        out.mkdir(parents=True)
        atomic_json(out/'job.json', {'name': 'model_001'})
        update_stage(out, 'shape', status='completed', duration=42.5, peak_vram_gb=7.6)
        update_stage(out, 'paint', status='failed', error='fixture error')
        update_stage(out, 'shape', status='completed', resumed=True)
        self.assertEqual(metadata(out)['shape']['duration'], 42.5)
        self.assertEqual(metadata(out)['shape']['peak_vram_gb'], 7.6)
        self.assertEqual(metadata(out)['shape']['status'], 'completed')
        self.assertEqual(metadata(out)['paint']['status'], 'failed')
        zip_path = export_all(self.root)
        import zipfile
        with zipfile.ZipFile(zip_path) as z:
            self.assertIn('outputs/model_001/metadata.json', z.namelist())
        self.assertFalse(list(out.glob('*.tmp')))

    def test_rejects_truncated_glb_without_model_dependencies(self):
        path = self.root/'bad.glb'
        for data in (b'', b'not a GLB'*4, struct.pack('<4sII', b'glTF', 2, 999)+b'\0'*32):
            path.write_bytes(data)
            with self.assertRaises(ValueError):
                validate_glb(path)

    def test_download_link_is_relative_to_notebook_directory(self):
        from studio.ui import download_link
        target = self.root/'generated_models.zip'
        target.write_bytes(b'fixture')
        relative = target.resolve().relative_to(Path.cwd().resolve()).as_posix()
        self.assertIn('href=\'' + relative + '\'', download_link(target)._repr_html_())

    def test_concurrent_checkpoint_exports(self):
        import threading
        import zipfile
        from concurrent.futures import ThreadPoolExecutor
        out = self.root/'outputs/model_001'
        out.mkdir(parents=True)
        atomic_json(out/'job.json', {'name': 'model_001'})
        (out/'input.png').write_bytes(b'checkpoint fixture')
        barrier = threading.Barrier(2)
        def checkpoint(_):
            barrier.wait(timeout=10)
            return export_all(self.root)

        with ThreadPoolExecutor(2) as pool:
            exports = list(pool.map(checkpoint, range(2)))
        self.assertEqual(exports[0], exports[1])
        with zipfile.ZipFile(exports[0]) as archive:
            self.assertIsNone(archive.testzip())
            self.assertIn('outputs/model_001/input.png', archive.namelist())
        self.assertFalse(list(exports[0].parent.glob('*.tmp')))

    @unittest.skipIf(trimesh is None, 'trimesh unavailable')
    def test_incomplete_worker_exports_retry_and_completed_exports_resume(self):
        """Exercise both real job loops with synthetic meshes and mocked model calls."""
        import ast
        import glob
        import time
        import traceback
        queue, out = self.root/'queue', self.root/'outputs/model_001'
        queue.mkdir()
        out.mkdir(parents=True)
        (queue/'_SHAPE_DONE').touch()
        image = Image.new('RGBA', (64, 64))
        image.paste('red', (16, 16, 48, 48))
        source = self.root/'source.png'
        image.save(source)
        image.save(out/'input.png')
        (out/'white_mesh.glb').write_bytes(b'partial')
        (out/'white_mesh.obj').write_bytes(b'partial')
        mesh = trimesh.creation.box()
        torch = Mock()
        torch.cuda.max_memory_allocated.return_value = 1024
        torch.cuda.OutOfMemoryError = type('FixtureOOM', (Exception,), {})
        cfg = dict(DEFAULTS, out=str(out.parent), queue=str(queue),
                   images=[dict(name=out.name, path=str(source))])
        context = dict(cfg=cfg, os=os, json=json, time=time, traceback=traceback,
                       torch=torch, trimesh=trimesh, Image=Image, center_image=center_image,
                       update_stage=update_stage, validate_glb=validate_glb,
                       pipe=Mock(return_value=[mesh]), face_reducer=lambda mesh, **_: mesh,
                       enqueue=Mock(), log=Mock())
        workers = Path(__file__).resolve().parents[1]/'workers'
        shape = ast.parse((workers/'shape_worker.py').read_text())
        shape_loop = ast.Module(body=[next(n for n in shape.body if isinstance(n, ast.For))], type_ignores=[])
        for _ in range(2):
            exec(compile(shape_loop, 'shape_worker.py', 'exec'), context)
        self.assertEqual(context['pipe'].call_count, 1)
        shape_status = json.loads((out/'shape_status.json').read_text())
        self.assertEqual(shape_status['status'], 'completed', shape_status)
        self.assertTrue(shape_status['resumed'])
        self.assertGreater(shape_status['validation']['faces'], 0)
        (out/'white_mesh.obj').write_bytes(b'partial')
        exec(compile(shape_loop, 'shape_worker.py', 'exec'), context)
        self.assertEqual(context['pipe'].call_count, 2)
        self.assertEqual(json.loads((out/'shape_status.json').read_text())['status'], 'completed')

        material = trimesh.visual.material.PBRMaterial(baseColorTexture=image, metallicRoughnessTexture=image)
        mesh.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(mesh.vertices), 2)), material=material)
        (out/'textured_mesh.glb').write_bytes(b'partial')
        context.update(glob=glob, pipe=Mock(return_value=str(out/'textured_mesh.obj')),
                       embed_textures=Mock(),
                       create_glb_with_pbr_materials=Mock(side_effect=lambda obj, textures, dest: mesh.export(dest)))
        paint = ast.parse((workers/'paint_worker.py').read_text())
        paint_loop = ast.Module(body=[
            next(n for n in paint.body if isinstance(n, ast.FunctionDef) and n.name == 'next_job'),
            next(n for n in paint.body if isinstance(n, ast.While))], type_ignores=[])
        for _ in range(2):
            atomic_json(queue/(out.name+'.json'), dict(name=out.name, dir=str(out)))
            exec(compile(paint_loop, 'paint_worker.py', 'exec'), context)
        self.assertEqual(context['pipe'].call_count, 1)
        paint_status = json.loads((out/'paint_status.json').read_text())
        self.assertEqual(paint_status['status'], 'completed')
        self.assertTrue(paint_status['resumed'])
        self.assertTrue(paint_status['validation']['pbr_verified'])
        self.assertFalse(list(queue.glob('*.processing')))

    def test_weights_integrity_and_detection(self):
        for rel in CRITICAL:
            path = self.root/rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'0'*2048)
        build_manifest(self.root)
        self.assertEqual(validate(self.root), self.root)
        self.assertIn(self.root, detect(self.root))
        (self.root/CRITICAL[0]).write_bytes(b'1'*2048)
        with self.assertRaises(ValueError):
            validate(self.root)

    def test_no_gpu_fails_without_launch(self):
        studio = Studio(self.root, self.root, self.root, self.root)
        with patch('studio.runner.gpu_info', return_value=[]):
            with self.assertRaisesRegex(RuntimeError, 'No NVIDIA GPU'):
                studio.run([])
        self.assertFalse(studio.lock.locked())
        self.assertFalse(studio.children)

    def test_automatic_weights_download_then_cache_reuse(self):
        cache = self.root/'working/private-3d-weights'

        def download(args, **kwargs):
            self.assertEqual(Path(args[-1]), cache)
            for rel in CRITICAL:
                path = cache/rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'0'*2048)

        with patch('studio.weights.detect', return_value=[]), \
             patch('studio.weights.subprocess.run', side_effect=download) as downloader:
            first = prepare(self.root, cache, allow_download=True)
            second = prepare(self.root, cache, allow_download=True)
        self.assertEqual(first, cache)
        self.assertEqual(second, cache)
        self.assertEqual(downloader.call_count, 1)
        self.assertTrue((cache/'weights_manifest.json').is_file())
        self.assertEqual(validate(cache), cache)

    def test_weights_source_can_be_reused_in_place(self):
        for rel in CRITICAL:
            path = self.root/rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'0'*2048)
        build_manifest(self.root)
        original = (self.root/CRITICAL[0]).read_bytes()
        with patch('studio.weights.subprocess.run') as staging:
            self.assertEqual(prepare(self.root, self.root, dataset=self.root), self.root)
        staging.assert_not_called()
        self.assertEqual((self.root/CRITICAL[0]).read_bytes(), original)

    def test_staging_invalidates_changed_dataset(self):
        import shutil
        from types import SimpleNamespace
        source, destination = self.root/'dataset', self.root/'staged'
        for rel in CRITICAL:
            p = source/rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'0'*2048)
        build_manifest(source)
        copies = []
        def stage(args, **kwargs):
            copies.append(args)
            shutil.copytree(source, destination, dirs_exist_ok=True)
            (destination/'_STAGED').touch()
            return SimpleNamespace(returncode=0)
        with patch('studio.weights.shutil.disk_usage', return_value=SimpleNamespace(free=1<<40)), \
             patch('studio.weights.subprocess.run', side_effect=stage):
            prepare(self.root, destination, dataset=source)
            prepare(self.root, destination, dataset=source)
            self.assertEqual(len(copies), 1)
            (source/CRITICAL[0]).write_bytes(b'1'*2048)
            build_manifest(source)
            prepare(self.root, destination, dataset=source)
            self.assertEqual(len(copies), 2)
            self.assertEqual((destination/CRITICAL[0]).read_bytes(), b'1'*2048)

    def test_worker_inference_contract(self):
        root = Path(__file__).resolve().parents[1]
        shape = (root/'workers/shape_worker.py').read_text()
        paint = (root/'workers/paint_worker.py').read_text()
        self.assertIn('Hunyuan3DDiTFlowMatchingPipeline.from_pretrained', shape)
        self.assertIn('Hunyuan3DPaintPipeline(conf)', paint)
        self.assertIn('create_glb_with_pbr_materials(', paint)
        self.assertIn('validate_glb(glb_path + ".pending.glb", textured=True)', paint)
        self.assertIn('OUT OF GPU MEMORY', paint)


if __name__ == '__main__':
    unittest.main(verbosity=2)
