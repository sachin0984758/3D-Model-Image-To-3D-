"""Portable scheduling, interface and exports using synthetic CPU fixtures."""
import io
import json
import sys
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import numpy as np
import trimesh
from PIL import Image

from studio.exports import export_model
from studio.runner import DEFAULTS, Studio
from studio.storage import atomic_json, prepare_image, update_stage, validate_glb


class PortableStudioTests(unittest.TestCase):
    def setUp(self):
        temp_root = Path(__file__).resolve().parents[1]/'.test-tmp'
        temp_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def reference(self):
        image = Image.new('RGBA', (64, 64))
        image.paste('red', (16, 8, 48, 56))
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return prepare_image(buffer.getvalue(), 'reference.png', self.root/'uploads')

    def fixture(self, folder, detail=False):
        folder.mkdir(parents=True, exist_ok=True)
        mesh = trimesh.creation.icosphere(subdivisions=2) if detail else trimesh.creation.box()
        mesh.export(folder/'white_mesh.glb')
        # Nondegenerate spherical UVs also exercise the texture-preserving simplifier.
        vertices = mesh.vertices
        uv = np.column_stack((np.arctan2(vertices[:, 1], vertices[:, 0])/(2*np.pi)+.5,
                              np.arccos(np.clip(vertices[:, 2]/np.linalg.norm(vertices, axis=1), -1, 1))/np.pi))
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new('RGB', (16, 16), 'red'),
            metallicRoughnessTexture=Image.new('RGB', (16, 16), (0, 128, 64)),
            roughnessFactor=.7, metallicFactor=.35)
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
        mesh.export(folder/'textured_mesh.glb')
        return mesh

    def run_with_devices(self, count, legacy=False):
        studio = Studio(self.root, self.root, self.root, self.root)
        image = self.reference()
        if legacy:
            settings = dict(DEFAULTS)
            settings.pop('remove_background')
            atomic_json(self.root/'outputs/model_001/job.json',
                        dict(name='model_001', input=image, settings=settings))
        events = []
        processes = {}

        test = self

        class Process:
            def __init__(self, args, **kwargs):
                self.stage = Path(args[1]).stem.split('_')[0]
                self.polls, self.exited = 0, False
                self.config = json.loads(Path(args[-1]).read_text())
                self.returncode = None
                events.append((self.stage, kwargs['env']['CUDA_VISIBLE_DEVICES']))
                processes[self.stage] = self
                if self.stage == 'paint':
                    test.assertEqual(processes['shape'].exited, count == 1)
                    test.assertTrue(self.config['textures'])
                for item in self.config['images']:
                    output = Path(self.config['out'])/item['name']
                    if self.stage == 'shape':
                        test.fixture(output)
                        atomic_json(Path(self.config['queue'])/(item['name']+'.json'),
                                    dict(name=item['name'], dir=str(output)))
                    update_stage(output, self.stage, status='completed')

            def poll(self):
                self.polls += 1
                if self.stage == 'paint' or self.polls >= 6:
                    self.exited, self.returncode = True, 0
                return self.returncode

            def wait(self, timeout=None):
                self.exited, self.returncode = True, 0
                return 0

            terminate = wait
            kill = wait

        devices = [dict(total_mb='15360', used_mb='0') for _ in range(count)]
        with patch('studio.runner.gpu_info', return_value=devices), \
             patch('studio.runner.subprocess.Popen', side_effect=Process), \
             patch('studio.runner.time.sleep'):
            jobs = studio.run([image], progress=lambda _: None)
        self.assertEqual(events, [('shape', '0'), ('paint', str(min(1, count-1)))])
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]['paint']['status'], 'completed')
        self.assertFalse(studio.lock.locked())
        self.assertEqual(studio.children, [])
        return jobs

    def test_single_gpu_finishes_shape_before_loading_paint(self):
        self.run_with_devices(1)

    def test_dual_gpu_starts_paint_while_shape_worker_is_alive(self):
        self.run_with_devices(2)

    def test_previous_job_settings_resume_without_duplicate_model_folder(self):
        jobs = self.run_with_devices(1, legacy=True)
        self.assertEqual(jobs[0]['name'], 'model_001')
        self.assertEqual(len(list((self.root/'outputs').glob('model_*'))), 1)

    def test_all_geometry_export_formats_and_pbr_bundles(self):
        model = self.root/'model_001'
        self.fixture(model)
        before = {p.name: p.read_bytes() for p in model.iterdir()}
        for kind in ('glb', 'obj', 'ply', 'stl'):
            with self.subTest(kind=kind):
                path = export_model(model, self.root/'exports', kind, include_texture=False)
                self.assertEqual(len(trimesh.load(path, force='mesh').faces), 12)
        glb = export_model(model, self.root/'exports', 'glb')
        self.assertTrue(validate_glb(glb, textured=True)['pbr_verified'])
        obj = export_model(model, self.root/'exports', 'obj')
        with zipfile.ZipFile(obj) as archive:
            self.assertIsNone(archive.testzip())
            self.assertTrue(any(name.endswith('.obj') for name in archive.namelist()))
            mtl = next(name for name in archive.namelist() if name.endswith('.mtl'))
            self.assertIn(b'map_Kd', archive.read(mtl))
            self.assertIn(b'map_Pr roughness.png', archive.read(mtl))
            self.assertIn(b'map_Pm metallic.png', archive.read(mtl))
            self.assertIn(b'Pr 0.7', archive.read(mtl))
            self.assertIn(b'Pm 0.35', archive.read(mtl))
            self.assertIn('roughness.png', archive.namelist())
            self.assertIn('metallic.png', archive.namelist())
        self.assertEqual(before, {p.name: p.read_bytes() for p in model.iterdir()})
        for kind in ('ply', 'stl'):
            with self.assertRaisesRegex(ValueError, 'GLB or OBJ'):
                export_model(model, self.root/'exports', kind, include_texture=True)

    def test_simplification_retains_uvs_and_pbr_material(self):
        model = self.root/'model_001'
        mesh = self.fixture(model, detail=True)
        for texture in (False, True):
            with self.subTest(texture=texture):
                result = export_model(model, self.root/'exports', 'glb', texture, max_faces=160)
                report = validate_glb(result, textured=texture)
                self.assertGreater(report['faces'], 0)
                self.assertLess(report['faces'], len(mesh.faces))
        self.assertEqual(validate_glb(model/'textured_mesh.glb')['faces'], len(mesh.faces))

    def test_colab_download_uses_native_helper(self):
        from studio.ui import download_file
        colab = ModuleType('google.colab')
        colab.files = Mock()
        path = self.root/'model.zip'
        path.write_bytes(b'fixture')
        with patch.dict(sys.modules, {'google.colab': colab}):
            self.assertIn('Download started', download_file(path)._repr_html_())
        colab.files.download.assert_called_once_with(str(path))

    def test_widget_generation_controls_and_export_download(self):
        from studio.ui import show
        studio = Studio(self.root, self.root, self.root, self.root)
        sample = self.root/'assets/example_images/example.png'
        sample.parent.mkdir(parents=True)
        Image.new('RGB', (64, 64), 'red').save(sample)
        calls = []

        def generate(images, settings, progress):
            calls.append((images, settings))
            folder = self.root/'outputs/model_001'
            self.fixture(folder)
            atomic_json(folder/'job.json', dict(name='model_001', input=images[0], settings=settings))
            update_stage(folder, 'shape', status='completed')
            update_stage(folder, 'paint', status='completed' if settings['textures'] else 'disabled')
            progress('Synthetic fixture complete')
            return studio.jobs()

        class InlineThread:
            def __init__(self, target, **kwargs):
                self.target = target

            def start(self):
                self.target()

        with patch('IPython.display.display') as display, \
             patch('studio.ui.gpu_info', return_value=[{}]), \
             patch('studio.ui.threading.Thread', InlineThread), \
             patch.object(studio, 'run', side_effect=generate), \
             patch('studio.ui.preview', return_value='fixture viewer'), \
             patch('studio.ui.download_file', return_value='fixture download') as download:
            show(studio)
            app = display.call_args.args[0]

            def walk(widget):
                yield widget
                for child in getattr(widget, 'children', ()):
                    yield from walk(child)

            widgets = list(walk(app))
            def description(text):
                return next(w for w in widgets if getattr(w, 'description', None) == text)

            def field(text):
                return next(w for w in widgets if getattr(w, 'tooltip', None) == text)

            field('example.png').click()
            description('Remove background').value = False
            field('Geometry preset').value = 'Low'
            description('Generate Shape').click()
            self.assertFalse(calls[-1][1]['textures'])
            self.assertFalse(calls[-1][1]['remove_background'])
            self.assertEqual(calls[-1][1]['steps'], 20)
            self.assertEqual(calls[-1][1]['octree_resolution'], 192)
            description('Generate Textured Shape').click()
            self.assertTrue(calls[-1][1]['textures'])
            self.assertFalse(description('Generate Shape').disabled)
            self.assertEqual(description('Model').value, 'model_001')
            field('Format').value = 'STL'
            self.assertFalse(description('Include textures').value)
            self.assertTrue(description('Include textures').disabled)
            description('Prepare export').click()
            self.assertFalse(description('Download').disabled)
            description('Download').click()
            self.assertEqual(download.call_args.args[0].suffix, '.stl')
            field('Format').value = 'GLB'
            self.assertTrue(description('Download').disabled)
            self.assertFalse(description('Include textures').disabled)


if __name__ == '__main__':
    unittest.main(verbosity=2)
