"""Owns only its child workers. Parent never imports torch or allocates CUDA memory."""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from .storage import atomic_json, metadata, validate_glb, export_all

DEFAULTS = dict(steps=30, guidance_scale=5.0, octree_resolution=256, num_chunks=8000,
                seed=1234, max_faces=40000, texture_views=6, texture_resolution=512,
                force_rembg=False, remove_background=True, skip_existing=True, textures=True)
PRESETS = {'Low': dict(steps=20, octree_resolution=192, max_faces=20000),
           'Medium': dict(steps=30, octree_resolution=256, max_faces=40000),
           'High': dict(steps=40, octree_resolution=320, max_faces=80000)}


def gpu_info():
    try:
        p = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=15)
        return [dict(zip(('index', 'name', 'total_mb', 'used_mb', 'utilization'), [x.strip() for x in line.split(',')]))
                for line in p.stdout.strip().splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


class Studio:
    def __init__(self, root, app_root, weights, repo='/tmp/Hunyuan3D-2.1'):
        self.root, self.app_root, self.weights, self.repo = map(Path, (root, app_root, weights, repo))
        for d in ('outputs', 'logs', 'exports', 'uploads', 'runs'):
            (self.root/d).mkdir(parents=True, exist_ok=True)
        self.children = []
        self.cancelled = threading.Event()
        self.lock = threading.Lock()
        self.state = 'Idle'

    def stop(self):
        self.cancelled.set()
        for p in list(self.children):
            if p.poll() is None:
                p.terminate()

    def jobs(self):
        return [metadata(p) for p in sorted((self.root/'outputs').glob('model_*')) if (p/'job.json').exists()]

    def run(self, images, settings=None, progress=print):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('A batch is already running.')
        try:
            self.cancelled.clear()
            return self._run(images, settings, progress)
        finally:
            for p in self.children:
                if p.poll() is None:
                    p.terminate()
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
            self.children.clear()
            self.lock.release()

    def _run(self, images, settings, progress):
        cfg = dict(DEFAULTS, **(settings or {}))
        devices = gpu_info()
        if not devices:
            raise RuntimeError('No NVIDIA GPU. Enable a GPU runtime in Kaggle or Colab.')
        paint_gpu = 1 if len(devices) > 1 else 0
        if cfg['textures'] and paint_gpu == 0:
            progress('One GPU detected. Geometry will finish before the paint worker starts.')
        if not (1 <= cfg['steps'] <= 60 and 128 <= cfg['octree_resolution'] <= 320 and
                1000 <= cfg['max_faces'] <= 80000 and 0 <= cfg['seed'] <= 2**32-1 and
                0 <= cfg['guidance_scale'] <= 15 and 500 <= cfg['num_chunks'] <= 16000):
            raise ValueError('Settings exceed the conservative T4 limits.')
        if cfg['texture_resolution'] not in (256, 512) or cfg['texture_views'] not in (6, 8):
            raise ValueError('Unsupported texture configuration.')
        if float(devices[0]['total_mb']) < 14000 or float(devices[0]['total_mb']) - float(devices[0]['used_mb']) < 10000:
            raise RuntimeError('Insufficient free GPU memory. Stop other workloads and retry.')
        if cfg['textures'] and float(devices[paint_gpu]['total_mb']) - float(devices[paint_gpu]['used_mb']) < 10000:
            raise RuntimeError('The paint GPU has less than 10 GB free. Free GPU memory and retry.')
        queue = self.root/'runs'/uuid.uuid4().hex
        queue.mkdir()
        cfg.update(repo=str(self.repo), queue=str(queue), out=str(self.root/'outputs'), weights=str(self.weights))
        cfg['images'] = []
        existing = self.jobs()
        for image in images:
            # Identity includes settings, so adjusting quality never overwrites an old model.
            prior = next((j for j in existing if j['input']['sha256'] == image['sha256']
                          and dict(DEFAULTS, **j['settings']) == {k: cfg[k] for k in DEFAULTS}), None)
            if prior:
                name = prior['name']
            else:
                used = {p.name for p in (self.root/'outputs').iterdir()}
                i = 1
                while f'model_{i:03d}' in used:
                    i += 1
                name = f'model_{i:03d}'
                out = self.root/'outputs'/name
                out.mkdir()
                atomic_json(out/'job.json', dict(name=name, input=image, checkpoint='tencent/Hunyuan3D-2.1',
                    shape_subfolder='hunyuan3d-dit-v2-1', paint_subfolder='hunyuan3d-paintpbr-v2-1',
                    settings={k: cfg[k] for k in DEFAULTS}, created=time.time()))
            cfg['images'].append(dict(name=name, path=image['path']))
        if not cfg['images']:
            raise ValueError('Upload at least one image.')
        config_path = queue.with_suffix('.json')
        atomic_json(config_path, cfg)

        def launch(stage, gpu):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HY3DGEN_MODELS=str(self.weights),
                U2NET_HOME=str(self.weights/'u2net'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                HF_DATASETS_OFFLINE='1', PYTHONUNBUFFERED='1',
                PYTHONPATH=str(self.app_root)+os.pathsep+os.environ.get('PYTHONPATH', ''))
            log = self.root/'logs'/(stage+'_worker.log')
            with log.open('a') as f:
                f.write('\nRUN '+queue.name+'\n')
                f.flush()
                p = subprocess.Popen([sys.executable, str(self.app_root/'workers'/(stage+'_worker.py')), str(config_path)],
                    stdout=f, stderr=subprocess.STDOUT, env=env)
            self.children.append(p)
            return p

        self.state = 'Loading shape model'
        shape, paint = launch('shape', 0), None
        started, last_report = time.monotonic(), 0
        while True:
            if self.cancelled.is_set():
                self.state = 'Cancelled; intermediate files preserved'
                break
            if time.monotonic()-started > 4*3600:
                raise TimeoutError('Batch exceeded four hours. Intermediate work is preserved.')
            if shape.poll() is not None:
                (queue/'_SHAPE_DONE').touch()
            if (cfg['textures'] and paint is None and list(queue.glob('*.json'))
                    and (paint_gpu != 0 or shape.poll() is not None)):
                paint = launch('paint', paint_gpu)
            if shape.poll() is not None and (paint is None or paint.poll() is not None):
                break
            if time.monotonic()-last_report > 15:
                last_report = time.monotonic()
                self.state = ('Loading paint model' if paint is not None and not (queue/'_PAINT_READY').exists()
                              else 'Generating' if (queue/'_SHAPE_READY').exists() else 'Loading shape model')
                progress(dict(state=self.state, shape_exit=shape.poll(), paint_exit=paint.poll() if paint else None, gpus=gpu_info()))
            time.sleep(1)
        # Record process crashes without fabricating successful inference.
        from .storage import update_stage
        for item in cfg['images']:
            out = self.root/'outputs'/item['name']
            try:
                validate_glb(out/'white_mesh.glb')
            except Exception as e:
                prior_path = out/'shape_status.json'
                prior = json.loads(prior_path.read_text()) if prior_path.exists() else {}
                if prior.get('status') != 'failed':
                    update_stage(out, 'shape', status='failed', error=str(e), worker_exit=shape.poll())
            shape_record = out/'shape_status.json'
            if shape_record.exists() and json.loads(shape_record.read_text()).get('status') == 'failed':
                update_stage(out, 'paint', status='not_attempted', reason='Geometry failed; see shape error and worker log')
                metadata(out)
                continue
            if cfg['textures']:
                try:
                    validate_glb(out/'textured_mesh.glb', textured=True)
                except Exception as e:
                    old = out/'paint_status.json'
                    if not old.exists() or json.loads(old.read_text()).get('status') not in ('failed', 'out_of_memory'):
                        update_stage(out, 'paint', status='failed', error=str(e), worker_exit=paint.poll() if paint else None)
            else:
                update_stage(out, 'paint', status='disabled', reason='Geometry-only mode')
            metadata(out)
        self.state = 'Finished; inspect per-model status' if not self.cancelled.is_set() else self.state
        return self.jobs()
