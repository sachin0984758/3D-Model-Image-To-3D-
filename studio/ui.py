"""Notebook-only widgets; no server, public tunnel or image API."""
import base64
import html
import json
import threading
from pathlib import Path
from .runner import DEFAULTS, PRESETS, gpu_info
from .storage import prepare_image, export_all


def download_link(path):
    """Use paths relative to the notebook's working directory for Kaggle file serving."""
    from IPython.display import FileLink
    relative = Path(path).resolve().relative_to(Path.cwd().resolve())
    return FileLink(relative.as_posix())


def preview(path, viewer_js):
    from IPython.display import HTML
    if not Path(viewer_js).exists():
        raise FileNotFoundError('Local viewer bundle missing. Rerun setup; GLB download remains available.')
    encoded = base64.b64encode(Path(path).read_bytes()).decode()
    # Isolate the viewer, permit only inline JS/data, and disallow outbound connections.
    js = Path(viewer_js).read_text(encoding='utf-8').replace('</script', '<\\/script')
    doc = ('<!doctype html><meta http-equiv="Content-Security-Policy" '
           'content="default-src \'none\'; script-src \'unsafe-inline\' blob:; '
           'style-src \'unsafe-inline\'; img-src data: blob:; connect-src data: blob:; worker-src blob:">'
           '<style>body{margin:0;background:#101722;color:#ddd;font-family:system-ui}'
           'model-viewer{width:100%;height:460px}button{margin:8px;padding:8px;background:#28384c;color:white}</style>'
           '<model-viewer id="v" camera-controls touch-action="none" shadow-intensity="1" '
           'src="data:model/gltf-binary;base64,'+encoded+'"></model-viewer>'
           '<button onclick="v.cameraOrbit=\'0deg 75deg auto\';v.cameraTarget=\'auto auto auto\';v.fieldOfView=\'auto\'">Reset camera</button>'
           '<span>Drag: rotate · wheel: zoom · Shift + drag: pan</span><script type="module">'+js+'</script>')
    return HTML('<iframe sandbox="allow-scripts" style="width:100%;height:520px;border:0" srcdoc="'+html.escape(doc, quote=True)+'"></iframe>')


def show(studio):
    import ipywidgets as w
    from IPython.display import display, HTML, clear_output, Image
    title = w.HTML('<div style="background:#111b2b;color:#e8f2ff;padding:24px;border-radius:14px">'
                   '<h1 style="margin:0">PRIVATE 3D STUDIO</h1>'
                   '<p>Personal image-to-3D workspace · local checkpoint inference</p></div>')
    dashboard, messages, results, originals = w.Output(), w.Output(), w.Output(), w.Output()
    upload = w.FileUpload(accept='.png,.jpg,.jpeg,.webp', multiple=True, description='Upload images')
    preset = w.Dropdown(options=list(PRESETS), value='Medium', description='Geometry')
    texture_quality = w.Dropdown(options=['Low (6 / 256)', 'Medium (6 / 512)', 'High (8 / 512)'],
                                  value='Medium (6 / 512)', description='Textures')
    steps = w.IntSlider(min=1, max=60, value=30, description='Steps')
    resolution = w.IntSlider(min=128, max=320, step=64, value=256, description='Octree')
    faces = w.BoundedIntText(min=1000, max=80000, value=40000, description='Max faces')
    seed = w.BoundedIntText(min=0, max=2**32-1, value=1234, description='Seed')
    guidance = w.FloatSlider(min=0, max=15, step=.5, value=5, description='Guidance')
    textures = w.Checkbox(value=len(gpu_info()) >= 2, description='Enable PBR textures')
    rembg = w.Checkbox(value=False, description='Remove background again (also transparent PNGs)')
    generate = w.Button(description='Generate 3D', button_style='primary', icon='cube')
    stop = w.Button(description='Stop workers', button_style='warning')
    refresh = w.Button(description='Refresh dashboard', icon='refresh')
    all_download = w.Button(description='Download All / checkpoint', icon='download')
    selections = w.Dropdown(options=[], description='Model')
    stage = w.Dropdown(options=['textured_mesh.glb', 'white_mesh.glb'], description='Preview')
    view = w.Button(description='Preview / downloads')
    images = []

    def log(value):
        with messages:
            print(json.dumps(value) if isinstance(value, dict) else value)

    def update(_=None):
        jobs = studio.jobs()
        with dashboard:
            clear_output(wait=True)
            print('State:', studio.state, '| Uploaded:', len(images), '| Geometry completed:',
                  sum(j['shape'].get('status') == 'completed' for j in jobs), '| Failed geometry:',
                  sum(j['shape'].get('status') == 'failed' for j in jobs), '| PBR completed:',
                  sum(j['paint'].get('status') == 'completed' for j in jobs), '| PBR failures:',
                  sum(j['paint'].get('status') in ('failed', 'out_of_memory') for j in jobs))
            for gpu in gpu_info():
                print('GPU {index}: {name} | {used_mb}/{total_mb} MiB | utilization {utilization}%'.format(**gpu))
        selections.options = [j['name'] for j in jobs]

    def received(change):
        nonlocal images
        values = upload.value.values() if isinstance(upload.value, dict) else upload.value
        with originals:
            clear_output()
            images = []
            for item in values:
                try:
                    image = prepare_image(bytes(item['content']), item['name'], studio.root/'uploads')
                    images.append(image)
                    print(image['filename'])
                    display(Image(filename=image['path'], width=220))
                except Exception as e:
                    log(str(e))
        update()

    def apply_preset(change):
        cfg = PRESETS[change['new']]
        steps.value, resolution.value, faces.value = cfg['steps'], cfg['octree_resolution'], cfg['max_faces']

    def run(_):
        if studio.lock.locked():
            return log('A batch is already running.')
        if not images:
            return log('Upload at least one image.')
        settings = dict(steps=steps.value, octree_resolution=resolution.value, max_faces=faces.value,
                        seed=seed.value, guidance_scale=guidance.value, textures=textures.value,
                        force_rembg=rembg.value, texture_views=8 if texture_quality.value.startswith('High') else 6,
                        texture_resolution=256 if texture_quality.value.startswith('Low') else 512)
        batch = list(images)
        generate.disabled = True
        def execute():
            try:
                studio.run(batch, settings, progress=lambda v: (log(v), update()))
            except Exception as e:
                log(type(e).__name__ + ': ' + str(e))
            finally:
                generate.disabled = False
                update()
        threading.Thread(target=execute, daemon=True).start()

    def downloads(_):
        with results:
            clear_output()
            path = export_all(studio.root)
            display(download_link(path))
            print('Download before stopping the session. Save a private notebook version to preserve outputs.')

    def inspect(_):
        if not selections.value:
            return
        out = studio.root/'outputs'/selections.value
        with results:
            clear_output()
            print((out/'metadata.json').read_text())
            path = out/stage.value
            if path.exists():
                try:
                    display(preview(path, studio.app_root/'assets/model-viewer.min.js'))
                except Exception as e:
                    print('Viewer:', e)
            else:
                print('Selected output does not exist. Select white_mesh.glb if texturing failed.')
            for p in out.iterdir():
                if p.is_file() and p.suffix in ('.glb', '.obj', '.json'):
                    display(download_link(p))

    upload.observe(received, names='value')
    preset.observe(apply_preset, names='value')
    generate.on_click(run)
    stop.on_click(lambda _: studio.stop())
    refresh.on_click(update)
    all_download.on_click(downloads)
    view.on_click(inspect)
    display(w.VBox([title, dashboard, refresh, w.HTML('<h3>Image upload</h3><p>Use one isolated object per image. '
        'Separate multiple buildings or props before reconstruction.</p>'), upload, originals,
        w.HTML('<h3>Generation settings</h3><p>High quality may exceed T4 memory; reduce quality after an OOM.</p>'),
        w.HBox([preset, texture_quality]), steps, resolution, faces, seed, guidance, textures, rembg,
        w.HBox([generate, stop]), messages, w.HBox([selections, stage, view]), all_download, results]))
    update()
    return studio
