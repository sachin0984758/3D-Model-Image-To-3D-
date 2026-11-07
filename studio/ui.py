"""Private notebook interface for Kaggle and Google Colab."""
import base64
import html
import json
import io
import secrets
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
    doc = ('<!doctype html><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" '
           'content="default-src \'none\'; script-src \'unsafe-inline\' blob:; '
           'style-src \'unsafe-inline\'; img-src data: blob:; connect-src data: blob:; worker-src blob:">'
           '<style>body{margin:0;background:#101722;color:#ddd;font-family:system-ui}'
           'model-viewer{width:100%;height:620px}button{margin:8px;padding:8px;background:#28384c;color:white}</style>'
           '<model-viewer id="v" camera-controls touch-action="none" shadow-intensity="1" '
           'src="data:model/gltf-binary;base64,'+encoded+'"></model-viewer>'
           '<button onclick="v.cameraOrbit=\'0deg 75deg auto\';v.cameraTarget=\'auto auto auto\';v.fieldOfView=\'auto\'">Reset camera</button>'
           '<span>Drag: rotate &middot; wheel: zoom &middot; Shift + drag: pan</span><script type="module">'+js+'</script>')
    return HTML('<iframe sandbox="allow-scripts" style="width:100%;height:680px;border:0" srcdoc="'+html.escape(doc, quote=True)+'"></iframe>')



def download_file(path):
    from IPython.display import HTML
    try:
        from google.colab import files
    except ImportError:
        return download_link(path)
    files.download(str(path))
    return HTML('Download started for <b>' + html.escape(Path(path).name) + '</b>.')


def show(studio):
    import ipywidgets as w
    from IPython.display import display, HTML
    from PIL import Image
    from .exports import export_model

    def field(label, widget):
        widget.tooltip = label
        widget.layout.width = '100%'
        widget.add_class('p3d-field')
        return w.VBox([w.HTML('<label>' + html.escape(label) + '</label>'), widget],
                      layout=w.Layout(flex='0 0 auto'))

    upload = w.FileUpload(accept='.png,.jpg,.jpeg,.webp', multiple=True, description='Upload image', icon='upload')
    reference = w.Image(layout=w.Layout(width='100%', height='210px', object_fit='contain', display='none'))
    empty_reference = w.HTML('<div class="p3d-empty-reference">Upload a reference image<br>or choose an example</div>')
    image_name = w.HTML('PNG, JPEG or WebP · one object.')
    geometry = w.Dropdown(options=list(PRESETS), value='Medium')
    texture_quality = w.Dropdown(options=['Low (6 / 256)', 'Medium (6 / 512)', 'High (8 / 512)'],
                                 value='Medium (6 / 512)')
    steps = w.IntSlider(min=1, max=60, value=30)
    octree = w.IntSlider(min=128, max=320, step=64, value=256)
    chunks = w.BoundedIntText(min=500, max=16000, value=8000)
    faces = w.BoundedIntText(min=1000, max=80000, value=40000)
    seed = w.BoundedIntText(min=0, max=2**32-1, value=1234)
    guidance = w.FloatSlider(min=0, max=15, step=.5, value=5)
    remove_background = w.Checkbox(value=True, description='Remove background', indent=False)
    random_seed = w.Checkbox(value=False, description='Randomize seed', indent=False)
    generate_shape = w.Button(description='Generate Shape', button_style='primary')
    generate_texture = w.Button(description='Generate Textured Shape', button_style='primary')
    stop = w.Button(description='Stop', icon='stop')
    refresh = w.Button(description='Refresh', icon='refresh', tooltip='Refresh status')
    model = w.Dropdown(options=[], description='Model', layout=w.Layout(width='100%'))
    stage = w.Dropdown(options=[('Textured mesh', 'textured_mesh.glb'), ('Geometry', 'white_mesh.glb')],
                       description='View', layout=w.Layout(width='100%'))
    render = w.Button(description='Show model', icon='cube')
    viewer = w.Output(layout=w.Layout(min_height='680px', width='100%'))
    messages = w.Output(layout=w.Layout(max_height='220px', overflow='auto'))
    status = w.HTML()
    export_format = w.Dropdown(options=['GLB', 'OBJ', 'PLY', 'STL'], value='GLB')
    simplify = w.Checkbox(value=False, description='Simplify mesh', indent=False)
    include_texture = w.Checkbox(value=True, description='Include textures', indent=False)
    target_faces = w.BoundedIntText(min=4, max=80000, value=20000)
    transform = w.Button(description='Prepare export')
    download = w.Button(description='Download', icon='download', button_style='primary', disabled=True)
    download_all = w.Button(description='Download all / checkpoint', icon='file-archive-o')
    export_message = w.HTML()
    links = w.Output()
    images = []
    ready_export = None

    def log(value):
        text = json.dumps(value) if isinstance(value, dict) else str(value)
        messages.append_stdout(text + '\n')

    def update(_=None):
        jobs = studio.jobs()
        selected = model.value
        options = [(j.get('input', {}).get('filename', 'Generated model') + ' · ' + j['name'], j['name'])
                   for j in jobs]
        if tuple(model.options) != tuple(options):
            model.options = options
            if selected in [value for _, value in options]:
                model.value = selected
        devices = gpu_info()
        completed = sum(j['paint'].get('status') == 'completed' for j in jobs)
        hardware = 'Two GPU workers' if len(devices) > 1 else 'One GPU · geometry then painting'
        status.value = ('<b>' + html.escape(studio.state) + '</b><br>'
                        + html.escape(hardware) + ' · ' + str(len(images)) + ' image(s) · '
                        + str(completed) + ' textured model(s)')

    def show_reference(prepared):
        nonlocal images
        images = prepared
        reference.value = Path(prepared[0]['path']).read_bytes() if prepared else b''
        reference.layout.display = '' if prepared else 'none'
        empty_reference.layout.display = 'none' if prepared else ''
        image_name.value = '<b>' + html.escape(', '.join(p['filename'] for p in prepared)) + '</b>'
        update()

    def received(change):
        values = upload.value.values() if isinstance(upload.value, dict) else upload.value
        prepared = []
        for item in values:
            try:
                prepared.append(prepare_image(bytes(item['content']), item['name'], studio.root/'uploads'))
            except Exception as error:
                log(error)
        show_reference(prepared)

    def view(_=None):
        viewer.outputs = ()
        if not model.value:
            viewer.append_display_data(HTML('<div class="p3d-empty-viewer"><div>'
                '<h3>Your 3D model appears here</h3>'
                '<p>Upload a reference image and choose a generation button.</p>'
                '</div></div>'))
            return
        path = studio.root/'outputs'/model.value/stage.value
        if not path.is_file():
            viewer.append_stdout('This stage is pending. Choose Geometry to view a completed shape.\n')
            return
        try:
            viewer.append_display_data(preview(path, studio.app_root/'assets/model-viewer.min.js'))
        except Exception as error:
            viewer.append_stdout(str(error) + '\n')

    def apply_preset(change):
        preset = PRESETS[change['new']]
        steps.value = preset['steps']
        octree.value = preset['octree_resolution']
        faces.value = preset['max_faces']

    def run(textured):
        if studio.lock.locked() or generate_shape.disabled:
            log('A batch is already running.')
            return
        if not images:
            log('Upload an image or choose an example.')
            return
        if random_seed.value:
            seed.value = secrets.randbits(32)
        settings = dict(steps=steps.value, octree_resolution=octree.value, max_faces=faces.value,
                        seed=seed.value, guidance_scale=guidance.value, num_chunks=chunks.value,
                        textures=textured, remove_background=remove_background.value,
                        texture_views=8 if texture_quality.value.startswith('High') else 6,
                        texture_resolution=256 if texture_quality.value.startswith('Low') else 512)
        batch = list(images)
        generate_shape.disabled = generate_texture.disabled = True

        def execute():
            try:
                records = studio.run(batch, settings, progress=lambda value: (log(value), update()))
                update()
                expected = dict(DEFAULTS, **settings)
                selected = next((job for job in reversed(records)
                                 if job['input']['sha256'] == batch[-1]['sha256']
                                 and dict(DEFAULTS, **job['settings']) == expected), None)
                if selected:
                    model.value = selected['name']
                    stage.value = ('textured_mesh.glb' if selected['paint'].get('status') == 'completed'
                                   else 'white_mesh.glb')
                    include_texture.value = stage.value == 'textured_mesh.glb'
                    view()
            except Exception as error:
                log(type(error).__name__ + ': ' + str(error))
            finally:
                generate_shape.disabled = generate_texture.disabled = False
                update()

        threading.Thread(target=execute, daemon=True).start()

    def reset_export(change=None):
        nonlocal ready_export
        ready_export = None
        download.disabled = True
        geometry_only = export_format.value in ('PLY', 'STL')
        if geometry_only:
            include_texture.value = False
        include_texture.disabled = geometry_only
        export_message.value = ('PLY and STL contain geometry.' if geometry_only
                                else 'OBJ downloads include its material and texture files in a ZIP.')

    def prepare_export(_):
        nonlocal ready_export
        if not model.value:
            export_message.value = 'Generate or select a model first.'
            return
        transform.disabled = True
        try:
            ready_export = export_model(studio.root/'outputs'/model.value, studio.root/'exports',
                                        file_format=export_format.value.lower(),
                                        include_texture=include_texture.value,
                                        max_faces=target_faces.value if simplify.value else None)
            export_message.value = 'Ready · ' + html.escape(ready_export.name)
            download.disabled = False
        except Exception as error:
            ready_export = None
            download.disabled = True
            export_message.value = html.escape(str(error))
            log(error)
        finally:
            transform.disabled = False

    def send_download(path):
        links.outputs = ()
        try:
            links.append_display_data(download_file(path))
        except Exception as error:
            log(error)

    def download_prepared(_):
        if ready_export:
            send_download(ready_export)

    def checkpoint(_):
        try:
            send_download(export_all(studio.root))
        except Exception as error:
            log(error)

    sample_buttons = []
    sample_css = []
    sample_dir = studio.repo/'assets/example_images'
    samples = sorted(p for p in sample_dir.glob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'))[:18]
    for index, path in enumerate(samples):
        try:
            with Image.open(path) as image:
                thumbnail = image.convert('RGB')
                thumbnail.thumbnail((150, 150))
                buffer = io.BytesIO()
                thumbnail.save(buffer, format='JPEG')
            encoded = base64.b64encode(buffer.getvalue()).decode()
            button = w.Button(tooltip=path.name, layout=w.Layout(width='100%', height='90px'))
            class_name = 'p3d-example-' + str(index)
            button.add_class(class_name)
            sample_css.append('.' + class_name + '{background-image:url(data:image/jpeg;base64,'
                              + encoded + ');background-position:center;background-repeat:no-repeat;background-size:contain;}')
            def select_example(_, selected=path):
                try:
                    show_reference([prepare_image(selected.read_bytes(), selected.name, studio.root/'uploads')])
                except Exception as error:
                    log(error)
            button.on_click(select_example)
            sample_buttons.append(button)
        except Exception as error:
            log('Example ' + path.name + ': ' + str(error))

    css = '''
    <style>
    .p3d-app{background:#0b1018;color:#e8eef7;padding:18px;border:1px solid #293446;border-radius:14px;font-family:Inter,Arial,sans-serif;
      --jp-layout-color0:#0b1018;--jp-layout-color1:#101722;--jp-layout-color2:#182333;--jp-layout-color3:#263347;
      --jp-content-font-color0:#edf3ff;--jp-content-font-color1:#e8eef7;--jp-content-font-color2:#b9c7da;
      --jp-ui-font-color0:#edf3ff;--jp-ui-font-color1:#e8eef7;--jp-ui-font-color2:#b9c7da;
      --jp-border-color1:#354357;--jp-border-color2:#303b4d;--jp-widgets-color:#e8eef7;--jp-widgets-label-color:#dce5f3}
    .p3d-app .widget-html-content,.p3d-app .output_subarea,.p3d-app pre{color:#e8eef7!important}
    .p3d-app .widget-label,.p3d-app label{color:#dce5f3!important}
    .p3d-app h2,.p3d-app h3{color:#edf3ff;margin:4px 0 14px}
    .p3d-app .widget-button{background-color:#202b3b;color:#e8eef7;border:1px solid #354357;border-radius:9px;min-height:36px}
    .p3d-app .widget-upload{background:#202b3b;color:#e8eef7;border:1px solid #354357;border-radius:9px}
    .p3d-app .mod-primary{background:#2455df!important;color:white!important}
    .p3d-app input,.p3d-app select{background:#263347!important;color:#eef3fb!important;border:1px solid #43516a!important;border-radius:6px}
    .p3d-app .p3d-field{margin-bottom:10px}
    .p3d-workspace{gap:16px;align-items:stretch}
    .p3d-panel{background:#101722;border:1px solid #303b4d;border-radius:12px;padding:14px}
    .p3d-app .widget-tab>.p-TabBar .p-TabBar-tab,.p3d-app .widget-tab>.lm-TabBar .lm-TabBar-tab{background:#182333;color:#dfe8f6}
    .p3d-app .widget-tab-contents{background:#101722;border:1px solid #303b4d;padding:10px}
    .p3d-app .p3d-reference{background:#182333;border:1px solid #354357;border-radius:10px;padding:10px;margin-bottom:10px}
    .p3d-empty-reference{height:210px;display:flex;align-items:center;justify-content:center;text-align:center;color:#b9c7da;border:1px dashed #43516a;border-radius:8px;line-height:24px}
    .p3d-viewer{background:#0b1018;border:1px solid #354357;border-radius:10px}
    .p3d-empty-viewer{height:660px;display:flex;align-items:center;justify-content:center;text-align:center;color:#b9c7da}
    .p3d-app .widget-readout{color:#dfe8f6}
    .p3d-generate{gap:8px;margin:4px 0 14px}
    .p3d-generate .widget-button{white-space:normal;line-height:17px;font-size:12px}
    .p3d-app .widget-checkbox{flex:0 0 auto}
    </style>
    '''
    advanced = w.VBox([remove_background, random_seed, field('Geometry preset', geometry),
                      field('Seed', seed), field('Inference steps', steps), field('Octree resolution', octree),
                      field('Guidance scale', guidance), field('Number of chunks', chunks),
                      field('Maximum faces', faces), field('Texture quality', texture_quality)],
                     layout=w.Layout(max_height='320px', overflow='auto'))
    exports = w.VBox([field('Format', export_format), simplify, include_texture,
                      field('Target face number', target_faces),
                      w.HBox([transform, download]), export_message, links])
    tabs = w.Tab(children=[advanced, exports])
    tabs.set_title(0, 'Advanced options')
    tabs.set_title(1, 'Export')
    reference_box = w.VBox([reference, empty_reference, image_name, upload])
    reference_box.add_class('p3d-reference')
    for button in (generate_shape, generate_texture):
        button.layout = w.Layout(width='50%', height='50px')
    generate_buttons = w.HBox([generate_shape, generate_texture])
    generate_buttons.add_class('p3d-generate')
    left = w.VBox([w.HTML('<h3>Reference image</h3>'), reference_box,
                   generate_buttons, tabs, download_all],
                  layout=w.Layout(width='300px', min_width='270px', flex='0 0 300px'))
    centre = w.VBox([w.HBox([model, stage]), render, viewer],
                    layout=w.Layout(min_width='360px', flex='1 1 420px'))
    viewer.add_class('p3d-viewer')
    gallery = w.GridBox(sample_buttons, layout=w.Layout(grid_template_columns='repeat(3, 1fr)', grid_gap='8px'))
    right = w.VBox([w.HTML('<h3>Examples</h3>'), gallery],
                   layout=w.Layout(width='250px', flex='0 0 250px'))
    for panel in (left, centre, right):
        panel.add_class('p3d-panel')
    download_all.layout.width = '100%'
    workspace = w.HBox([left, centre, right], layout=w.Layout(width='100%', flex_flow='row wrap'))
    workspace.add_class('p3d-workspace')
    logs = w.Accordion(children=[messages], selected_index=None)
    logs.set_title(0, 'Logs')
    app = w.VBox([w.HTML(css + '<style>' + ''.join(sample_css) + '</style><h2>Private 3D Studio</h2>'),
                  w.HBox([status, refresh, stop]), workspace, logs], layout=w.Layout(width='100%'))
    app.add_class('p3d-app')

    upload.observe(received, names='value')
    geometry.observe(apply_preset, names='value')
    model.observe(lambda change: (reset_export(), view()), names='value')
    stage.observe(view, names='value')
    for control in (export_format, simplify, include_texture, target_faces):
        control.observe(reset_export, names='value')
    generate_shape.on_click(lambda _: run(False))
    generate_texture.on_click(lambda _: run(True))
    transform.on_click(prepare_export)
    download.on_click(download_prepared)
    download_all.on_click(checkpoint)
    render.on_click(view)
    stop.on_click(lambda _: studio.stop())
    refresh.on_click(update)
    display(app)
    update()
    view()
    return studio
