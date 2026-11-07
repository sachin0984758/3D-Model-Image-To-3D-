"""Reproducible small adaptations to the retained MIT workers; no inference replacement."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT/'upstream/build_notebook.py').read_text(encoding='utf-8-sig'))
values = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
          if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)}
for name, key in [('shape_worker.py', 'SHAPE_WORKER_PY'), ('paint_worker.py', 'PAINT_WORKER_PY'),
                  ('stage_weights.py', 'STAGE_PY'), ('prefetch.py', 'PREFETCH_PY')]:
    text = values[key]
    if name == 'shape_worker.py':
        text = text.replace('import json, os, sys, time, traceback',
            'import json, os, sys, time, traceback\nimport trimesh\nfrom studio.storage import update_stage, validate_glb, center_image')
        text = text.replace('        t = time.time()\n        torch.cuda.reset_peak_memory_stats()',
            '        update_stage(out, "shape", status="running", started=time.time())\n'
            '        t = time.time()\n        torch.cuda.reset_peak_memory_stats()', 1)
        start = text.index('        if cfg["skip_existing"]')
        end = text.index('        update_stage(out, "shape", status="running"', start)
        text = text[:start] + '''        existing = [os.path.join(out, filename) for filename in
                    ("white_mesh.obj", "white_mesh.glb", "input.png")]
        if cfg["skip_existing"] and all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in existing):
            try:
                validation = validate_glb(os.path.join(out, "white_mesh.glb"))
                saved_mesh = trimesh.load(os.path.join(out, "white_mesh.obj"), force="mesh", process=False)
                if len(saved_mesh.faces) != validation["faces"]:
                    raise ValueError("Saved OBJ triangles differ from the GLB")
                with Image.open(os.path.join(out, "input.png")) as saved_image:
                    saved_image.verify()
            except Exception as e:
                log(f"{name}: existing geometry is incomplete; regenerating ({e})")
            else:
                log(f"{name}: mesh already exists, skipping")
                update_stage(out, "shape", status="completed", resumed=True, validation=validation)
                enqueue({"name": name, "dir": out})
                continue
''' + text[end:]
        text = text.replace('if image.mode != "RGBA" or cfg["force_rembg"]:',
            'if image.mode != "RGBA" or image.getchannel("A").getextrema() == (255, 255) or cfg["force_rembg"]:')
        text = text.replace('        image.save(', '        image = center_image(image)\n        image.save(', 1)
        text = text.replace('        enqueue({"name": name, "dir": out})\n    except Exception:',
            '        validation = validate_glb(os.path.join(out, "white_mesh.glb"))\n'
            '        update_stage(out, "shape", status="completed", duration=time.time()-t, seed=cfg["seed"], '
            'peak_vram_gb=torch.cuda.max_memory_allocated()/2**30, validation=validation)\n'
            '        enqueue({"name": name, "dir": out})\n    except Exception:')
        text = text.replace('        log(f"{name}: FAILED',
            '        update_stage(out, "shape", status="failed", error=traceback.format_exc())\n        log(f"{name}: FAILED')
    if name == 'paint_worker.py':
        text = text.replace('    job = json.load(open(job_file))',
            '    with open(job_file) as queued:\n        job = json.load(queued)')
        text = text.replace('import glob, json, os, sys, time, traceback',
            'import glob, json, os, sys, time, traceback\nfrom studio.storage import update_stage, validate_glb')
        text = text.replace('log(f"model loaded on {torch.cuda.get_device_name(0)} in {time.time() - t:.0f}s")',
            'log(f"model loaded on {torch.cuda.get_device_name(0)} in {time.time() - t:.0f}s")\n'
            'open(os.path.join(cfg["queue"], "_PAINT_READY"), "w").close()')
        text = text.replace('    os.remove(job_file)',
            '    claimed_file = job_file + ".processing"\n    os.replace(job_file, claimed_file)')
        text = text.replace('    if cfg["skip_existing"] and os.path.exists(glb_path):',
            '    if cfg["skip_existing"] and os.path.exists(glb_path):')
        text = text.replace('        embed_textures(glb_path)  # also repairs GLBs written before this fix',
            '        embed_textures(glb_path)  # also repairs GLBs written before this fix\n'
            '        validate_glb(glb_path, textured=True)\n'
            '        update_stage(out, "paint", status="completed", resumed=True)\n'
            '        os.remove(claimed_file)')
        text = text.replace('    try:\n        t = time.time()',
            '    try:\n        update_stage(out, "paint", status="running", started=time.time())\n        t = time.time()')
        text = text.replace('            glb_path,\n        )\n        embed_textures(glb_path)',
            '            glb_path + ".pending.glb",\n        )\n'
            '        embed_textures(glb_path + ".pending.glb")\n'
            '        validation = validate_glb(glb_path + ".pending.glb", textured=True)\n'
            '        os.replace(glb_path + ".pending.glb", glb_path)\n'
            '        update_stage(out, "paint", status="completed", duration=time.time()-t, '
            'peak_vram_gb=torch.cuda.max_memory_allocated()/2**30, validation=validation)')
        text = text.replace('    except torch.cuda.OutOfMemoryError:\n',
            '    except torch.cuda.OutOfMemoryError:\n'
            '        update_stage(out, "paint", status="out_of_memory", error=traceback.format_exc(), geometry_preserved=True)\n')
        text = text.replace('    except Exception:\n',
            '    except Exception:\n        update_stage(out, "paint", status="failed", error=traceback.format_exc(), geometry_preserved=True)\n')
        text = text.replace('    finally:\n        torch.cuda.empty_cache()',
            '    finally:\n        if os.path.exists(claimed_file):\n            os.remove(claimed_file)\n        torch.cuda.empty_cache()')
        # Resume validation must also be inside the per-job error boundary.
        start = text.index('    if cfg["skip_existing"] and os.path.exists(glb_path):')
        end = text.index('    try:\n', start)
        block = text[start:end]
        text = text[:start] + '    try:\n' + ''.join('    '+line if line.strip() else line for line in block.splitlines(True)) + text[end+len('    try:\n'):]
        start = text.index('        if cfg["skip_existing"]')
        end = text.index('        update_stage(out, "paint", status="running"', start)
        text = text[:start] + '''        if cfg["skip_existing"] and os.path.exists(glb_path):
            try:
                embed_textures(glb_path)
                validation = validate_glb(glb_path, textured=True)
            except Exception as e:
                log(f"{name}: existing texture export is incomplete; repainting ({e})")
            else:
                update_stage(out, "paint", status="completed", resumed=True, validation=validation)
                log(f"{name}: textured mesh already exists, skipping")
                continue
''' + text[end:]
    if name == 'prefetch.py':
        # Critical-file checks in the original downloader did not detect partially missing folders.
        # Hub cache itself avoids repeat downloads of already present files.
        start = text.index('    lambda: hub("tencent/Hunyuan3D-2.1"')
        end = text.index('    lambda: url(', start)
        text = text[:start] + '''    lambda: hub("tencent/Hunyuan3D-2.1", ["hunyuan3d-dit-v2-1/*", "hunyuan3d-paintpbr-v2-1/*", "LICENSE", "NOTICE"]),
    lambda: hub("facebook/dinov2-giant", ["*.json", "*.safetensors", "*.txt", "LICENSE"]),
''' + text[end:]
        text = text.replace('    shutil.rmtree(os.path.join(W, repo_id, ".cache"), ignore_errors=True)',
                            '    pass  # Retain local download metadata to avoid repeat transfers.')
    text = text.replace('cfg = json.load(open(sys.argv[1]))',
        'with open(sys.argv[1]) as config_file:\n    cfg = json.load(config_file)')
    (ROOT/'workers'/name).write_text('# Adapted for Private 3D Studio; original copyright Gaurav Kumar (MIT).\n' + text, encoding='utf-8', newline='\n')
print('Adapted workers; original sources retained in upstream/.')
