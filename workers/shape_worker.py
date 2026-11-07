# Adapted for Private 3D Studio; original copyright Gaurav Kumar (MIT).
"""GPU worker 1: image -> untextured mesh. Each finished mesh is queued for paint_worker.py."""
import json, os, sys, time, traceback
import trimesh
from studio.storage import update_stage, validate_glb, center_image

with open(sys.argv[1]) as config_file:
    cfg = json.load(config_file)
os.chdir(cfg["repo"])
sys.path.insert(0, os.path.join(cfg["repo"], "hy3dshape"))

import torch
from PIL import Image
from hy3dshape import FaceReducer
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.rembg import BackgroundRemover


def log(msg):
    print(f"[shape {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def enqueue(job):
    tmp = os.path.join(cfg["queue"], f".{job['name']}.tmp")
    with open(tmp, "w") as f:
        json.dump(job, f)
    os.replace(tmp, os.path.join(cfg["queue"], f"{job['name']}.json"))  # atomic: paint never sees half a file


import faulthandler
faulthandler.dump_traceback_later(300, repeat=True)  # while loading: log the stack every 5 min, so a stall shows where
t = time.time()
pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    "tencent/Hunyuan3D-2.1", subfolder="hunyuan3d-dit-v2-1", use_safetensors=False, device="cuda")
rembg = BackgroundRemover()
face_reducer = FaceReducer()
faulthandler.cancel_dump_traceback_later()
log(f"model loaded on {torch.cuda.get_device_name(0)} in {time.time() - t:.0f}s")
open(os.path.join(cfg["queue"], "_SHAPE_READY"), "w").close()

for item in cfg["images"]:
    name, out = item["name"], os.path.join(cfg["out"], item["name"])
    os.makedirs(out, exist_ok=True)
    try:
        existing = [os.path.join(out, filename) for filename in
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
        update_stage(out, "shape", status="running", started=time.time())
        t = time.time()
        torch.cuda.reset_peak_memory_stats()
        image = Image.open(item["path"])
        if image.mode != "RGBA" or image.getchannel("A").getextrema() == (255, 255) or cfg["force_rembg"]:
            image = rembg(image.convert("RGB"))
        image = center_image(image)
        image.save(os.path.join(out, "input.png"))

        mesh = pipe(
            image=image,
            num_inference_steps=cfg["steps"],
            guidance_scale=cfg["guidance_scale"],
            generator=torch.Generator().manual_seed(cfg["seed"]),
            octree_resolution=cfg["octree_resolution"],
            num_chunks=cfg["num_chunks"],
            output_type="trimesh",
        )[0]
        mesh = face_reducer(mesh, max_facenum=cfg["max_faces"])
        mesh.export(os.path.join(out, "white_mesh.glb"))
        mesh.export(os.path.join(out, "white_mesh.obj"), include_normals=False)
        log(f"{name}: {len(mesh.faces)} faces in {time.time() - t:.0f}s "
            f"(peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)")
        validation = validate_glb(os.path.join(out, "white_mesh.glb"))
        update_stage(out, "shape", status="completed", duration=time.time()-t, seed=cfg["seed"], peak_vram_gb=torch.cuda.max_memory_allocated()/2**30, validation=validation)
        enqueue({"name": name, "dir": out})
    except Exception:
        update_stage(out, "shape", status="failed", error=traceback.format_exc())
        log(f"{name}: FAILED\n{traceback.format_exc()}")
    finally:
        torch.cuda.empty_cache()

open(os.path.join(cfg["queue"], "_SHAPE_DONE"), "w").close()
log("all shapes done")
