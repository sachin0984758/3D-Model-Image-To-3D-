# Adapted for Private 3D Studio; original copyright Gaurav Kumar (MIT).
"""GPU worker 2: untextured mesh + reference image -> PBR-textured GLB, fed by shape_worker.py."""
import glob, json, os, sys, time, traceback
from studio.storage import update_stage, validate_glb

with open(sys.argv[1]) as config_file:
    cfg = json.load(config_file)
os.chdir(cfg["repo"])
sys.path[:0] = [cfg["repo"], os.path.join(cfg["repo"], "hy3dpaint")]

import base64
import pygltflib
import pymeshlab
import torch
import textureGenPipeline
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from convert_utils import create_glb_with_pbr_materials


def log(msg):
    print(f"[paint {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def remesh_to_max_faces(mesh_path, remesh_path):
    """Replaces the repo's remesh_mesh, which cuts every mesh to 40k faces (with open3d, which isn't installed).

    That cap silently threw away the detail a higher max_faces setting asked for; this keeps up to max_faces.
    """
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path)
    if ms.current_mesh().face_number() > cfg["max_faces"]:
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=cfg["max_faces"])
    ms.save_current_mesh(remesh_path, save_textures=False)


textureGenPipeline.remesh_mesh = remesh_to_max_faces  # the pipeline looks it up as a module global at call time


def embed_textures(path):
    """Moves textures into the GLB binary chunk with the right mimeType.

    create_glb_with_pbr_materials stores them as JSON data URIs all labelled image/png (the base colour is a JPEG).
    Web viewers accept that, but it is invalid GLB: Windows 3D Viewer and stricter importers show no texture.
    """
    gltf = pygltflib.GLTF2().load(path)
    blob = bytearray(gltf.binary_blob() or b"")
    changed = False
    for image in gltf.images:
        if not (image.uri or "").startswith("data:"):
            continue
        data = base64.b64decode(image.uri.split(",", 1)[1])
        blob += b"\x00" * (-len(blob) % 4)  # bufferViews must be 4-byte aligned
        gltf.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=len(blob), byteLength=len(data)))
        blob += data
        mime = "image/jpeg" if data[:3] == b"\xff\xd8\xff" else "image/png"
        image.uri, image.bufferView, image.mimeType = None, len(gltf.bufferViews) - 1, mime
        changed = True
    if not changed:
        return
    for material in gltf.materials:
        if material.alphaMode in (None, "OPAQUE"):
            material.alphaCutoff = None  # only meaningful with alphaMode MASK
    blob += b"\x00" * (-len(blob) % 4)
    gltf.buffers[0].byteLength = len(blob)
    gltf.set_binary_blob(bytes(blob))
    gltf.save_binary(path)


def next_job():
    jobs = sorted(glob.glob(os.path.join(cfg["queue"], "*.json")), key=os.path.getmtime)
    return jobs[0] if jobs else None


# The texture model always calls snapshot_download(); serve it from the weights folder instead.
import huggingface_hub
_download = huggingface_hub.snapshot_download
def _local_first(repo_id, *args, **kwargs):
    local = os.path.join(cfg["weights"], repo_id)
    return local if os.path.isdir(local) else _download(repo_id, *args, **kwargs)
huggingface_hub.snapshot_download = _local_first  # multiview_utils looks it up at call time

import faulthandler
faulthandler.dump_traceback_later(300, repeat=True)  # while loading: log the stack every 5 min, so a stall shows where
t = time.time()
conf = Hunyuan3DPaintConfig(cfg["texture_views"], cfg["texture_resolution"])
conf.dino_ckpt_path = os.path.join(cfg["weights"], "facebook/dinov2-giant")
conf.realesrgan_ckpt_path = os.path.join(cfg["weights"], "RealESRGAN_x4plus.pth")
conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
pipe = Hunyuan3DPaintPipeline(conf)
faulthandler.cancel_dump_traceback_later()
log(f"model loaded on {torch.cuda.get_device_name(0)} in {time.time() - t:.0f}s")
open(os.path.join(cfg["queue"], "_PAINT_READY"), "w").close()

while True:
    job_file = next_job()
    if job_file is None:
        # Shape writes _SHAPE_DONE after its last enqueue, so re-check the queue once it appears.
        if os.path.exists(os.path.join(cfg["queue"], "_SHAPE_DONE")) and next_job() is None:
            break
        time.sleep(2)
        continue

    with open(job_file) as queued:
        job = json.load(queued)
    claimed_file = job_file + ".processing"
    os.replace(job_file, claimed_file)
    name, out = job["name"], job["dir"]
    glb_path = os.path.join(out, "textured_mesh.glb")
    try:
        if cfg["skip_existing"] and os.path.exists(glb_path):
            try:
                embed_textures(glb_path)
                validation = validate_glb(glb_path, textured=True)
            except Exception as e:
                log(f"{name}: existing texture export is incomplete; repainting ({e})")
            else:
                update_stage(out, "paint", status="completed", resumed=True, validation=validation)
                log(f"{name}: textured mesh already exists, skipping")
                continue
        update_stage(out, "paint", status="running", started=time.time())
        t = time.time()
        torch.cuda.reset_peak_memory_stats()
        obj_path = pipe(
            mesh_path=os.path.join(out, "white_mesh.obj"),
            image_path=os.path.join(out, "input.png"),
            output_mesh_path=os.path.join(out, "textured_mesh.obj"),
            save_glb=False,
        )
        base = obj_path[:-len(".obj")]
        create_glb_with_pbr_materials(
            obj_path,
            {"albedo": base + ".jpg", "metallic": base + "_metallic.jpg", "roughness": base + "_roughness.jpg"},
            glb_path + ".pending.glb",
        )
        embed_textures(glb_path + ".pending.glb")
        validation = validate_glb(glb_path + ".pending.glb", textured=True)
        os.replace(glb_path + ".pending.glb", glb_path)
        update_stage(out, "paint", status="completed", duration=time.time()-t, peak_vram_gb=torch.cuda.max_memory_allocated()/2**30, validation=validation)
        log(f"{name}: textured in {time.time() - t:.0f}s "
            f"(peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)")
    except torch.cuda.OutOfMemoryError:
        update_stage(out, "paint", status="out_of_memory", error=traceback.format_exc(), geometry_preserved=True)
        log(f"{name}: OUT OF GPU MEMORY - lower texture_views / texture_resolution in the settings cell")
    except Exception:
        update_stage(out, "paint", status="failed", error=traceback.format_exc(), geometry_preserved=True)
        log(f"{name}: FAILED\n{traceback.format_exc()}")
    finally:
        if os.path.exists(claimed_file):
            os.remove(claimed_file)
        torch.cuda.empty_cache()

log("all textures done")
