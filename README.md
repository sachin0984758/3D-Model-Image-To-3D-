# PRIVATE 3D STUDIO

A private, notebook-based image-to-3D application using the existing Hunyuan3D-2.1 Kaggle worker implementation. Geometry runs in a separate process on GPU 0; PBR paint runs on GPU 1, started only after a geometry job is queued. One-GPU sessions use geometry only. No public server or tunnel is created.

## Notebooks

| Notebook | Description | Kaggle | Colab | Lightning AI | Hugging Face | Video |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| [**PRIVATE 3D STUDIO**](private_3d_studio.ipynb) | Hunyuan3D-2.1 image-to-3D studio with PBR textures, image uploads, batch queues, preview controls, GLB/OBJ files and ZIP export. Uses **T4 x2** for geometry and painting. | [![Import on Kaggle](https://img.shields.io/badge/Kaggle-Import-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/new) | — | — | [![Model weights](https://img.shields.io/badge/Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/tencent/Hunyuan3D-2.1) | — |
| [**Weights Builder**](private_3d_weights_builder.ipynb) | Downloads the checkpoints and writes a SHA-256 integrity manifest. Run on **CPU**, then save the output as a private Kaggle dataset for future sessions. | [![Import on Kaggle](https://img.shields.io/badge/Kaggle-Import-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/new) | — | — | [![Model weights](https://img.shields.io/badge/Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/tencent/Hunyuan3D-2.1) | — |
| [**GPU Validation**](private_3d_studio_validation.ipynb) | Runs CPU checks and a two-image geometry/PBR batch on **T4 x2**. Records generation times, peak VRAM, mesh checks and texture validation, then exports the results. | [![Import on Kaggle](https://img.shields.io/badge/Kaggle-Import-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/new) | — | — | [![Model weights](https://img.shields.io/badge/Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/tencent/Hunyuan3D-2.1) | — |

The Kaggle badges open a new notebook. Download the chosen `.ipynb` from its notebook link, then select **File > Import Notebook** in Kaggle. The Hugging Face badges open the official model repository.

## Quick start

1. Run **Weights Builder** with Internet enabled and no accelerator. Save the completed output as a **Private** dataset.
2. Import **PRIVATE 3D STUDIO**, choose **GPU T4 x2**, enable Internet for setup and attach the weights dataset through **Add Input**.
3. Run the setup cells, upload an image and select **Generate 3D**. Start with Medium geometry and six texture views at 512.
4. Open **Preview / downloads** or run the final **Save your work** cell, then download `generated_models.zip`.

Use **GPU Validation** to run the two-image test. It automatically downloads missing checkpoints and starts real generation when its cells run.

## Validation

The saved validation run on two Kaggle T4s generated two textured GLBs. Both had 40,000 faces and passed the embedded PBR texture checks. The current notebooks include relative download links, retries for incomplete exports and a final download cell. Twelve local tests passed. Live preview, browser download and reuse of a persistent weights dataset still need verification. The saved GPU run predates the latest fixes. See [the per-image results](TEST_RESULTS.md).

## Files

| File | Purpose |
| --- | --- |
| `private_3d_studio.ipynb` | Main application, with embedded source |
| `private_3d_weights_builder.ipynb` | One-time downloads and checksum manifest |
| `private_3d_studio_validation.ipynb` | Explicit validation run: downloads permitted, CPU tests, two real image jobs; consumes GPU quota |
| `studio/` | Image validation, weights, worker controller, notebook UI, preview and exports |
| `workers/` | Adapted shape/paint workers, upstream installer, downloader and fast staging |
| `tests/` | CPU-only validation; synthetic fixtures are not inference results |
| `tools/` | Reproducible worker adaptation and notebook generation |
| `upstream/` | Retained original source notebooks and generator |
| `LICENSE`, `NOTICE`, `TENCENT_LICENSE.txt` | Attribution and applicable license terms |

## 1. Build your private weights dataset once

1. Open Kaggle and create a notebook. Use **File > Import Notebook** (some editor versions call this **Open/Upload Notebook**) and choose `private_3d_weights_builder.ipynb`.
2. Check notebook visibility is **Private**. Select **no accelerator** and turn Internet on.
3. Run all cells. This downloads the official shape, paint, DINOv2, RealESRGAN and U2Net files. It prints actual sizes and writes `weights_manifest.json` with SHA-256 hashes. The manifest detects later corruption; it is not a vendor signature.
4. Choose **Save Version > Save & Run All**. Wait for a successful saved run.
5. From the saved version's output, create a dataset containing `private-3d-weights`. Set visibility to **Private** before saving. Name it `private-hunyuan3d-weights` or any personal name.
6. Keep the builder output below Kaggle's displayed output limit. Upstream estimates about 17.8 GB of model files; the actual output size is authoritative.

Expected root inside the dataset:

```text
tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/
tencent/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1/
facebook/dinov2-giant/
RealESRGAN_x4plus.pth
u2net/u2net.onnx
weights_manifest.json
```

The studio detects nested Kaggle mount roots. If multiple model datasets are attached, specify the intended root in `DATASET` in the weights cell. The notebook cannot enforce account-level sharing settings; check notebook, weights and input datasets yourself.

## 2. Start the studio

1. Import `private_3d_studio.ipynb`. Check **Private** visibility.
2. Select **Settings > Accelerator > GPU T4 x2**. Phone verification and remaining quota are required. Turn Internet on for installation and viewer-code download.
3. Attach your private weights dataset with **Add Input > Your Datasets**.
4. Run the source, environment, installation, weights and studio cells in order. The installer keeps Kaggle's PyTorch, applies the existing spandrel/optional-Blender fixes, and compiles the C++ inpainter and CUDA rasterizer for the devices in this session. Any failed command stops setup.
5. `ALLOW_DOWNLOAD` defaults to `False`. If no model dataset is attached, stop and build one, or explicitly set this option to `True` to download into temporary storage. Downloaded temporary weights disappear with the session.
6. Upload PNG, JPEG or WebP images using the notebook widget. Images stay within your Kaggle session. Use one isolated object per reference. Automatic validation does not identify/count semantic objects; the UI gives an explicit separation warning.
7. Start with Medium geometry, six views, 512 texture diffusion resolution and one image. Select **Generate 3D**. Logs and GPU readings update periodically. Use **Refresh dashboard** if the widget display lags.
8. Select a completed model and preview stage, then choose **Preview / downloads**. Rotate by dragging, zoom with the wheel, pan with Shift-drag, and use Reset camera. The GLB is embedded locally; the preview's content policy blocks external connections.
9. Use **Download All / checkpoint**, or run the final **Save your work** cell and click its ZIP link. Links are relative to the notebook working directory. In Kaggle Output, the ZIP is at `private_3d_studio/exports/generated_models.zip`. Save a private version or download the assets before stopping the session.

The 512 texture setting controls multiview diffusion input resolution. The retained upstream pipeline has its own 2048 render/4096 bake settings; the UI does not claim that 512 is the final baked atlas size.

## 3. Outputs and resume

```text
/kaggle/working/private_3d_studio/
  app/                       embedded application source
  uploads/                   validated upload references
  outputs/model_001/
    input.png
    white_mesh.obj
    white_mesh.glb
    textured_mesh.obj         only when paint reaches that stage
    textured_mesh.glb         only after embedded-PBR validation succeeds
    job.json
    shape_status.json
    paint_status.json
    metadata.json
  logs/shape_worker.log
  logs/paint_worker.log
  runs/                      unique queue per batch
  exports/generated_models.zip
```

Metadata includes the original input filename and SHA-256, checkpoint and subfolders, settings and seed, per-stage state, errors, duration/peak VRAM on completed stages and output filenames. PBR failure never converts geometry into a reported textured success. Geometry is retained for download.

Rerunning the same inputs and settings resumes matching jobs and skips validated completed files. Changing settings allocates a new model folder. Successful outputs are not deliberately overwritten. The shape and paint stages own separate status files to avoid concurrent metadata writes.

Kaggle session storage is temporary. Download the ZIP or save a **private** notebook version to retain output. In a new session, attach the saved output/private dataset and copy its `outputs` folder to the location above, then upload the same references. Source paths from the previous session may be stale; the new upload replaces the queue reference. Failed paint jobs can be retried using the same settings without rerunning valid geometry. Retain `job.json` and stage metadata with the meshes.

**Stop workers** terminates only processes launched by this Studio instance. A hard notebook/session restart terminates GPU processes; recover intermediate outputs from a saved version or downloaded archive. No notebook can continue after Kaggle terminates its resources.

## 4. Tests

CPU tests, from the project folder:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m unittest discover -s tests -v
python tools/verify_notebooks.py
```

`Pillow`, `numpy`, `trimesh` and `IPython` are needed for the complete CPU suite. `nbformat` optionally validates notebook schemas. `workers/install.sh` is for the Linux Kaggle session, not Windows or your local 4 GB GPU.

On Kaggle, after running the embedded-source and setup cells:

```python
import unittest
suite = unittest.defaultTestLoader.discover(str(APP/'tests'))
result = unittest.TextTestRunner(verbosity=2).run(suite)
assert result.wasSuccessful() and not result.skipped
```

The optional GPU smoke-test cell uses two official example images and the same controller as the UI. Set `RUN_GPU_TEST=True` manually, execute it, inspect `gpu_test_results.json`, and download the output. This consumes GPU quota. A geometry pass requires an actual mesh that reloads with triangles. A PBR pass requires successful paint plus UVs, embedded albedo and metallic/roughness assignments. Visually inspect the resulting GLB in the notebook or Blender as a separate check.

The validation notebook enables downloads and GPU tests by default. The main application keeps these options disabled by default. The earlier eight-test CPU suite passed on Kaggle. The current twelve-test suite passed locally, including incomplete-export retries, relative download links, concurrent ZIP exports and staging-cache invalidation. Its Kaggle rerun is pending. See `TEST_RESULTS.md`.

To inspect an already executed validation run without consuming GPU time, download its notebook with cell outputs using Kaggle's **File > Download Notebook**, then run:

```powershell
python tools/inspect_kaggle_run.py downloaded_validation.ipynb --output saved_run_report.json
```

This reads actual saved outputs only. It reports worker-recorded geometry/PBR checks separately from observation-only log messages. It cannot certify missing output or replace independent inspection of the generated GLBs and live interface. Do not rerun generation merely to collect existing results.

For an unattended batch commit, replace widget uploads with explicit `prepare_image` calls over your selected private input folder and invoke `studio.run(images, settings)` in a cell. Widget callbacks cannot run in Kaggle's noninteractive commit mode. A saved commit must include the actual generation call to regenerate assets; saving UI cells alone only saves the app setup.

## 5. Troubleshooting and limits

- **No GPUs / exhausted quota:** stop and choose a supported accelerator when Kaggle permits it. Do not run this model on the local 4 GB GPU.
- **One GPU:** geometry-only mode. The two T4 memories are never combined into a shared 32 GB allocation.
- **OOM:** lower octree/face count or disable textures. For paint try six views and 256 resolution. No automatic quality-changing retry is performed, and High is not guaranteed to fit. The upstream precision and inference behavior are retained.
- **Weights missing/corrupt:** rerun the builder or attach the correct private dataset. A manifest mismatch stops loading. Legacy datasets without a manifest receive only basic integrity checks.
- **Installer conflict messages:** Kaggle includes unrelated packages. The live test displayed conflicts for Gradio/litellm/dopamine, while required extension/import checks still passed. This application uses ipywidgets rather than Gradio. A failed required import is a setup failure and must be fixed, not ignored.
- **Preview unavailable:** download the GLB. Widget/iframe rendering may differ between Kaggle editor versions; this requires browser verification in your session. The preview does not send your model to a public viewer.
- **FBX:** not implemented. Import GLB into Blender and export separately if required.
- **Game readiness:** no claim of production topology, scale, collision meshes, UV quality or game optimization is made. Inspect and clean generated assets in a 3D tool.

## Licenses and provenance

The original notebook code is MIT licensed by Gaurav Kumar. The original notices and notebooks remain included. Official Tencent runtime code is checked against revision `82920d643c0dc2f7bfd7255f45f62d386edfe60c`. Read `NOTICE`, `LICENSE`, `TENCENT_LICENSE.txt`, and component licenses before use or distribution. PRIVATE 3D STUDIO is personal application branding; it does not claim exclusive authorship of reused source or affiliation with Tencent.
