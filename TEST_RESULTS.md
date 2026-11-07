# PRIVATE 3D STUDIO validation results

## Two-image Kaggle run, inspected 2026-11-07

The downloaded `private-3d-studio-validation.ipynb` contains eight executed code cells, including the GPU test and the final export cell. All eight CPU tests passed. The notebook contains no saved cell exceptions.

| Input | Export folder | Shape | Paint | Faces in textured GLB | Embedded images | Textured GLB bytes | Peak paint VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `004.png` | `model_001` | 123.8 s | 157.4 s | 40,000 | 2 | 4,890,088 | 13.4 GiB |
| `052.png` | `model_002` | 141.5 s | 164.3 s | 40,000 | 2 | 5,660,936 | 12.3 GiB |

Both shape and paint workers recorded completion for each image. The Kaggle validator reloaded each GLB, counted triangles and checked UV coordinates, embedded images, and base-color and metallic/roughness texture assignments. Both paint records contain `pbr_verified: true`. The run used two Tesla T4s, seed 1234, 30 shape steps, six texture views and a 512 diffusion resolution.

Each model's file list includes `white_mesh.glb`, `textured_mesh.glb`, OBJ geometry and separate color, metallic and roughness images. The final cell linked `/kaggle/working/private_3d_studio/exports/generated_models.zip`. The notebook contains run records and a link. Inspecting the models visually requires the actual GLBs.

The extracted [saved run report](tests/evidence/studio_saved_run_report.json) includes the source notebook's SHA-256. The current project has later weight-staging, download, installation, retry and startup changes. The run used temporary weights and did not exercise those changes or establish that a private weights dataset was saved.

## Changes checked locally on 2026-11-07

- Downloads now use paths relative to the notebook working directory. Both the main and validation notebooks include a final download cell.
- The installer fetches the tested runtime commit directly. Its earlier shallow clone followed the moving default branch before checking the pinned revision.
- Incomplete GLB exports trigger regeneration or repainting on retry. Geometry reuse also checks the OBJ triangle count and saved input image. Valid completed stages retain their validation records and original timing.
- Concurrent ZIP requests are serialized and use unique temporary files. Transient JSON files are excluded from metadata listings and archives.
- The main notebook now handles setup, downloads, generation and exports in one workflow. Missing weights download automatically into `/kaggle/working/private-3d-weights`, which can be included in saved notebook outputs. Existing cache files and attached weight folders are reused. The weights builder is an optional CPU-only helper.
- Weight staging checks free space on the cache destination's filesystem. Reusing a weights folder in place preserves its files.

All **14 CPU tests passed with no skips**. Synthetic fixtures exercised both worker job loops without loading GPU models. The main notebook's actual weights cell also passed first-download, repeat-run cache reuse and attached-output staging checks using tiny synthetic checkpoint files. Local widget callbacks passed upload, quality-preset and ZIP export checks. The latest notebooks passed schema validation, cell compilation and embedded-source comparison.

The latest fixes still require a Kaggle run. This Windows sandbox blocks Git Bash from starting, so the changed installer could not be executed locally. Browser rendering, model appearance and persistent-dataset reuse remain open.

## Earlier tests and scope

This report distinguishes the upstream runtime test from the new studio wrapper.
No synthetic fixture is a Hunyuan inference result.

| Test | Result | Evidence / limit |
| --- | --- | --- |
| Local image validation and proportion-preserving centering | PASS | Pillow fixtures; invalid/transparent inputs rejected |
| Atomic stage records and checkpoint ZIP | PASS | Separate shape/paint records and archive contents checked |
| No-GPU handling and lock cleanup | PASS | GPU discovery patched to no devices; no model launched |
| Missing/corrupt weights detection | PASS | Synthetic SHA-256 fixtures; same-size corruption detected |
| Invalid GLB rejection | PASS | Truncated and malformed containers rejected |
| Retained worker call contract | PASS | Actual model-loading, paint and PBR export calls remain in source |
| Valid geometry/PBR GLB reload on local host | PASS | Synthetic geometry and embedded-PBR fixtures reloaded with trimesh 4.4.7 |
| Python module compilation | PASS | All new Python source and adapted workers compiled |
| Notebook cell syntax and embedded source consistency | PASS | Seventeen Python cells across three notebooks parsed; embedded payload compared to delivered source |
| Notebook nbformat schema validation | PASS | All three notebooks validated with nbformat 5.10.4 |
| Kaggle account/GPU preflight | PASS | Signed-in, phone-verified session; two Tesla T4 devices, each 15360 MiB |
| Kaggle environment | OBSERVED | Python 3.13.16; torch 2.11.0+cu130; CUDA 13.0; NVIDIA driver 580.178.04; nvcc available |
| Upstream installer on Kaggle | PASS | Exit 0; C++ inpainter and CUDA rasterizer built for capability 7.5; required imports passed; `Install OK (numpy 2.1.3)` |
| Official checkpoint download | PASS | Downloader exit 0; all five critical model groups present; 17.7 GiB reported under `/tmp/private-3d-weights` |
| Real upstream geometry inference | GENERATED | Official `assets/example_images/004.png`; 30 steps, guidance 5, octree 256, chunks 8000, seed 1234. Worker exit 0; 40000 faces in 124 seconds; peak VRAM 7.6 GiB. `white_mesh.glb` 720864 bytes and `white_mesh.obj` 1461858 bytes observed. Original session ended before independent mesh reload/visual checks. |
| Real upstream paint model loading | PASS | Tesla T4 on GPU 1; model loaded in 58 seconds. |
| Real upstream PBR paint completion | NOT VERIFIED | Six views / 512; paint began but the session ended without a saved completed texture result. Model loading is not a PBR pass. |
| Delivered installer on fresh Kaggle session | PASS | Exit success in 209.27 seconds after Internet was enabled; required imports and viewer-code setup completed. Initial attempt with Internet off failed explicitly with Git host-resolution error, then succeeded on retry. |
| Full CPU suite on Kaggle | PASS | All 8 tests passed in 0.325 seconds, no skips, including valid geometry/PBR GLB reload fixtures. These are synthetic CPU fixtures, not inference results. |
| Studio controller and two-image batch on Kaggle | PASS | The downloaded run records two validated geometry and PBR exports |
| Failure recovery on actual GPUs | UNVERIFIED | The completed batch used two successful jobs |
| Interactive preview, upload and downloads in Kaggle | UNVERIFIED | The UI rendered and the final cell displayed a ZIP link. Browser interaction and visual inspection remain open |
| Game-ready geometry/material quality | NOT VERIFIED | Requires visual/topology/UV inspection |

Latest local suite results are recorded in [the test log](tests/evidence/local-tests-20261107.txt).
Dependencies were installed into a scoped local test directory after the host's redirected
temporary folder blocked installation. The earlier eight-test Kaggle pass is recorded above.

The Kaggle installer displayed dependency conflicts involving unrelated preinstalled
Gradio, litellm and dopamine packages. Required runtime import checks still passed.
The studio does not use those packages. TensorFlow emitted a CUDA-driver warning;
PyTorch's separate CUDA preflight detected both GPUs and real shape diffusion ran.

Live private validation notebook:
https://www.kaggle.com/code/sachinprabhakarrao/private-3d-studio-validation/edit

The earlier upstream observed output is retained in
`tests/evidence/upstream-observed-output.txt`. A CPU-only private weights-builder
commit has also been started; dataset persistence must not be claimed until its
saved output and private dataset are confirmed.

The test's current session logs and output are under:
`/kaggle/working/private3d_validation/` and `/tmp/private3d-test/`.
Weights in `/tmp` are temporary until saved as a private dataset.

During the batch, GPU 0 generated the second shape while GPU 1 painted the first.
The private weights-builder run reached notebook HTML conversion. Its saved output
and the creation of a persistent private dataset remain unconfirmed.
