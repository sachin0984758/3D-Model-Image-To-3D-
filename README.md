# Private 3D Studio

Turn a reference image into a 3D model with Hunyuan3D-2.1. One notebook installs the runtime, downloads model weights and opens an image-to-3D interface on **Google Colab or Kaggle**. Your selected cloud runtime performs the generation.

## Open the studio

| Notebook | Description | Kaggle | Colab | Lightning AI | Hugging Face | Video |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| [**Private 3D Studio**](private_3d_studio.ipynb) | Image uploads, example gallery, shape and PBR texture generation, interactive 3D preview, mesh simplification and GLB / OBJ / PLY / STL exports. Uses two Kaggle T4s in parallel or one Colab GPU sequentially. | [![Import on Kaggle](https://img.shields.io/badge/Kaggle-Import-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/new) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sachin0984758/3D-Model-Image-To-3D-/blob/Python_Study/private_3d_studio.ipynb) | — | [![Model weights](https://img.shields.io/badge/Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/tencent/Hunyuan3D-2.1) | — |

**Open In Colab** opens the notebook directly. **Import on Kaggle** opens Kaggle's notebook editor, where you import the linked `.ipynb` through **File > Import Notebook**. Both use `private_3d_studio.ipynb`.

To update an existing Kaggle notebook, import this file into that notebook and use **Private 3D Studio** as its title. Download any existing outputs first. A separate weights-builder or validation notebook is no longer needed. The optional validation cells are included in the studio.

## Run it

| Platform | Before running | Start |
| --- | --- | --- |
| Google Colab | Open the Colab badge. Choose **Runtime > Change runtime type > T4 GPU**. | **Runtime > Run all** |
| Kaggle | Import the notebook. Choose **GPU T4 x2**, enable **Internet** and check **Private** visibility. You may attach previously saved weights through **Add Input**. | **Run All** |

The first setup downloads roughly **18 GiB** of weights and compiles the rendering extensions. Existing downloads are reused. GPU availability depends on the platform and your remaining quota.

When **Private 3D Studio** appears:

1. Upload a PNG, JPEG or WebP, or select an example from the gallery. Use an image with one clear object. Multiple uploads form a batch.
2. Start with **Medium** settings. Click **Generate Shape** for geometry or **Generate Textured Shape** for geometry with PBR textures.
3. Select the completed model in the centre panel. Drag to rotate, use the wheel to zoom, or Shift-drag to pan.
4. Open **Export**, choose a format and click **Prepare export > Download**. Use **Download all / checkpoint** to save the full generation output and recovery records.

The interface has reference images and settings on the left, the 3D preview in the centre and clickable examples on the right. **Advanced options** includes background removal, seed, steps, octree resolution, guidance, chunks, face count and texture quality. **Export** includes optional simplification and a target face count.

GLB contains its PBR textures. Textured OBJ downloads include the mesh, material and image files together in a ZIP, including roughness and metallic maps. PLY and STL export geometry. Simplifying a textured mesh preserves its UV coordinates and PBR material. The generated source files are retained.

## GPU and storage

On two GPUs, geometry runs in one process on GPU 0 and painting starts on GPU 1 after the first shape is queued. On one GPU, the geometry process exits before the paint process loads. GPU memory is released between those stages. Two T4s do not form a shared 32 GB allocation.

The first-run cache and studio output use these paths:

| Data | Kaggle | Colab |
| --- | --- | --- |
| Weights | `/kaggle/working/private-3d-weights` | `/content/private-3d-weights` |
| Studio files | `/kaggle/working/private_3d_studio` | `/content/private_3d_studio` |
| Model ZIP | `private_3d_studio/exports/generated_models.zip` under the runtime's working folder | Same relative path |

The notebook downloads checkpoints from their official sources, validates them and loads the local copies for inference. Kaggle can also discover an attached weights dataset or saved notebook output. Set `DATASET` in the weights cell when selecting among several attached sources. Set `ALLOW_DOWNLOAD=False` to require existing weights.

Runtime files are temporary. On Kaggle, save a **Private** notebook version with outputs to retain the weights cache, then attach that output through **Add Input** in a later session. On Colab, download your generated models before the runtime is cleared. The file-download helper starts Colab downloads directly. Kaggle displays a relative file link and also lists exports in Output.

## Recovery

Each model has its own folder with input, meshes, settings, validation and per-stage records. Repeating an input with the same settings reuses validated completed stages. Different settings create another model folder. Failed painting retains valid geometry for retry or download.

To resume in a new runtime, restore the saved `outputs/` folder under `private_3d_studio/`, upload the same references and use the same settings. Keep `job.json` and the stage metadata with each mesh. **Stop** terminates the workers launched by the current studio. Work must be saved before a runtime is terminated.

## Validation

A saved run on **two Kaggle T4s** generated two textured GLBs with **40,000 faces each**. Both passed geometry, UV, embedded-image and PBR material checks. See [TEST_RESULTS.md](TEST_RESULTS.md) for timings and the original evidence.

The Colab integration and revised interface have local CPU checks using synthetic fixtures. A real Colab GPU run remains to be verified. High settings may exceed T4 memory. Lower the octree resolution and face count for geometry, or use six views / 256 resolution for painting if memory is exhausted. The texture-resolution control is the diffusion input resolution, not the final baked atlas size.

The notebook includes an optional two-image GPU smoke test. Set `RUN_GPU_TEST=True` and run that cell manually after setup. It consumes GPU quota and records actual outcomes. Inspect the resulting GLBs visually as well as checking the metadata.

## Project files

| Path | Purpose |
| --- | --- |
| `private_3d_studio.ipynb` | The single notebook for Kaggle and Colab, with embedded application source |
| `studio/` | Images, weights, scheduling, interface, viewer and exports |
| `workers/` | Installation, downloads, geometry and PBR painting |
| `tests/` | CPU tests and saved GPU-run evidence |
| `tools/` | Reproducible worker adaptation, notebook building and verification |
| `upstream/` | Retained original generator and provenance |
| `LICENSE`, `NOTICE`, `TENCENT_LICENSE.txt` | Copyright notices and applicable terms |

For development, run from the project folder with Pillow, NumPy, trimesh, PyMeshLab, IPython and ipywidgets installed. Install nbformat for notebook schema validation.

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m unittest discover -s tests -v
python tools/build_notebook.py
python tools/verify_notebook.py
```

`tools/adapt_workers.py` regenerates the workers from the retained original generator. `workers/install.sh` runs inside the Linux cloud runtime with its existing PyTorch. Running the notebook does not require a local GPU or local model installation. For unattended generation, call `studio.run(images, settings)` explicitly in a notebook cell because widget clicks do not execute during a noninteractive commit.

## Licenses

The original notebook implementation is MIT licensed by Gaurav Kumar. Its generator, copyright and license are retained. The official Tencent runtime is pinned to revision `82920d643c0dc2f7bfd7255f45f62d386edfe60c`. Model use follows the **Tencent Hunyuan 3D 2.1 Community License**. Read the included terms and component licenses before redistribution or commercial use. Private 3D Studio is personal application branding and is not affiliated with Tencent.
