# Source review and implementation checklist

Reviewed the complete functional community implementation at revision
`b06a56a42b73257d31e877bcdf2106613177cd6a`: README, MIT license,
`build_notebook.py`, both generated notebooks, and repository tree. The only
remaining assets are a promotional cover image and .gitignore; they contain no
inference functionality. The original generator is retained in `upstream` and used
to reproduce the adapted workers. Generated notebook copies were removed in favour
of the single Kaggle/Colab application notebook.

Official runtime review covered `hy3dshape/hy3dshape/pipelines.py`,
`hy3dpaint/textureGenPipeline.py`, `hy3dpaint/utils/multiview_utils.py`,
`hy3dpaint/convert_utils.py`, and the official example-image listing. Runtime
revision is pinned/checked in the installer.

| Requirement | Implementation |
| --- | --- |
| Shape loading | Retained `Hunyuan3DDiTFlowMatchingPipeline.from_pretrained`; local `HY3DGEN_MODELS`, FP16 default, original inference call and face reducer |
| Paint loading | Retained `Hunyuan3DPaintPipeline`, local snapshot override, DINO and RealESRGAN checkpoint paths |
| Two T4s | Separate processes with `CUDA_VISIBLE_DEVICES=0/1`; no shared-memory claim |
| Queue | Original atomic filesystem enqueue retained; unique run queues, atomic claim and stage records added |
| Reduced-memory behavior | FP16 model initialization, memory-mapped zip checkpoints, six views/512 default, spandrel upscaler and face-limit remesher |
| Dependency compatibility | Existing numpy/Python branching, pinned dependencies, optional bpy, C++ inpainter and per-device rasterizer build retained |
| Weights | Original official download URLs and 16-reader staging retained; dynamic discovery, integrity manifest and automatic missing downloads in the main notebook |
| GLB/PBR | Original metallic/roughness packing and MIME/alignment repair retained; UV/PBR/triangle validation before final texture export added |
| User interface | Three-column private notebook widgets, example gallery, reference upload, separate generation buttons, settings, local viewer and multi-format downloads |
| Recovery | Input/settings identity, append-only logs, completed-output reuse, separate metadata and checkpoint ZIP added |
| Single GPU | Geometry worker exits before painting loads on the same GPU; scheduling checked with synthetic fixtures; live Colab GPU test pending |
| Colab | Portable working paths, widget-manager setup, direct notebook badge and native file-download helper |
| Privacy | No public web app, tunnel, hosted model API or external image upload |
| Validation | CPU suite and executable GPU smoke test; actual outcomes recorded in TEST_RESULTS.md |

`tools/adapt_workers.py` regenerates the four worker helpers from the retained
original generator. `workers/install.sh` is maintained separately because it
also adds runtime-revision checks and modification notices. Model algorithms,
sampling and PBR generation are not replaced with simulated results.
