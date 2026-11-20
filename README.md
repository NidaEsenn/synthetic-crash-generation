# CrashGen — synthetic crash scenes for autonomous-driving data

Crash events are well under 1% of real driving data, and you cannot stage them safely.
CrashGen turns a plain-text crash report into a dashcam-style image or video, so rare,
high-risk edge cases can be added to perception training sets.

Background, related work and design notes: [CrashGen write-up](https://github.com/NidaEsenn/rare-case-car-crash-scenario-generation).

## Pipeline

```
crash report (text)
  → LLM parser (Groq, Llama 3.3 70B) → structured CrashScenario (Pydantic)
  → bbox-augmented prompt (GeoDiffusion-style spatial tokens)
  → depth map (Depth Anything V2 + programmatic edits)
  → image: SDXL + depth ControlNet (optional IP-Adapter, optional dashcam LoRA)
  → video: Ctrl-Crash | Wan2.1 I2V | Stable Video Diffusion
  → evaluation: CLIP text–image score, YOLO object check, temporal CLIP consistency
```

All stages are orchestrated by `CrashScenePipeline` in `utils/pipeline.py`, which also
tracks VRAM per stage and unloads models between steps (`utils/vram_manager.py`).

## Repository layout

| Path | What it does |
|---|---|
| `config/schema.py` | `CrashScenario`, `SceneObject`, weather and incident-type enums |
| `utils/parser.py` | Crash report → structured scenario via LLM |
| `utils/bbox_prompt.py`, `utils/bbox_sequence_generator.py` | Bounding-box prompts and per-frame box trajectories |
| `utils/depth_generator.py` | Depth maps for ControlNet conditioning |
| `utils/controlnet_generator.py`, `utils/image_generator.py` | SDXL image generation |
| `utils/ctrl_crash_generator.py`, `utils/wan_video_generator.py`, `utils/video_generator.py` | Video backends |
| `utils/evaluator.py` | Automatic quality checks |
| `scripts/prepare_dashcam_data.py` | Pulls dashcam frames (CCD, DoTA, Nexar, BDD100K) for LoRA training |
| `scripts/train_dashcam_lora.py` | LoRA fine-tuning of SDXL on dashcam frames, LoRA layers written from scratch |
| `notebooks/demo_pipeline.ipynb` | End-to-end demo |

## Quick start

Requires Python 3.13 and a CUDA GPU (the default profile targets an A100 40 GB; use
`skip_video=True` on smaller cards).

```bash
uv sync            # or: pip install -r requirements.txt
export GROQ_API_KEY=...
```

```python
from utils.pipeline import CrashScenePipeline

pipe = CrashScenePipeline(video_backend="svd", skip_video=False)
result = pipe.run(
    "At dusk on a wet highway, a sedan rear-ends a stopped truck in the right lane.",
    scenario_name="rear_end_wet",
)
print(result)
```

Outputs (images, videos, scores) are written to `outputs/`.

## Status

Research project, in progress. Image generation, the LoRA training script and the
evaluation stage are implemented; video backends depend on large checkpoints and GPU memory.
