"""End-to-end crash scene generation pipeline — A100 architecture.

Orchestrates the full multi-model pipeline with paper-inspired techniques:
1. Parse crash report → structured scene (Groq LLM, CrashAgent-inspired)
2. Generate bbox-augmented prompt (GeoDiffusion-inspired)
3. Generate/manipulate depth map (Depth Anything V2 + programmatic)
4. Generate conditioned image (ControlNet + SDXL + optional IP-Adapter)
5. Generate video (Ctrl-Crash / Wan2.1 / SVD)
6. Evaluate quality (CLIP + YOLO + temporal consistency)

Supports multiple video backends:
- "ctrl_crash": Crash-specific conditioned video (arxiv 2506.00227) — flagship
- "wan": Wan2.1 I2V 14B 480P — high-quality general-purpose
- "svd": Stable Video Diffusion — lightweight fallback
"""

import json
import os
import gc
import torch
from datetime import datetime
from PIL import Image
from typing import Optional

from config.schema import CrashScenario
from utils.vram_manager import VRAMManager
from utils.parser import LLMParser
from utils.depth_generator import DepthGenerator
from utils.controlnet_generator import ControlNetGenerator
from utils.bbox_prompt import BboxPromptAugmenter
from utils.bbox_sequence_generator import BboxSequenceGenerator
from utils.evaluator import ScenarioEvaluator


class CrashScenePipeline:
    """End-to-end pipeline: crash report text → dashcam video + evaluation.

    A100 40GB architecture with paper-inspired techniques:
    - GeoDiffusion bbox prompt augmentation for better spatial control
    - IP-Adapter for dashcam-realistic style from reference images
    - Ctrl-Crash for crash-specific video generation
    - Wan2.1 14B as high-quality fallback
    - Temporal CLIP consistency for video evaluation
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        video_backend: str = "ctrl_crash",
        use_ip_adapter: bool = False,
        use_bbox_prompts: bool = True,
        skip_video: bool = False,
        skip_eval: bool = False,
        gpu_profile: str = "a100_40",
    ):
        """
        Args:
            output_dir: Directory for generated outputs
            video_backend: "ctrl_crash", "wan", or "svd"
            use_ip_adapter: Enable IP-Adapter reference image conditioning
            use_bbox_prompts: Enable GeoDiffusion-style bbox prompt augmentation
            skip_video: If True, only generate images
            skip_eval: If True, skip evaluation
            gpu_profile: GPU profile for VRAM tracking ("a100_40", "t4", etc.)
        """
        self.output_dir = output_dir
        self.video_backend = video_backend
        self.use_ip_adapter = use_ip_adapter
        self.use_bbox_prompts = use_bbox_prompts
        self.skip_video = skip_video
        self.skip_eval = skip_eval

        os.makedirs(output_dir, exist_ok=True)

        # Shared VRAM manager
        self.vram = VRAMManager(gpu_profile=gpu_profile)

        # Initialize components (models loaded on demand)
        self.parser = LLMParser()
        self.depth_gen = DepthGenerator(vram_manager=self.vram)
        self.controlnet_gen = ControlNetGenerator(
            vram_manager=self.vram,
            use_ip_adapter=use_ip_adapter,
        )
        self.bbox_augmenter = BboxPromptAugmenter()
        self.bbox_seq_gen = BboxSequenceGenerator()
        self.evaluator = ScenarioEvaluator(device=self.vram.device)

        # Video generator loaded on demand based on backend
        self._video_gen = None

    def _get_video_generator(self):
        """Lazily load the selected video backend."""
        if self._video_gen is not None:
            return self._video_gen

        if self.video_backend == "ctrl_crash":
            from utils.ctrl_crash_generator import CtrlCrashGenerator
            self._video_gen = CtrlCrashGenerator(vram_manager=self.vram)
        elif self.video_backend == "wan":
            from utils.wan_video_generator import WanVideoGenerator
            self._video_gen = WanVideoGenerator(vram_manager=self.vram)
        else:  # "svd" fallback
            from utils.video_generator import VideoGenerator
            self._video_gen = VideoGenerator(vram_manager=self.vram)

        return self._video_gen

    def run(self, crash_report: str, scenario_name: str = "scenario") -> dict:
        """Run the full pipeline on a single crash report.

        Args:
            crash_report: Natural language crash report text
            scenario_name: Name for output files (e.g., "wet_highway")

        Returns:
            Complete result dict with paths, metrics, and VRAM report
        """
        result = {
            "scenario_name": scenario_name,
            "crash_report": crash_report,
            "timestamp": datetime.now().isoformat(),
            "video_backend": self.video_backend,
            "ip_adapter": self.use_ip_adapter,
            "bbox_prompts": self.use_bbox_prompts,
        }

        # === Stage 1: Parse crash report (CPU, no GPU) ===
        print(f"\n{'='*60}")
        print(f"STAGE 1: Parsing crash report")
        print(f"{'='*60}")
        scenario = self.parser.parse(crash_report)
        result["scenario"] = scenario.model_dump()
        print(f"  Incident: {scenario.incident_type}")
        print(f"  Objects: {[o.type for o in scenario.scene_objects]}")
        print(f"  Weather: {scenario.weather}, Road: {scenario.road_type}")

        # === Stage 1b: GeoDiffusion bbox prompt augmentation (CPU) ===
        if self.use_bbox_prompts and scenario.scene_objects:
            print(f"\n  Augmenting prompt with bbox tokens (GeoDiffusion)...")
            scene_obj_dicts = [obj.model_dump() for obj in scenario.scene_objects]
            augmented_prompt = self.bbox_augmenter.augment_prompt(
                scenario.image_prompt, scene_obj_dicts
            )
            scenario.bbox_augmented_prompt = augmented_prompt
            print(f"  Augmented: ...{augmented_prompt[-80:]}")

        # === Stage 2: Generate depth map + bbox sequence (<2GB) ===
        print(f"\n{'='*60}")
        print(f"STAGE 2: Generating depth map + bbox sequence")
        print(f"{'='*60}")
        base_depth = self.depth_gen.create_base_depth(scenario.road_type)

        scene_obj_dicts = [obj.model_dump() for obj in scenario.scene_objects]
        if scene_obj_dicts:
            manipulated_depth = self.depth_gen.manipulate_depth(base_depth, scene_obj_dicts)
        else:
            manipulated_depth = base_depth

        depth_img = self.depth_gen.depth_to_pil(manipulated_depth)
        depth_path = os.path.join(self.output_dir, f"{scenario_name}_depth.png")
        depth_img.save(depth_path)
        result["depth_path"] = depth_path
        print(f"  Saved depth map: {depth_path}")

        # Generate bbox sequence for Ctrl-Crash conditioning
        bbox_sequence = None
        if self.video_backend == "ctrl_crash" and scene_obj_dicts:
            bbox_sequence = self.bbox_seq_gen.generate_sequence(scene_obj_dicts)
            print(f"  Generated {len(bbox_sequence)}-frame bbox sequence")

        # === Stage 3: Generate conditioned image (~14GB with IP-Adapter) ===
        print(f"\n{'='*60}")
        print(f"STAGE 3: Generating conditioned image (ControlNet + SDXL)")
        if self.use_ip_adapter:
            print(f"         + IP-Adapter style conditioning")
        print(f"{'='*60}")

        # Use bbox-augmented prompt if available
        prompt_to_use = scenario.bbox_augmented_prompt or scenario.image_prompt

        image = self.controlnet_gen.generate(
            prompt=prompt_to_use,
            depth_image=depth_img,
            controlnet_conditioning_scale=0.25,
        )
        image_path = os.path.join(self.output_dir, f"{scenario_name}_image.png")
        image.save(image_path)
        result["image_path"] = image_path
        print(f"  Saved image: {image_path}")

        # Stage 3b: Crash-moment keyframes
        print(f"\n  Generating crash-moment keyframes...")
        keyframes = self.controlnet_gen.generate_crash_keyframes(
            scenario=scenario,
            depth_image=depth_img,
        )
        result["keyframe_paths"] = {}
        for timepoint, kf_image in keyframes.items():
            kf_path = os.path.join(self.output_dir, f"{scenario_name}_{timepoint}.png")
            kf_image.save(kf_path)
            result["keyframe_paths"][timepoint] = kf_path
            print(f"  Saved keyframe [{timepoint}]: {kf_path}")

        # === Stage 4: Generate video ===
        frames = None
        if not self.skip_video:
            print(f"\n{'='*60}")
            print(f"STAGE 4: Generating video ({self.video_backend})")
            print(f"{'='*60}")

            # Unload ControlNet to make room for video model
            self.controlnet_gen.unload_model()
            gc.collect()
            if self.vram.device == "cuda":
                torch.cuda.empty_cache()

            video_gen = self._get_video_generator()

            if self.video_backend == "ctrl_crash":
                frames = video_gen.generate_from_scenario(
                    scenario=scenario,
                    source_image=image,
                    bbox_sequence=bbox_sequence,
                )
            elif self.video_backend == "wan":
                frames = video_gen.generate_from_scenario(
                    scenario=scenario,
                    source_image=image,
                )
            else:  # svd
                frames = video_gen.generate_from_scenario(
                    scenario=scenario,
                    source_image=image,
                )

            video_path = os.path.join(self.output_dir, f"{scenario_name}_video.mp4")
            video_gen.export_video(frames, video_path)
            result["video_path"] = video_path

            video_gen.unload_model()
        else:
            self.controlnet_gen.unload_model()

        # === Stage 5: Evaluate ===
        if not self.skip_eval:
            print(f"\n{'='*60}")
            print(f"STAGE 5: Evaluating quality")
            print(f"{'='*60}")
            eval_report = self.evaluator.evaluate_scenario(
                image=image,
                scenario=scenario,
                label=f"controlnet_{self.video_backend}",
                video_frames=frames,
            )
            result["evaluation"] = eval_report
            print(f"  CLIP score: {eval_report['clip_score']}")
            print(f"  Quality: {eval_report['quality_assessment']}")
            if "video_metrics" in eval_report:
                vm = eval_report["video_metrics"]
                print(f"  Video consistency: {vm['mean_frame_consistency']}")

            eval_path = os.path.join(self.output_dir, f"{scenario_name}_eval.json")
            self.evaluator.save_report(eval_report, eval_path)
            result["eval_path"] = eval_path

        # === VRAM Report ===
        result["vram_report"] = self.vram.report()
        self.vram.print_report()

        # Save full result
        result_path = os.path.join(self.output_dir, f"{scenario_name}_result.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nFull result saved: {result_path}")

        return result

    def run_batch(self, reports: dict[str, str]) -> list[dict]:
        """Run pipeline on multiple crash reports.

        Args:
            reports: Dict mapping scenario names to crash report text

        Returns:
            List of result dicts
        """
        results = []
        for i, (name, report) in enumerate(reports.items()):
            print(f"\n{'#'*60}")
            print(f"# Scenario {i+1}/{len(reports)}: {name}")
            print(f"# Report: {report}")
            print(f"{'#'*60}")
            result = self.run(report, scenario_name=name)
            results.append(result)

        # Summary
        print(f"\n{'='*60}")
        print(f"BATCH COMPLETE: {len(results)} scenarios generated")
        print(f"Video backend: {self.video_backend}")
        print(f"{'='*60}")
        for r in results:
            clip = r.get("evaluation", {}).get("clip_score", "N/A")
            video_con = r.get("evaluation", {}).get("video_metrics", {}).get(
                "mean_frame_consistency", "N/A"
            )
            print(f"  {r['scenario_name']}: CLIP={clip}, Video={video_con}")

        return results


if __name__ == "__main__":
    # Quick test with image-only mode (no video, no eval — for fast iteration)
    pipeline = CrashScenePipeline(
        output_dir="outputs",
        video_backend="ctrl_crash",
        use_bbox_prompts=True,
        skip_video=True,
        skip_eval=True,
        gpu_profile="a100_40",
    )

    result = pipeline.run(
        crash_report="Vehicle traveling 45mph on wet highway, hydroplaned into guardrail",
        scenario_name="test_hydroplane",
    )
