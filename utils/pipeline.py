import json
import os
from datetime import datetime
from PIL import Image
from typing import Optional

from config.schema import CrashScenario
from utils.vram_manager import VRAMManager
from utils.parser import LLMParser
from utils.depth_generator import DepthGenerator
from utils.controlnet_generator import ControlNetGenerator
from utils.video_generator import VideoGenerator
from utils.evaluator import ScenarioEvaluator


class CrashScenePipeline:
    """End-to-end pipeline: crash report text -> dashcam video + evaluation.

    Orchestrates the full multi-model pipeline with VRAM-aware sequential loading:
    1. Parse crash report -> structured scene representation (Groq LLM)
    2. Generate/manipulate depth map (Depth Anything V2 + programmatic)
    3. Generate conditioned image (ControlNet + SDXL)
    4. Generate video from image/text (Wan2.1)
    5. Evaluate quality (CLIP + YOLO)

    Models are loaded and unloaded sequentially to fit within T4's 15GB VRAM.
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        video_mode: str = "t2v",
        skip_video: bool = False,
        skip_eval: bool = False,
    ):
        """
        Args:
            output_dir: Directory for generated outputs
            video_mode: "t2v" (text-to-video, 1.3B) or "i2v" (image-to-video, 5B)
            skip_video: If True, only generate images (faster iteration)
            skip_eval: If True, skip CLIP/YOLO evaluation
        """
        self.output_dir = output_dir
        self.video_mode = video_mode
        self.skip_video = skip_video
        self.skip_eval = skip_eval

        os.makedirs(output_dir, exist_ok=True)

        # Shared VRAM manager tracks memory across all stages
        self.vram = VRAMManager()

        # Initialize components (models loaded on demand)
        self.parser = LLMParser()
        self.depth_gen = DepthGenerator(vram_manager=self.vram)
        self.controlnet_gen = ControlNetGenerator(vram_manager=self.vram)
        self.video_gen = VideoGenerator(vram_manager=self.vram, mode=video_mode)
        self.evaluator = ScenarioEvaluator(device=self.vram.device)

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
        }

        # --- Stage 1: Parse crash report (CPU, no GPU needed) ---
        print(f"\n{'='*60}")
        print(f"STAGE 1: Parsing crash report")
        print(f"{'='*60}")
        scenario = self.parser.parse(crash_report)
        result["scenario"] = scenario.model_dump()
        print(f"  Incident: {scenario.incident_type}")
        print(f"  Objects: {[o.type for o in scenario.scene_objects]}")
        print(f"  Weather: {scenario.weather}, Road: {scenario.road_type}")

        # --- Stage 2: Generate depth map (lightweight, <2GB) ---
        print(f"\n{'='*60}")
        print(f"STAGE 2: Generating depth map")
        print(f"{'='*60}")
        base_depth = self.depth_gen.create_base_depth(scenario.road_type)

        # Manipulate depth with scene objects from parser
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

        # --- Stage 3: Generate conditioned image (10-12GB) ---
        print(f"\n{'='*60}")
        print(f"STAGE 3: Generating conditioned image (ControlNet + SDXL)")
        print(f"{'='*60}")
        self.depth_gen.unload_model()  # free depth model VRAM before loading SDXL

        image = self.controlnet_gen.generate_from_scenario(
            scenario=scenario,
            depth_image=depth_img,
            controlnet_conditioning_scale=0.5,
        )
        image_path = os.path.join(self.output_dir, f"{scenario_name}_image.png")
        image.save(image_path)
        result["image_path"] = image_path
        print(f"  Saved image: {image_path}")

        # --- Stage 3b: Generate crash-moment keyframe ---
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

        # --- Stage 4: Generate video (8-10GB, after unloading ControlNet) ---
        if not self.skip_video:
            print(f"\n{'='*60}")
            print(f"STAGE 4: Generating video (Wan2.1)")
            print(f"{'='*60}")
            self.controlnet_gen.unload_model()  # free ControlNet before loading Wan

            if self.video_mode == "i2v":
                frames = self.video_gen.generate_from_scenario(
                    scenario=scenario, source_image=image
                )
            else:
                frames = self.video_gen.generate_from_scenario(scenario=scenario)

            video_path = os.path.join(self.output_dir, f"{scenario_name}_video.mp4")
            self.video_gen.export_video(frames, video_path)
            result["video_path"] = video_path

            self.video_gen.unload_model()  # free video model
        else:
            self.controlnet_gen.unload_model()

        # --- Stage 5: Evaluate (lightweight, <1GB) ---
        if not self.skip_eval:
            print(f"\n{'='*60}")
            print(f"STAGE 5: Evaluating quality")
            print(f"{'='*60}")
            eval_report = self.evaluator.evaluate_scenario(
                image=image, scenario=scenario, label="controlnet_depth"
            )
            result["evaluation"] = eval_report
            print(f"  CLIP score: {eval_report['clip_score']}")
            print(f"  Quality: {eval_report['quality_assessment']}")

            # Save evaluation report
            eval_path = os.path.join(self.output_dir, f"{scenario_name}_eval.json")
            self.evaluator.save_report(eval_report, eval_path)
            result["eval_path"] = eval_path

        # --- VRAM Report ---
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
        print(f"{'='*60}")
        for r in results:
            clip = r.get("evaluation", {}).get("clip_score", "N/A")
            print(f"  {r['scenario_name']}: CLIP={clip}")

        return results


if __name__ == "__main__":
    # Quick test with image-only mode (no video, no eval — for fast iteration)
    pipeline = CrashScenePipeline(
        output_dir="outputs",
        skip_video=True,
        skip_eval=True,
    )

    result = pipeline.run(
        crash_report="Vehicle traveling 45mph on wet highway, hydroplaned into guardrail",
        scenario_name="test_hydroplane",
    )
