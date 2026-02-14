import torch
import numpy as np
from PIL import Image
from typing import Optional
import json
from datetime import datetime


class ScenarioEvaluator:
    """Automated quality evaluation for generated crash scenarios.

    Measures how well generated images match the intended scenario using:
    1. CLIP score — text-image alignment (does the image match the prompt?)
    2. YOLO verification — object detection (are the intended objects present?)
    3. Quality metrics — overall image quality assessment

    These metrics enable:
    - Quantitative comparison: raw SDXL vs ControlNet pipeline
    - Honest failure analysis: which crash elements the model can/can't render
    - Portfolio-grade evaluation methodology
    """

    def __init__(self, device: Optional[str] = None):
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self._clip_model = None
        self._clip_processor = None
        self._yolo_model = None

    def _load_clip(self):
        """Load CLIP model for text-image alignment scoring."""
        from transformers import CLIPModel, CLIPProcessor

        print("Loading CLIP for evaluation...")
        self._clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(self.device)
        self._clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self._clip_model.eval()

    def _load_yolo(self):
        """Load YOLOv8 nano for object detection verification."""
        from ultralytics import YOLO

        print("Loading YOLOv8n for object detection...")
        self._yolo_model = YOLO("yolov8n.pt")

    def clip_score(self, image: Image.Image, text: str) -> float:
        """Compute CLIP similarity score between image and text prompt.

        Higher score = image better matches the text description.
        Range: roughly 0.15-0.35 for typical generations.

        Args:
            image: Generated image
            text: Text prompt or scenario description

        Returns:
            CLIP cosine similarity score (float)
        """
        if self._clip_model is None:
            self._load_clip()

        inputs = self._clip_processor(
            text=[text], images=[image], return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._clip_model(**inputs)

        # Cosine similarity between image and text embeddings
        score = outputs.logits_per_image.item() / 100.0  # normalize from logits
        return round(score, 4)

    def detect_objects(self, image: Image.Image, confidence: float = 0.3) -> list[dict]:
        """Run YOLO object detection on generated image.

        Returns detected objects with class names and confidence.
        Used to verify: did the model actually render the pedestrian/vehicle/cyclist?

        Args:
            image: Generated image
            confidence: Minimum detection confidence threshold

        Returns:
            List of detected objects with class, confidence, and bounding box
        """
        if self._yolo_model is None:
            self._load_yolo()

        results = self._yolo_model(image, conf=confidence, verbose=False)

        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                detections.append({
                    "class": result.names[cls_id],
                    "confidence": round(float(box.conf[0]), 3),
                    "bbox": box.xyxy[0].tolist(),
                })

        return detections

    def verify_objects(
        self, image: Image.Image, expected_objects: list[str]
    ) -> dict:
        """Check if expected objects were actually generated.

        Compares the scene_objects from the parser (what we wanted)
        against YOLO detections (what was actually generated).

        Args:
            image: Generated image
            expected_objects: List of expected object types
                (e.g., ["pedestrian", "car", "guardrail"])

        Returns:
            Dict with found/missing objects and overall pass/fail
        """
        detections = self.detect_objects(image)
        detected_classes = {d["class"].lower() for d in detections}

        # Map scenario object types to YOLO class names
        type_mapping = {
            "pedestrian": {"person"},
            "car": {"car"},
            "truck": {"truck"},
            "cyclist": {"bicycle", "person"},
            "motorcycle": {"motorcycle"},
            "guardrail": set(),  # YOLO doesn't detect guardrails
            "debris": set(),  # YOLO doesn't detect debris
        }

        found = []
        missing = []
        for obj_type in expected_objects:
            yolo_classes = type_mapping.get(obj_type, {obj_type})
            if not yolo_classes:
                # YOLO can't detect this type — skip
                continue
            if yolo_classes & detected_classes:
                found.append(obj_type)
            else:
                missing.append(obj_type)

        detectable_count = len(found) + len(missing)
        return {
            "found": found,
            "missing": missing,
            "detection_rate": round(len(found) / detectable_count, 2) if detectable_count > 0 else 1.0,
            "all_detections": detections,
        }

    def temporal_clip_consistency(self, frames: list[Image.Image], text: str = "") -> dict:
        """Measure CLIP embedding consistency across video frames.

        Computes CLIP embeddings for each frame and reports:
        - Mean pairwise cosine similarity (higher = more consistent video)
        - Per-frame CLIP score against text prompt (if provided)

        This is a video-level quality metric: temporally consistent videos
        have high frame-to-frame CLIP similarity.

        Args:
            frames: List of video frame PIL Images
            text: Optional text prompt to score each frame against

        Returns:
            Dict with consistency metrics
        """
        if self._clip_model is None:
            self._load_clip()

        # Sample frames to avoid excessive computation (max 10 frames)
        step = max(1, len(frames) // 10)
        sampled = frames[::step][:10]

        embeddings = []
        per_frame_scores = []

        for frame in sampled:
            inputs = self._clip_processor(
                images=[frame], return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                img_emb = self._clip_model.get_image_features(**inputs)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                embeddings.append(img_emb)

            if text:
                per_frame_scores.append(self.clip_score(frame, text))

        # Compute pairwise cosine similarities
        if len(embeddings) > 1:
            emb_stack = torch.cat(embeddings, dim=0)
            sim_matrix = torch.mm(emb_stack, emb_stack.T)
            # Extract upper triangle (excluding diagonal)
            n = sim_matrix.shape[0]
            mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
            pairwise_sims = sim_matrix[mask].tolist()
            mean_consistency = round(float(np.mean(pairwise_sims)), 4)
            min_consistency = round(float(np.min(pairwise_sims)), 4)
        else:
            mean_consistency = 1.0
            min_consistency = 1.0

        return {
            "mean_frame_consistency": mean_consistency,
            "min_frame_consistency": min_consistency,
            "num_frames_sampled": len(sampled),
            "per_frame_clip_scores": per_frame_scores if per_frame_scores else None,
            "mean_frame_clip_score": round(float(np.mean(per_frame_scores)), 4) if per_frame_scores else None,
        }

    def evaluate_scenario(
        self,
        image: Image.Image,
        scenario,
        label: str = "",
        video_frames: Optional[list[Image.Image]] = None,
    ) -> dict:
        """Full evaluation of a generated scenario image + optional video.

        Args:
            image: Generated image
            scenario: CrashScenario with prompt and expected objects
            label: Optional label for this evaluation (e.g., "controlnet", "raw_sdxl")
            video_frames: Optional list of video frame images for temporal eval

        Returns:
            Complete evaluation report as dict
        """
        # CLIP score
        clip = self.clip_score(image, scenario.image_prompt or scenario.description)

        # Object verification
        expected = [obj.type for obj in scenario.scene_objects]
        obj_verify = self.verify_objects(image, expected) if expected else {
            "found": [], "missing": [], "detection_rate": 1.0, "all_detections": []
        }

        report = {
            "label": label,
            "timestamp": datetime.now().isoformat(),
            "scenario_description": scenario.description,
            "incident_type": scenario.incident_type,
            "clip_score": clip,
            "object_verification": obj_verify,
            "expected_objects": expected,
            "quality_assessment": self._assess_quality(clip, obj_verify),
        }

        # Video-level evaluation (if frames provided)
        if video_frames and len(video_frames) > 1:
            prompt_text = scenario.image_prompt or scenario.description
            report["video_metrics"] = self.temporal_clip_consistency(
                video_frames, text=prompt_text
            )

        return report

    def _assess_quality(self, clip_score: float, obj_verify: dict) -> str:
        """Human-readable quality grade based on metrics."""
        detection_rate = obj_verify.get("detection_rate", 0)

        if clip_score > 0.28 and detection_rate >= 0.8:
            return "GOOD — image matches prompt and contains expected objects"
        elif clip_score > 0.22 and detection_rate >= 0.5:
            return "MODERATE — partial match, some objects missing"
        elif clip_score > 0.18:
            return "WEAK — image captures atmosphere but misses specific objects"
        else:
            return "POOR — image does not match the intended scenario"

    def compare(
        self,
        images: dict[str, Image.Image],
        scenario,
    ) -> dict:
        """Compare multiple generation approaches for the same scenario.

        Args:
            images: Dict mapping approach name to generated image
                e.g., {"raw_sdxl": img1, "controlnet_depth": img2}
            scenario: CrashScenario

        Returns:
            Comparison report with per-approach scores and winner
        """
        results = {}
        for name, img in images.items():
            results[name] = self.evaluate_scenario(img, scenario, label=name)

        # Determine winner
        best_name = max(results, key=lambda n: results[n]["clip_score"])
        best_clip = results[best_name]["clip_score"]

        return {
            "scenario": scenario.description,
            "approaches": results,
            "best_approach": best_name,
            "best_clip_score": best_clip,
            "improvement": self._compute_improvement(results) if len(results) > 1 else None,
        }

    def _compute_improvement(self, results: dict) -> Optional[dict]:
        """Compute improvement of controlnet over raw_sdxl if both present."""
        if "raw_sdxl" in results and "controlnet_depth" in results:
            raw = results["raw_sdxl"]["clip_score"]
            cn = results["controlnet_depth"]["clip_score"]
            if raw > 0:
                pct = round((cn - raw) / raw * 100, 1)
                return {
                    "raw_sdxl_clip": raw,
                    "controlnet_clip": cn,
                    "improvement_pct": pct,
                }
        return None

    @staticmethod
    def save_report(report: dict, path: str):
        """Save evaluation report as JSON."""
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Saved evaluation report: {path}")


if __name__ == "__main__":
    from config.schema import CrashScenario, SceneObject

    evaluator = ScenarioEvaluator()

    # Create a test scenario
    scenario = CrashScenario(
        ego_speed_mps=20.0,
        weather="rain",
        lighting="day",
        road_type="highway",
        road_condition="wet",
        incident_type="hydroplane",
        severity="moderate",
        scene_objects=[
            SceneObject(type="guardrail", distance_m=10, lateral_position=-0.8),
        ],
        description="Vehicle hydroplaned on wet highway into guardrail",
        image_prompt="dashcam photo, wet highway, heavy rain, guardrail visible",
    )

    # Test with a dummy image (replace with real generated image in Colab)
    test_img = Image.new("RGB", (512, 512), color=(100, 100, 100))
    report = evaluator.evaluate_scenario(test_img, scenario, label="test")
    print(json.dumps(report, indent=2, default=str))
