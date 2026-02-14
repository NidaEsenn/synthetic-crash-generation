"""GeoDiffusion-inspired bounding box prompt augmentation.

Based on GeoDiffusion (arxiv 2306.04607): encodes spatial layout as position
tokens in text prompts, enabling pre-trained T2I models to understand WHERE
objects should appear without retraining.

The full GeoDiffusion paper fine-tunes the text encoder on detection data.
Our implementation is a lightweight approximation: we inject normalized
bbox coordinates directly into the prompt as text tokens. This works because
SDXL has some understanding of spatial descriptions, and combining this with
ControlNet depth conditioning gives two reinforcing spatial signals.

Example:
    Input:  "dashcam photo, car ahead on highway"
    Output: "dashcam photo, car at [0.35,0.40,0.65,0.70] ahead on highway"
"""

import numpy as np
from typing import Optional


class BboxPromptAugmenter:
    """Augments image prompts with GeoDiffusion-style bounding box tokens.

    Converts SceneObject positions (distance, lateral_position) into
    normalized [x1, y1, x2, y2] bounding boxes using perspective projection,
    then injects them as text tokens into the image prompt.
    """

    def scene_objects_to_bboxes(
        self,
        objects: list[dict],
        image_width: int = 1024,
        image_height: int = 1024,
    ) -> list[dict]:
        """Convert SceneObject list to normalized bounding boxes.

        Uses the same perspective projection math as DepthGenerator,
        ensuring consistency between depth map placement and prompt tokens.

        Args:
            objects: List of scene object dicts with distance_m, lateral_position, etc.
            image_width: Target image width
            image_height: Target image height

        Returns:
            List of dicts with 'type', 'bbox' [x1, y1, x2, y2] normalized to [0,1],
            and 'action'.
        """
        results = []
        for obj in objects:
            distance_m = obj.get("distance_m", 30.0)
            lateral = obj.get("lateral_position", 0.0)
            obj_width_frac = obj.get("width_fraction", 0.05)
            obj_height_frac = obj.get("height_fraction", 0.15)

            # Perspective projection (same math as depth_generator.py)
            vanishing_y = 0.4
            scale_factor = 1.0 / (1.0 + distance_m / 10.0)
            center_y_frac = vanishing_y + (1.0 - vanishing_y) * scale_factor

            center_x_frac = 0.5 + lateral * 0.4

            # Scale with distance
            size_scale = max(0.02, 1.0 / (1.0 + distance_m / 5.0))
            w_frac = obj_width_frac * size_scale * 3
            h_frac = obj_height_frac * size_scale * 3

            # Normalized bbox [x1, y1, x2, y2]
            x1 = max(0.0, center_x_frac - w_frac / 2)
            y1 = max(0.0, center_y_frac - h_frac / 2)
            x2 = min(1.0, center_x_frac + w_frac / 2)
            y2 = min(1.0, center_y_frac + h_frac / 2)

            results.append({
                "type": obj.get("type", "object"),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "action": obj.get("action", ""),
                "distance_m": distance_m,
            })

        return results

    def augment_prompt(
        self,
        base_prompt: str,
        scene_objects: list[dict],
        image_width: int = 1024,
        image_height: int = 1024,
    ) -> str:
        """Inject bounding box position tokens into an image prompt.

        Inspired by GeoDiffusion Section 3.2: spatial layout encoded as
        text tokens that the diffusion model can interpret.

        Args:
            base_prompt: Original image prompt from LLM parser
            scene_objects: List of SceneObject dicts
            image_width: Target image width
            image_height: Target image height

        Returns:
            Augmented prompt with bbox tokens inserted.
        """
        if not scene_objects:
            return base_prompt

        bboxes = self.scene_objects_to_bboxes(scene_objects, image_width, image_height)

        # Build position token strings
        position_tokens = []
        for bbox_info in bboxes:
            obj_type = bbox_info["type"]
            bbox = bbox_info["bbox"]
            action = bbox_info["action"]

            token = f"{obj_type} at [{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]"
            if action:
                token += f" {action}"
            position_tokens.append(token)

        # Insert position descriptions before the prompt's scene description
        bbox_text = ", ".join(position_tokens)
        augmented = f"{base_prompt}, with {bbox_text}"

        return augmented


if __name__ == "__main__":
    augmenter = BboxPromptAugmenter()

    # Test with sample scene objects
    objects = [
        {"type": "car", "distance_m": 30.0, "lateral_position": 0.0,
         "width_fraction": 0.1, "height_fraction": 0.12, "action": "approaching"},
        {"type": "pedestrian", "distance_m": 15.0, "lateral_position": 0.3,
         "width_fraction": 0.04, "height_fraction": 0.2, "action": "crossing"},
        {"type": "guardrail", "distance_m": 10.0, "lateral_position": -0.8,
         "width_fraction": 0.15, "height_fraction": 0.05, "action": "stationary"},
    ]

    # Test bbox computation
    bboxes = augmenter.scene_objects_to_bboxes(objects)
    print("Bounding boxes:")
    for b in bboxes:
        print(f"  {b['type']}: {b['bbox']} ({b['action']})")

    # Test prompt augmentation
    base_prompt = (
        "dashcam photo, point of view from inside a car, photorealistic, "
        "RAW photo, sharp focus, highway in rain, wet asphalt"
    )
    augmented = augmenter.augment_prompt(base_prompt, objects)
    print(f"\nBase prompt:\n  {base_prompt}")
    print(f"\nAugmented prompt:\n  {augmented}")
