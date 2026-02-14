"""Temporal bounding box sequence generation for Ctrl-Crash conditioning.

Inspired by DriveDreamer (arxiv 2309.09777) temporal trajectory reasoning:
generates per-frame bounding box sequences showing how objects move over time.

Ctrl-Crash (arxiv 2506.00227) conditions video generation on:
1. An initial dashcam frame
2. Per-frame bounding boxes showing vehicle trajectories
3. Crash type classification

This module generates (2) — interpolating SceneObject positions over N frames
based on their action (approaching, crossing, stopped, etc.).
"""

import numpy as np
from utils.bbox_prompt import BboxPromptAugmenter


class BboxSequenceGenerator:
    """Generates temporal bounding box sequences for video conditioning.

    Takes static scene_objects (single-frame positions) and interpolates
    motion trajectories over N frames based on each object's action type.
    """

    def __init__(self):
        self.bbox_augmenter = BboxPromptAugmenter()

    def generate_sequence(
        self,
        scene_objects: list[dict],
        num_frames: int = 25,
        fps: int = 7,
        image_width: int = 1024,
        image_height: int = 576,
    ) -> list[list[dict]]:
        """Generate per-frame bounding box sequence from scene objects.

        Each object's trajectory depends on its action:
        - 'approaching': starts far, gets closer (bbox grows)
        - 'crossing': moves laterally across the frame
        - 'stopped'/'stationary': stays in place
        - 'merging': shifts toward ego lane
        - 'turning': curves away laterally

        Args:
            scene_objects: List of SceneObject dicts (from CrashScenario)
            num_frames: Number of video frames to generate bboxes for
            fps: Frames per second (for timing calculations)
            image_width: Video frame width
            image_height: Video frame height

        Returns:
            List of N frame-level bbox lists. Each frame contains a list of
            dicts with 'type', 'bbox' [x1, y1, x2, y2] normalized to [0, 1].
        """
        duration_s = num_frames / fps
        frames = []

        for frame_idx in range(num_frames):
            t = frame_idx / max(num_frames - 1, 1)  # 0.0 to 1.0
            frame_bboxes = []

            for obj in scene_objects:
                interpolated_obj = self._interpolate_object(obj, t, duration_s)
                bbox_list = self.bbox_augmenter.scene_objects_to_bboxes(
                    [interpolated_obj], image_width, image_height
                )
                if bbox_list:
                    frame_bboxes.append(bbox_list[0])

            frames.append(frame_bboxes)

        return frames

    def _interpolate_object(
        self, obj: dict, t: float, duration_s: float
    ) -> dict:
        """Interpolate a single object's position at time t.

        Args:
            obj: SceneObject dict with initial position
            t: Normalized time (0.0 = start, 1.0 = end of clip)
            duration_s: Total duration in seconds

        Returns:
            Modified SceneObject dict with interpolated position.
        """
        action = obj.get("action", "stationary") or "stationary"
        distance_m = obj.get("distance_m", 30.0)
        lateral = obj.get("lateral_position", 0.0)

        result = dict(obj)  # copy

        if action == "approaching":
            # Object gets closer: distance decreases over time
            # Accelerates toward the end (crash moment)
            # Use quadratic ease-in: objects approach faster as they get near
            approach_factor = t ** 1.5
            min_distance = max(2.0, distance_m * 0.1)
            result["distance_m"] = distance_m - (distance_m - min_distance) * approach_factor

        elif action == "crossing":
            # Object moves laterally across the frame
            # Start from one side, cross to the other
            cross_range = 1.2  # total lateral distance to cross
            start_lateral = lateral - cross_range / 2
            result["lateral_position"] = start_lateral + cross_range * t

        elif action in ("stopped", "stationary"):
            # Object doesn't move, but ego vehicle approaches it
            # So the object appears to get slightly closer
            ego_approach = distance_m * 0.3 * t
            result["distance_m"] = max(2.0, distance_m - ego_approach)

        elif action == "merging":
            # Object shifts toward ego lane (lateral → 0)
            result["lateral_position"] = lateral * (1.0 - t * 0.7)
            # Also gets slightly closer
            result["distance_m"] = max(5.0, distance_m - distance_m * 0.2 * t)

        elif action == "turning":
            # Object curves away laterally
            turn_direction = 1.0 if lateral >= 0 else -1.0
            result["lateral_position"] = lateral + turn_direction * 0.5 * t
            result["distance_m"] = distance_m + 5.0 * t  # moves away

        return result

    def sequence_to_frames_array(
        self,
        sequence: list[list[dict]],
        image_width: int = 1024,
        image_height: int = 576,
        max_objects: int = 5,
    ) -> np.ndarray:
        """Convert bbox sequence to a numpy array for model conditioning.

        Some models expect bbox conditioning as a tensor of shape
        [num_frames, max_objects, 4] with zero-padding.

        Args:
            sequence: Output from generate_sequence()
            image_width: Frame width (for denormalization if needed)
            image_height: Frame height
            max_objects: Maximum objects per frame (zero-padded)

        Returns:
            numpy array of shape [num_frames, max_objects, 4],
            with normalized [x1, y1, x2, y2] coordinates.
        """
        num_frames = len(sequence)
        result = np.zeros((num_frames, max_objects, 4), dtype=np.float32)

        for f_idx, frame_bboxes in enumerate(sequence):
            for o_idx, bbox_info in enumerate(frame_bboxes[:max_objects]):
                result[f_idx, o_idx] = bbox_info["bbox"]

        return result


if __name__ == "__main__":
    gen = BboxSequenceGenerator()

    # Test scenario: car approaching from ahead
    objects = [
        {"type": "car", "distance_m": 40.0, "lateral_position": 0.0,
         "width_fraction": 0.1, "height_fraction": 0.12, "action": "approaching"},
        {"type": "pedestrian", "distance_m": 20.0, "lateral_position": 0.3,
         "width_fraction": 0.04, "height_fraction": 0.2, "action": "crossing"},
    ]

    sequence = gen.generate_sequence(objects, num_frames=25, fps=7)

    print(f"Generated {len(sequence)} frames")
    for i in [0, 6, 12, 18, 24]:
        print(f"\nFrame {i}:")
        for bbox in sequence[i]:
            print(f"  {bbox['type']}: {bbox['bbox']}")

    # Test array conversion
    arr = gen.sequence_to_frames_array(sequence)
    print(f"\nArray shape: {arr.shape}")
    print(f"Frame 0, Object 0: {arr[0, 0]}")
    print(f"Frame 24, Object 0: {arr[24, 0]}")
