import torch
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from typing import Optional
from utils.vram_manager import VRAMManager


class DepthGenerator:
    """Depth map generation using Depth Anything V2 + custom scene manipulation.

    Two modes of operation:
    1. Extract depth from a reference image (Depth Anything V2 inference)
    2. Manipulate depth maps based on scene graph (programmatic geometric reasoning)

    The manipulated depth maps serve as ControlNet conditioning signals,
    telling the diffusion model WHERE objects should appear in 3D space.
    """

    def __init__(self, vram_manager: Optional[VRAMManager] = None):
        self.vram = vram_manager or VRAMManager()
        self.device = self.vram.device
        self.pipe = None

    def load_model(self):
        """Load Depth Anything V2 (vitb variant — best quality/speed ratio)."""
        from transformers import pipeline

        print(f"Loading Depth Anything V2 on {self.device}...")
        self.vram.snapshot("before_depth_model")

        self.pipe = pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Base-hf",
            device=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.vram.register("depth_anything_v2", self.pipe)
        self.vram.snapshot("after_depth_model")
        print("Depth Anything V2 loaded")

    def unload_model(self):
        """Free depth model from GPU memory."""
        self.pipe = None
        self.vram.unload("depth_anything_v2")
        self.vram.snapshot("after_depth_unload")

    def estimate_depth(self, image: Image.Image) -> np.ndarray:
        """Extract depth map from an image using Depth Anything V2.

        Args:
            image: Input PIL image (e.g., a reference dashcam photo)

        Returns:
            Normalized depth map as float32 numpy array, shape (H, W), range [0, 1].
            0 = nearest, 1 = farthest (following ControlNet convention).
        """
        if self.pipe is None:
            self.load_model()

        result = self.pipe(image)
        depth_pil = result["depth"]

        depth_array = np.array(depth_pil, dtype=np.float32)
        if depth_array.max() > 0:
            depth_array = depth_array / depth_array.max()

        return depth_array

    def manipulate_depth(
        self,
        base_depth: np.ndarray,
        objects: list[dict],
        image_width: int = 1024,
        image_height: int = 1024,
    ) -> np.ndarray:
        """Programmatically modify a depth map based on scene object positions.

        This is NOT just running a model — it's applying geometric reasoning
        to place objects at correct distances for ControlNet conditioning.

        Args:
            base_depth: Base depth map, shape (H, W), range [0, 1]
            objects: List of scene objects with spatial info. Each dict has:
                - type: "pedestrian", "vehicle", "cyclist", "guardrail", etc.
                - distance_m: Distance from ego vehicle in meters
                - lateral_position: -1.0 (far left) to 1.0 (far right)
                - width_fraction: Fraction of image width the object spans (0.0-1.0)
                - height_fraction: Fraction of image height (0.0-1.0)
            image_width: Output width
            image_height: Output height

        Returns:
            Modified depth map with objects placed at correct depths.
        """
        depth = base_depth.copy()
        if depth.shape != (image_height, image_width):
            depth = np.array(
                Image.fromarray((depth * 255).astype(np.uint8)).resize(
                    (image_width, image_height), Image.BILINEAR
                ),
                dtype=np.float32,
            ) / 255.0

        for obj in objects:
            depth = self._place_object_in_depth(depth, obj, image_width, image_height)

        # Apply Gaussian blur to the final depth map to prevent ControlNet
        # from seeing any remaining hard edges (prevents tiling artifacts)
        depth = gaussian_filter(depth, sigma=8.0)

        return np.clip(depth, 0.0, 1.0)

    def _place_object_in_depth(
        self,
        depth: np.ndarray,
        obj: dict,
        img_w: int,
        img_h: int,
    ) -> np.ndarray:
        """Place a single object into the depth map using a 2D Gaussian blob.

        Uses perspective projection for position/scale, then stamps a smooth
        Gaussian blob instead of a sharp rectangle. This prevents ControlNet
        from seeing hard edges that cause tiling artifacts.
        """
        distance_m = obj.get("distance_m", 30.0)
        lateral = obj.get("lateral_position", 0.0)
        obj_width_frac = obj.get("width_fraction", 0.05)
        obj_height_frac = obj.get("height_fraction", 0.15)

        # Convert distance to depth value (0=near, 1=far)
        depth_value = 1.0 - (1.0 / (1.0 + distance_m / 50.0))

        # Perspective projection: farther objects appear higher and smaller
        vanishing_y = 0.4
        scale_factor = 1.0 / (1.0 + distance_m / 10.0)
        center_y_frac = vanishing_y + (1.0 - vanishing_y) * scale_factor

        # X position from lateral offset
        center_x_frac = 0.5 + lateral * 0.4

        # Scale object size with distance (farther = smaller)
        size_scale = max(0.02, 1.0 / (1.0 + distance_m / 5.0))
        sigma_x = img_w * obj_width_frac * size_scale * 1.5
        sigma_y = img_h * obj_height_frac * size_scale * 1.5

        cx = int(center_x_frac * img_w)
        cy = int(center_y_frac * img_h)

        # Create 2D Gaussian blob over the full image (vectorized, fast)
        y_coords = np.arange(img_h)
        x_coords = np.arange(img_w)
        xx, yy = np.meshgrid(x_coords, y_coords)

        # Clamp sigma to avoid division by zero or tiny blobs
        sigma_x = max(sigma_x, 3.0)
        sigma_y = max(sigma_y, 3.0)

        gaussian = np.exp(
            -0.5 * (((xx - cx) / sigma_x) ** 2 + ((yy - cy) / sigma_y) ** 2)
        )

        # Blend: where Gaussian is strong, pull depth toward object_depth
        depth = depth * (1.0 - gaussian) + depth_value * gaussian

        return depth

    def create_base_depth(
        self, road_type: str = "highway", width: int = 1024, height: int = 1024
    ) -> np.ndarray:
        """Create a synthetic base depth map for a driving scene.

        When no reference image is available, generate a reasonable depth map
        based on road geometry. This encodes perspective: road surface recedes
        to a vanishing point near the top, objects on the sides are at
        intermediate depth.

        Args:
            road_type: "highway", "urban", "residential", "intersection"
            width: Output width
            height: Output height

        Returns:
            Synthetic depth map, shape (height, width), range [0, 1].
        """
        depth = np.zeros((height, width), dtype=np.float32)

        # Smooth gradient from top (far=1.0) to bottom (near=0.0)
        # No hard sky/road boundary — smooth transition prevents ControlNet artifacts
        for row in range(height):
            depth[row, :] = 1.0 - (row / height)

        # Road-type-specific adjustments (subtle, blended)
        if road_type in ("highway", "urban"):
            center_start = width // 4
            center_end = 3 * width // 4
            side_depth_offset = 0.03
            for row in range(height // 2, height):
                progress = (row - height // 2) / (height // 2)
                depth[row, :center_start] -= side_depth_offset * progress
                depth[row, center_end:] -= side_depth_offset * progress

        if road_type == "intersection":
            cross_y_start = int(height * 0.55)
            cross_y_end = int(height * 0.65)
            depth[cross_y_start:cross_y_end, :] *= 0.95

        # Smooth the base depth map
        depth = gaussian_filter(depth, sigma=12.0)

        return np.clip(depth, 0.0, 1.0)

    def depth_to_pil(self, depth: np.ndarray) -> Image.Image:
        """Convert depth array to PIL Image for ControlNet input."""
        depth_uint8 = (depth * 255).astype(np.uint8)
        return Image.fromarray(depth_uint8, mode="L").convert("RGB")


if __name__ == "__main__":
    print("Testing DepthGenerator...")
    gen = DepthGenerator()

    # Test 1: Create synthetic base depth
    base = gen.create_base_depth("highway")
    print(f"Base depth shape: {base.shape}, range: [{base.min():.2f}, {base.max():.2f}]")

    # Test 2: Manipulate depth with objects
    objects = [
        {"type": "pedestrian", "distance_m": 20.0, "lateral_position": 0.2, "width_fraction": 0.04, "height_fraction": 0.2},
        {"type": "vehicle", "distance_m": 40.0, "lateral_position": -0.1, "width_fraction": 0.1, "height_fraction": 0.12},
    ]
    manipulated = gen.manipulate_depth(base, objects)
    print(f"Manipulated depth range: [{manipulated.min():.2f}, {manipulated.max():.2f}]")

    # Test 3: Convert to PIL
    depth_img = gen.depth_to_pil(manipulated)
    depth_img.save("outputs/test_depth.png")
    print("Saved test depth map to outputs/test_depth.png")

    gen.vram.print_report()
