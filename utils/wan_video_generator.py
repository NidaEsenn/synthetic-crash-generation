"""Wan2.1 Image-to-Video generator — high-quality fallback video generation.

Uses Wan-AI/Wan2.1-I2V-14B-480P for image-to-video generation at 480P.
This is the general-purpose high-quality video backend, used when:
- Ctrl-Crash is not available or fails
- Non-crash driving scenarios (normal driving, weather conditions)
- Higher quality is needed than SVD

Why 480P not 720P: The 14B model at 720P needs ~40-45GB VRAM without
optimizations, which exceeds A100 40GB. 480P with model_cpu_offload
fits comfortably at ~28-32GB.

VRAM: ~28-32GB on A100 with model_cpu_offload
"""

import torch
import gc
from PIL import Image
from typing import Optional
from utils.vram_manager import VRAMManager
import warnings

warnings.filterwarnings("ignore")


class WanVideoGenerator:
    """High-quality image-to-video generation using Wan2.1 14B.

    Generates 480P video (832x480) from a dashcam image + text prompt.
    Uses model_cpu_offload to fit the 14B parameter model within A100 40GB.
    """

    def __init__(self, vram_manager: Optional[VRAMManager] = None):
        self.vram = vram_manager or VRAMManager()
        self.device = self.vram.device
        self.pipe = None

    def load_model(self):
        """Load Wan2.1 I2V 14B 480P model."""
        from diffusers import WanImageToVideoPipeline
        from diffusers.utils import export_to_video

        print(f"Loading Wan2.1 I2V 14B on {self.device}...")
        self.vram.snapshot("before_wan_load")

        self.pipe = WanImageToVideoPipeline.from_pretrained(
            "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            torch_dtype=torch.bfloat16,
        )

        # model_cpu_offload: moves layers to GPU only when needed
        # This allows the 14B model to fit in 40GB by keeping most
        # parameters on CPU and shuttling them to GPU layer-by-layer
        self.pipe.enable_model_cpu_offload()

        self.vram.register("wan2.1_i2v", self.pipe)
        self.vram.snapshot("after_wan_load")
        print("Wan2.1 I2V 14B loaded")

    def unload_model(self):
        """Free Wan2.1 from memory."""
        self.pipe = None
        self.vram.unload("wan2.1_i2v")
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        self.vram.snapshot("after_wan_unload")

    def generate(
        self,
        image: Image.Image,
        prompt: str = "",
        negative_prompt: str = "",
        num_frames: int = 81,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        width: int = 832,
        height: int = 480,
    ) -> list[Image.Image]:
        """Generate video from image + text prompt.

        Args:
            image: Source dashcam image to animate
            prompt: Text description guiding the video motion
            negative_prompt: What to avoid
            num_frames: Output frames (81 = ~5s at 16fps)
            num_inference_steps: Denoising steps
            guidance_scale: Prompt adherence strength
            width: Output width (832 for 480P)
            height: Output height (480 for 480P)

        Returns:
            List of PIL Images (video frames)
        """
        if self.pipe is None:
            self.load_model()

        # Resize to Wan2.1 expected dimensions
        image = image.resize((width, height), Image.LANCZOS)

        if not negative_prompt:
            negative_prompt = (
                "Bright tones, overexposed, static, blurry details, "
                "subtitles, style, works, paintings, images, static, "
                "overall gray, worst quality, low quality, JPEG artifacts, "
                "ugly, incomplete, extra fingers, poorly drawn hands, "
                "poorly drawn faces, deformed, disfigured, misshapen limbs, "
                "fused fingers, still picture, messy background, "
                "three legs, many people in the background, walking backwards"
            )

        print(f"Generating video ({num_frames} frames at {width}x{height})...")
        self.vram.snapshot("before_wan_generate")

        output = self.pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
        )
        frames = output.frames[0]

        self.vram.snapshot("after_wan_generate")
        print(f"Generated {len(frames)} frames")
        return frames

    def generate_from_scenario(
        self,
        scenario,
        source_image: Image.Image,
    ) -> list[Image.Image]:
        """Generate video from a CrashScenario + source image.

        Constructs a motion-focused prompt from the scenario's temporal
        description for better video dynamics.

        Args:
            scenario: CrashScenario object
            source_image: Dashcam image from ControlNet pipeline

        Returns:
            List of PIL Images (video frames)
        """
        # Build motion-focused prompt from scenario
        prompt_parts = [
            "dashcam video footage",
            f"{scenario.weather} weather",
            f"{scenario.lighting} lighting",
            f"{scenario.road_type.replace('_', ' ')} road",
        ]

        if scenario.temporal_description:
            prompt_parts.append(scenario.temporal_description[:200])
        else:
            prompt_parts.append(
                f"vehicle {scenario.ego_action.replace('_', ' ')}, "
                f"{scenario.incident_type.replace('_', ' ')} incident"
            )

        prompt = ", ".join(prompt_parts)
        print(f"Wan2.1 prompt: {prompt[:100]}...")

        return self.generate(image=source_image, prompt=prompt)

    @staticmethod
    def export_video(frames: list, output_path: str, fps: int = 16):
        """Save video frames to MP4.

        Args:
            frames: List of PIL Images
            output_path: Output .mp4 path
            fps: Frames per second (16 is Wan2.1 default)
        """
        from diffusers.utils import export_to_video
        export_to_video(frames, output_path, fps=fps)
        print(f"Saved video: {output_path} ({len(frames)} frames @ {fps}fps)")


if __name__ == "__main__":
    vram = VRAMManager()
    gen = WanVideoGenerator(vram_manager=vram)

    # Test with dummy image (replace with real generated image on HPC)
    test_img = Image.new("RGB", (832, 480), color=(100, 120, 100))
    frames = gen.generate(
        test_img,
        prompt="dashcam footage, car driving on highway in rain, wet road",
        num_frames=41,  # shorter test: ~2.5s
    )
    gen.export_video(frames, "outputs/test_wan_video.mp4")
    gen.unload_model()
    vram.print_report()
