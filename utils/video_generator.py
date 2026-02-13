import torch
import numpy as np
from PIL import Image
from typing import Optional
from utils.vram_manager import VRAMManager
import warnings

warnings.filterwarnings("ignore")


class VideoGenerator:
    """Dashcam video generation using Wan2.1/2.2 video diffusion models.

    Two modes:
    1. Text-to-Video (T2V): Wan2.1 1.3B — 8GB VRAM, fits T4 easily
    2. Image-to-Video (I2V): Wan2.2 5B — ~12GB VRAM, tight on T4

    T2V generates video directly from the crash scenario text prompt.
    I2V takes a generated dashcam image (from ControlNet) and animates it.

    IMPORTANT: T4 GPUs do NOT support bfloat16. We use float16 instead
    and keep the VAE in float32 for decoding quality.
    """

    def __init__(self, vram_manager: Optional[VRAMManager] = None, mode: str = "t2v"):
        """
        Args:
            vram_manager: Shared VRAMManager instance
            mode: "t2v" for text-to-video (1.3B, recommended for T4)
                  "i2v" for image-to-video (5B, tight on T4)
        """
        self.vram = vram_manager or VRAMManager()
        self.device = self.vram.device
        self.mode = mode
        self.pipe = None

    def load_model(self):
        """Load the appropriate Wan video model."""
        if self.mode == "t2v":
            self._load_t2v()
        else:
            self._load_i2v()

    def _load_t2v(self):
        """Load Wan2.1 1.3B text-to-video model (~8GB VRAM)."""
        from diffusers import WanPipeline, AutoencoderKLWan
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        print(f"Loading Wan2.1 T2V 1.3B on {self.device}...")
        self.vram.snapshot("before_video_model")

        # VAE must stay in float32 for quality
        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32
        )

        # Use float16 (NOT bfloat16 — T4 doesn't support bf16)
        self.pipe = WanPipeline.from_pretrained(
            model_id, vae=vae, torch_dtype=torch.float16
        )

        # flow_shift=3.0 for 480p generation
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config, flow_shift=3.0
        )

        # Use CPU offload to minimize VRAM usage
        self.pipe.enable_model_cpu_offload()

        self.vram.register("wan_t2v", self.pipe)
        self.vram.snapshot("after_video_model")
        print("Wan2.1 T2V loaded")

    def _load_i2v(self):
        """Load Wan2.2 5B image-to-video model (~12GB VRAM)."""
        from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        from transformers import CLIPVisionModel

        model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        print(f"Loading Wan2.2 I2V 5B on {self.device}...")
        self.vram.snapshot("before_video_model")

        image_encoder = CLIPVisionModel.from_pretrained(
            model_id, subfolder="image_encoder", torch_dtype=torch.float32
        )
        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32
        )

        self.pipe = WanImageToVideoPipeline.from_pretrained(
            model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.float16
        )

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config, flow_shift=3.0
        )

        self.pipe.enable_model_cpu_offload()

        self.vram.register("wan_i2v", self.pipe)
        self.vram.snapshot("after_video_model")
        print("Wan2.2 I2V loaded")

    def unload_model(self):
        """Free video model from GPU memory."""
        self.pipe = None
        model_name = "wan_t2v" if self.mode == "t2v" else "wan_i2v"
        self.vram.unload(model_name)
        self.vram.snapshot("after_video_unload")

    def generate_t2v(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted, static, frozen",
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
    ) -> list:
        """Generate video from text prompt using Wan2.1 T2V.

        Args:
            prompt: Scene description
            negative_prompt: What to avoid
            height: Video height (480 recommended for T4)
            width: Video width (832 for ~16:9 aspect)
            num_frames: Number of frames (81 = ~5s at 16fps)
            num_inference_steps: Denoising steps
            guidance_scale: Prompt adherence strength

        Returns:
            List of PIL Images (video frames)
        """
        if self.pipe is None:
            self.load_model()

        print(f"Generating T2V: {prompt[:80]}...")
        self.vram.snapshot("before_video_generate")

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        self.vram.snapshot("after_video_generate")
        return output.frames[0]

    def generate_i2v(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted, static, frozen",
        num_frames: int = 81,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
    ) -> list:
        """Generate video from an image using Wan2.2 I2V.

        Args:
            image: Input dashcam image to animate
            prompt: Motion/scene description
            negative_prompt: What to avoid
            num_frames: Number of frames
            num_inference_steps: Denoising steps
            guidance_scale: Prompt adherence strength

        Returns:
            List of PIL Images (video frames)
        """
        if self.pipe is None:
            self.load_model()

        # Resize image to fit VAE constraints
        max_area = 480 * 832
        aspect_ratio = image.height / image.width
        mod_value = self.pipe.vae_scale_factor_spatial * self.pipe.transformer.config.patch_size[1]
        h = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        w = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((w, h))

        print(f"Generating I2V ({w}x{h}): {prompt[:80]}...")
        self.vram.snapshot("before_video_generate")

        output = self.pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=h,
            width=w,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        self.vram.snapshot("after_video_generate")
        return output.frames[0]

    def generate_from_scenario(
        self,
        scenario,
        source_image: Optional[Image.Image] = None,
    ) -> list:
        """Generate video from a CrashScenario.

        If a source_image is provided AND mode is "i2v", animates that image.
        Otherwise, generates video from the scenario's text prompt.

        Args:
            scenario: CrashScenario with image_prompt
            source_image: Optional dashcam image to animate (for I2V mode)

        Returns:
            List of PIL Images (video frames)
        """
        # Build a video-oriented prompt (more action/motion focused)
        base_prompt = scenario.image_prompt or ""
        video_prompt = (
            f"dashcam video footage, driving forward, {base_prompt}, "
            f"realistic motion, camera mounted on dashboard, "
            f"approaching {scenario.incident_type.replace('_', ' ')} scenario"
        )

        if self.mode == "i2v" and source_image is not None:
            return self.generate_i2v(image=source_image, prompt=video_prompt)
        else:
            return self.generate_t2v(prompt=video_prompt)

    @staticmethod
    def export_video(frames: list, output_path: str, fps: int = 16):
        """Save video frames to MP4 file.

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

    # T2V mode (recommended for T4)
    gen = VideoGenerator(vram_manager=vram, mode="t2v")

    frames = gen.generate_t2v(
        prompt=(
            "dashcam video footage, driving forward on a wet highway in heavy rain, "
            "point of view from inside a car, photorealistic, "
            "windshield wipers moving, water spray from road, "
            "vehicle ahead with red brake lights visible, slowing down"
        ),
        num_frames=81,
    )

    gen.export_video(frames, "outputs/test_video.mp4")

    gen.unload_model()
    vram.print_report()
