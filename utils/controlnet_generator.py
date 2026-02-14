import torch
from PIL import Image
from typing import Optional
from utils.vram_manager import VRAMManager
import warnings

warnings.filterwarnings("ignore")


class ControlNetGenerator:
    """Depth-conditioned image generation using ControlNet + SDXL + optional IP-Adapter.

    Multi-signal conditioning stack:
    - ControlNet: spatial layout from depth map (WHERE objects appear)
    - SDXL: photorealistic image generation (WHAT things look like)
    - IP-Adapter (arxiv 2308.06721): reference image style transfer (HOW it looks)
      Transfers dashcam-specific qualities (lens distortion, color cast, exposure)
      from real reference images. Only 22M params (+2GB VRAM).

    VRAM: ~12GB without IP-Adapter, ~14GB with IP-Adapter
    """

    def __init__(self, vram_manager: Optional[VRAMManager] = None, use_ip_adapter: bool = False):
        self.vram = vram_manager or VRAMManager()
        self.device = self.vram.device
        self.pipe = None
        self.use_ip_adapter = use_ip_adapter
        self._ip_adapter_loaded = False

        self.negative_prompt = (
            "cartoon, anime, drawing, painting, illustration, "
            "blurry, low quality, distorted, deformed, "
            "text, watermark, signature, extra limbs, "
            "unrealistic, 3d render, cgi"
        )

    def load_model(self):
        """Load ControlNet depth model + SDXL base."""
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

        print(f"Loading ControlNet + SDXL on {self.device}...")
        self.vram.snapshot("before_controlnet_load")

        dtype = torch.float16 if self.device == "cuda" else torch.float32

        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            torch_dtype=dtype,
            use_safetensors=True,
        )

        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if self.device == "cuda" else None,
        )

        self.pipe = self.pipe.to(self.device)

        # IP-Adapter: reference image style conditioning (arxiv 2308.06721)
        if self.use_ip_adapter:
            print("Loading IP-Adapter for style conditioning...")
            self.pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name="ip-adapter_sdxl.bin",
            )
            self.pipe.set_ip_adapter_scale(0.35)
            self._ip_adapter_loaded = True
            print("IP-Adapter loaded (scale=0.35)")

        self.vram.register("controlnet_sdxl", self.pipe)
        self.vram.snapshot("after_controlnet_load")
        print("ControlNet + SDXL loaded")

    def unload_model(self):
        """Free ControlNet + SDXL from GPU memory."""
        self.pipe = None
        self.vram.unload("controlnet_sdxl")
        self.vram.snapshot("after_controlnet_unload")

    def generate(
        self,
        prompt: str,
        depth_image: Image.Image,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        controlnet_conditioning_scale: float = 0.25,
        ip_adapter_image: Optional[Image.Image] = None,
    ) -> Image.Image:
        """Generate a depth-conditioned dashcam image.

        Args:
            prompt: Text description of the scene
            depth_image: RGB depth map image (ControlNet conditioning signal)
            height: Output image height
            width: Output image width
            num_inference_steps: Diffusion denoising steps (more = better quality, slower)
            guidance_scale: How strictly to follow the text prompt (7.0 is balanced)
            controlnet_conditioning_scale: How strongly depth constrains generation.
                0.0 = ignore depth (like raw SDXL), 1.0 = strict depth adherence.
                0.25 = gentle guidance (prevents tiling from programmatic depth maps).
            ip_adapter_image: Optional reference dashcam image for style transfer.
                Transfers lighting, color grading, lens characteristics from a real
                dashcam photo. Requires use_ip_adapter=True in constructor.

        Returns:
            Generated PIL Image
        """
        if self.pipe is None:
            self.load_model()

        # Ensure depth image matches target size
        if depth_image.size != (width, height):
            depth_image = depth_image.resize((width, height), Image.BILINEAR)

        print(f"Generating (ControlNet): {prompt[:80]}...")
        self.vram.snapshot("before_controlnet_generate")

        pipe_kwargs = dict(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            image=depth_image,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )

        # Add IP-Adapter reference image if available
        if ip_adapter_image is not None and self._ip_adapter_loaded:
            pipe_kwargs["ip_adapter_image"] = ip_adapter_image

        image = self.pipe(**pipe_kwargs).images[0]

        self.vram.snapshot("after_controlnet_generate")
        return image

    def generate_from_scenario(
        self,
        scenario,
        depth_image: Image.Image,
        controlnet_conditioning_scale: float = 0.25,
    ) -> Image.Image:
        """Generate a driving scene from a CrashScenario + depth map.

        Args:
            scenario: CrashScenario with image_prompt field
            depth_image: Depth conditioning image (from DepthGenerator)
            controlnet_conditioning_scale: Depth influence strength

        Returns:
            Generated dashcam image
        """
        prompt = scenario.image_prompt
        if not prompt:
            prompt = (
                f"dashcam photo, point of view from inside a car, photorealistic, RAW photo, "
                f"sharp focus, {scenario.weather} weather, {scenario.lighting} lighting, "
                f"{scenario.road_type.replace('_', ' ')} road, "
                f"{scenario.incident_type.replace('_', ' ')}"
            )
        return self.generate(
            prompt=prompt,
            depth_image=depth_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )

    def generate_crash_keyframes(
        self,
        scenario,
        depth_image: Image.Image,
        controlnet_conditioning_scale: float = 0.25,
    ) -> dict[str, Image.Image]:
        """Generate multiple keyframes for a crash scenario timeline.

        Attempts to generate: pre-crash (safe), approaching hazard, and
        crash moment. Crash moment quality will vary — this is documented
        as part of the portfolio's honest evaluation.

        Args:
            scenario: CrashScenario
            depth_image: Base depth conditioning
            controlnet_conditioning_scale: Depth influence strength

        Returns:
            Dict mapping timepoint names to generated images.
        """
        base_prompt_prefix = (
            "dashcam photo, point of view from inside a car, photorealistic, RAW photo, "
            "sharp focus, "
        )

        weather_desc = f"{scenario.weather} weather, {scenario.lighting} lighting, "
        road_desc = f"{scenario.road_type.replace('_', ' ')} road, {scenario.road_condition} road surface, "

        # Build temporal prompts
        prompts = {}
        incident = scenario.incident_type.replace("_", " ")
        other = scenario.other_agent_type or "obstacle"
        other_action = scenario.other_agent_action or ""

        prompts["3s_before"] = (
            base_prompt_prefix + weather_desc + road_desc
            + f"{other} visible in the distance {other_action}, normal driving conditions, "
            + "calm scene, no immediate danger"
        )

        prompts["1s_before"] = (
            base_prompt_prefix + weather_desc + road_desc
            + f"{other} dangerously close {other_action}, imminent {incident}, "
            + "tense moment, brake lights visible, motion blur"
        )

        prompts["impact"] = (
            base_prompt_prefix + weather_desc + road_desc
            + f"moment of {incident} impact, {other} collision, "
            + "vehicle damage visible, debris, dramatic angle, motion blur, "
            + "shattered glass, crumpled metal, airbag deployment"
        )

        keyframes = {}
        for timepoint, prompt in prompts.items():
            print(f"  Generating keyframe: {timepoint}")
            keyframes[timepoint] = self.generate(
                prompt=prompt,
                depth_image=depth_image,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
            )

        return keyframes


if __name__ == "__main__":
    from utils.depth_generator import DepthGenerator

    vram = VRAMManager()

    # Step 1: Generate synthetic depth map
    depth_gen = DepthGenerator(vram_manager=vram)
    base_depth = depth_gen.create_base_depth("highway")
    objects = [
        {"type": "vehicle", "distance_m": 30, "lateral_position": 0.0, "width_fraction": 0.1, "height_fraction": 0.1},
    ]
    manipulated = depth_gen.manipulate_depth(base_depth, objects)
    depth_img = depth_gen.depth_to_pil(manipulated)
    depth_img.save("outputs/test_depth_controlnet.png")
    print("Saved depth map")

    # Step 2: Generate conditioned image (requires GPU — run on Colab)
    cn_gen = ControlNetGenerator(vram_manager=vram)
    image = cn_gen.generate(
        prompt=(
            "dashcam photo, point of view from inside a car, photorealistic, RAW photo, "
            "sharp focus, highway in rain, wet asphalt, vehicle ahead braking, "
            "red brake lights visible, spray of water, motion blur"
        ),
        depth_image=depth_img,
    )
    image.save("outputs/test_controlnet.png")
    print("Saved ControlNet image")

    # Step 3: Clean up and report
    cn_gen.unload_model()
    vram.print_report()
