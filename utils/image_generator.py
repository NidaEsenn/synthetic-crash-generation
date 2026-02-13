import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import warnings

warnings.filterwarnings("ignore")


class BasicImageGenerator:
    """Text-to-image generation using Stable Diffusion XL"""

    def __init__(self, device: str = None):
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"Loading SDXL on {self.device}...")

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            use_safetensors=True,
            variant="fp16" if self.device != "cpu" else None,
        )

        self.negative_prompt = (
            "cartoon, anime, drawing, painting, illustration, "
            "blurry, low quality, distorted, deformed, "
            "text, watermark, signature, extra limbs, "
            "unrealistic, 3d render, cgi"
        )

        self.pipe = self.pipe.to(self.device)
        self.pipe.enable_attention_slicing()

        print("SDXL loaded")

    def generate(self, prompt: str, height: int = 1024, width: int = 1024) -> Image.Image:
        """Generate image from text prompt"""
        print(f"Generating: {prompt[:80]}...")

        image = self.pipe(
            prompt,
            negative_prompt=self.negative_prompt,
            height=height,
            width=width,
            num_inference_steps=30,
            guidance_scale=7.0,
        ).images[0]

        return image

    def generate_from_scenario(self, scenario) -> Image.Image:
        """Generate driving scene from CrashScenario using LLM-generated prompt"""
        if scenario.image_prompt:
            return self.generate(scenario.image_prompt)

        # Fallback if image_prompt is empty (shouldn't happen with updated parser)
        prompt = (
            f"dashcam photo, point of view from inside a car, photorealistic, RAW photo, "
            f"sharp focus, {scenario.weather} weather, {scenario.lighting} lighting, "
            f"{scenario.road_type.replace('_', ' ')} road, {scenario.incident_type.replace('_', ' ')}"
        )
        return self.generate(prompt)


# Test
if __name__ == "__main__":
    from utils.parser import LLMParser

    gen = BasicImageGenerator()
    parser = LLMParser()

    scenario = parser.parse("Car going 50mph on highway in rain")
    image = gen.generate_from_scenario(scenario)
    image.save("outputs/test_day1.png")
    print("Saved to outputs/test_day1.png")
