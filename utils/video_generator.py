import torch
import gc
from PIL import Image
from typing import Optional
from utils.vram_manager import VRAMManager
import warnings

warnings.filterwarnings("ignore")


class VideoGenerator:
    """Dashcam video generation from a generated image.

    Uses Stable Video Diffusion (SVD) — confirmed working on free Colab T4.
    Takes a dashcam image (from ControlNet pipeline) and animates it
    into a 2-4 second video clip.

    Why SVD instead of Wan2.1:
    - Wan2.1 1.3B crashes Colab free tier's 12GB system RAM
    - SVD is image-to-video (animates our ControlNet output directly)
    - SVD is ~6GB VRAM with optimizations (fits T4 with room to spare)
    - decode_chunk_size=1 keeps memory low by decoding one frame at a time

    VRAM: ~6-8GB on T4 with optimizations
    """

    def __init__(self, vram_manager: Optional[VRAMManager] = None):
        self.vram = vram_manager or VRAMManager()
        self.device = self.vram.device
        self.pipe = None

    def load_model(self):
        """Load Stable Video Diffusion (SVD-XT) for image-to-video."""
        from diffusers import StableVideoDiffusionPipeline

        print(f"Loading SVD on {self.device}...")
        self.vram.snapshot("before_video_model")

        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16,
            variant="fp16",
        )

        # Sequential CPU offload: only one layer on GPU at a time
        # Much more memory efficient than model_cpu_offload
        self.pipe.enable_sequential_cpu_offload()

        self.vram.register("svd", self.pipe)
        self.vram.snapshot("after_video_model")
        print("SVD loaded")

    def unload_model(self):
        """Free SVD from memory."""
        self.pipe = None
        self.vram.unload("svd")
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        self.vram.snapshot("after_video_unload")

    def generate(
        self,
        image: Image.Image,
        num_frames: int = 25,
        num_inference_steps: int = 25,
        decode_chunk_size: int = 1,
        motion_bucket_id: int = 127,
        fps: int = 7,
    ) -> list:
        """Generate video from a dashcam image using SVD.

        Args:
            image: Input dashcam image to animate
            num_frames: Number of output frames (25 = ~3.5s at 7fps)
            num_inference_steps: Denoising steps (25 is good balance)
            decode_chunk_size: Frames decoded at once (1 = lowest memory)
            motion_bucket_id: Motion intensity (127 = moderate, higher = more motion)
            fps: Frames per second for conditioning (7 is SVD default)

        Returns:
            List of PIL Images (video frames)
        """
        if self.pipe is None:
            self.load_model()

        # SVD expects 1024x576 input
        image = image.resize((1024, 576), Image.LANCZOS)

        print(f"Generating video ({num_frames} frames)...")
        self.vram.snapshot("before_video_generate")

        frames = self.pipe(
            image,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            decode_chunk_size=decode_chunk_size,
            motion_bucket_id=motion_bucket_id,
            fps=fps,
        ).frames[0]

        self.vram.snapshot("after_video_generate")
        print(f"Generated {len(frames)} frames")
        return frames

    def generate_from_scenario(
        self,
        scenario,
        source_image: Image.Image,
    ) -> list:
        """Generate video from a CrashScenario + source image.

        Args:
            scenario: CrashScenario (used for logging/metadata)
            source_image: Dashcam image to animate (from ControlNet)

        Returns:
            List of PIL Images (video frames)
        """
        print(f"Generating video for: {scenario.incident_type}")
        return self.generate(image=source_image)

    @staticmethod
    def export_video(frames: list, output_path: str, fps: int = 7):
        """Save video frames to MP4 file.

        Args:
            frames: List of PIL Images
            output_path: Output .mp4 path
            fps: Frames per second (7 is SVD default)
        """
        from diffusers.utils import export_to_video
        export_to_video(frames, output_path, fps=fps)
        print(f"Saved video: {output_path} ({len(frames)} frames @ {fps}fps)")


if __name__ == "__main__":
    vram = VRAMManager()
    gen = VideoGenerator(vram_manager=vram)

    # Test with a dummy image (replace with real generated image on Colab)
    test_img = Image.new("RGB", (1024, 576), color=(100, 120, 100))
    frames = gen.generate(test_img, num_frames=14)  # 14 frames = ~2s
    gen.export_video(frames, "outputs/test_video.mp4")

    gen.unload_model()
    vram.print_report()
