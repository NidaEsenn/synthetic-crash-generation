"""Prepare dashcam crash dataset for SDXL LoRA fine-tuning.

Four modes:
1. DOTA: Download traffic anomaly dashcam frames from DoTA dataset (RECOMMENDED)
2. NEXAR: Download collision dashcam videos from Nexar dataset, extract crash-moment frames
3. BDD100K: Download general driving images from BDD100K
4. LOCAL: Use your own folder of dashcam images

Then auto-caption each image using BLIP and create
a metadata.jsonl file in HuggingFace Dataset format.

Output structure:
    data/dashcam_lora/
    ├── img_0001.png
    ├── img_0002.png
    ├── ...
    └── metadata.jsonl   ← {"file_name": "img_0001.png", "text": "dashcam photo of..."}

Usage:
    # Nexar crash videos (recommended):
    python scripts/prepare_dashcam_data.py --source nexar --num_images 500

    # BDD100K general driving:
    python scripts/prepare_dashcam_data.py --source bdd100k --num_images 500

    # From Jupyter:
    from prepare_dashcam_data import prepare_dataset
    prepare_dataset(source="nexar", num_images=500)
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image


def download_dota_frames(output_dir: str, num_images: int = 500) -> list[str]:
    """Download traffic anomaly dashcam frames from DoTA dataset.

    DoTA (Detection of Traffic Anomaly) contains 4,677 real dashcam videos
    with 18 anomaly categories (collision, pedestrian, vehicle turning, etc.)
    Pre-extracted JPG frames are available on Google Drive.

    The 55GB dataset is split into 5×10GB + 1×5GB archives.
    We download the smallest split and extract frames from it.

    GitHub: https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly

    Args:
        output_dir: Directory to save extracted frames
        num_images: Target number of frames to extract

    Returns:
        List of saved image paths
    """
    import subprocess
    import tarfile
    import random

    os.makedirs(output_dir, exist_ok=True)

    # Google Drive folder with split archives
    gdrive_folder = "1_WzhwZC2NIpzZIpX7YCvapq66rtBc67n"
    temp_dir = os.path.join(output_dir, "_temp_dota")
    os.makedirs(temp_dir, exist_ok=True)

    print("Downloading DoTA frames from Google Drive...")
    print("  (This may take 5-10 minutes for the first archive)")

    # Use gdown to download from the split files folder
    try:
        subprocess.run(
            ["pip", "install", "--user", "-q", "gdown"],
            capture_output=True,
        )
        import gdown

        # Download the folder listing and get the smallest file
        # Direct link to the split folder
        url = f"https://drive.google.com/drive/folders/{gdrive_folder}"
        gdown.download_folder(url, output=temp_dir, quiet=False, remaining_ok=True)

    except Exception as e:
        print(f"\n  gdown folder download failed: {e}")
        print("  Trying single file download...")

        # Fallback: download the main 55GB file (will be slow)
        # Use the known file ID
        file_id = "1RQp4hOP9X7TW6S3_vbqZvFSkrbbFzrRj"
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            os.path.join(temp_dir, "dota_frames.tar.gz"),
            quiet=False,
        )

    # Find and extract archive(s)
    print("Extracting frames...")
    all_frames = []
    archives = sorted(Path(temp_dir).glob("*.tar.gz")) + sorted(Path(temp_dir).glob("*.zip"))

    if not archives:
        # Maybe frames were extracted directly
        archives = []
        all_frames = sorted(Path(temp_dir).rglob("*.jpg")) + sorted(Path(temp_dir).rglob("*.png"))

    for archive in archives:
        print(f"  Extracting {archive.name}...")
        try:
            if str(archive).endswith(".tar.gz"):
                with tarfile.open(archive, "r:gz") as tar:
                    tar.extractall(path=temp_dir)
            elif str(archive).endswith(".zip"):
                import zipfile
                with zipfile.ZipFile(archive, "r") as z:
                    z.extractall(temp_dir)
        except Exception as e:
            print(f"  Warning: extraction error: {e}")
            continue

    # Collect all extracted JPG/PNG frames
    if not all_frames:
        all_frames = sorted(Path(temp_dir).rglob("*.jpg")) + sorted(Path(temp_dir).rglob("*.png"))

    print(f"  Found {len(all_frames)} total frames")

    if len(all_frames) == 0:
        raise RuntimeError("No frames found after extraction. Check DoTA download.")

    # Sample randomly for diversity
    if len(all_frames) > num_images:
        random.seed(42)
        selected = random.sample(all_frames, num_images)
    else:
        selected = all_frames[:num_images]

    # Process and save
    saved_paths = []
    for i, frame_path in enumerate(selected):
        img = Image.open(frame_path).convert("RGB")
        img = center_crop_resize(img, 1024)

        filename = f"img_{i:04d}.png"
        path = os.path.join(output_dir, filename)
        img.save(path, "JPEG", quality=95)
        saved_paths.append(path)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(selected)} frames")

    print(f"Saved {len(saved_paths)} DoTA frames to {output_dir}")
    return saved_paths


def download_nexar_crash_frames(output_dir: str, num_images: int = 500) -> list[str]:
    """Extract crash-moment frames from Nexar collision prediction videos.

    Nexar dataset (arxiv 2503.03848) contains 1500 real dashcam videos:
    - 750 collision/near-miss videos with time_of_event timestamps
    - 750 normal driving videos
    - 1280x720, 30fps, ~40 seconds each

    For each collision video, we extract multiple frames around the
    crash moment (before, during, after) to capture the full progression.

    Args:
        output_dir: Directory to save extracted frames
        num_images: Target number of frames to extract

    Returns:
        List of saved image paths
    """
    from datasets import load_dataset
    import numpy as np

    print(f"Loading Nexar Collision Prediction dataset...")

    ds = load_dataset(
        "nexar-ai/nexar_collision_prediction",
        split="train",
        streaming=True,
    )

    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    img_counter = 0

    # We extract frames from collision videos (label=1)
    # For each video: 5 frames around the event time
    #   -3s, -1s, event, +1s, +3s
    frames_per_video = 5
    max_videos = (num_images // frames_per_video) + 1

    video_count = 0

    for sample in ds:
        if img_counter >= num_images:
            break

        # Only use collision videos (label=1) — these have crash scenes
        label = sample.get("label")
        if label != 1:
            continue

        video_count += 1
        time_of_event = sample.get("time_of_event", 20.0)
        weather = sample.get("weather", "unknown")
        scene = sample.get("scene", "unknown")
        light = sample.get("light_conditions", "unknown")

        try:
            video = sample["video"]

            # video is a list of PIL Images (frames decoded by datasets)
            total_frames = len(video)
            fps = 30  # Nexar videos are 30fps

            event_frame = int(time_of_event * fps)
            event_frame = min(event_frame, total_frames - 1)

            # Extract frames at: -3s, -1s, event, +1s, +3s
            offsets_seconds = [-3, -1, 0, 1, 3]
            for offset in offsets_seconds:
                if img_counter >= num_images:
                    break

                frame_idx = event_frame + int(offset * fps)
                frame_idx = max(0, min(frame_idx, total_frames - 1))

                frame = video[frame_idx]
                if not isinstance(frame, Image.Image):
                    frame = Image.fromarray(frame)

                if frame.mode != "RGB":
                    frame = frame.convert("RGB")

                frame = center_crop_resize(frame, 1024)

                filename = f"img_{img_counter:04d}.png"
                path = os.path.join(output_dir, filename)
                frame.save(path, "JPEG", quality=95)
                saved_paths.append(path)
                img_counter += 1

        except Exception as e:
            print(f"  Skipping video {video_count}: {e}")
            continue

        if video_count % 10 == 0:
            print(f"  Processed {video_count} videos, extracted {img_counter} frames")

    print(f"Extracted {len(saved_paths)} frames from {video_count} Nexar collision videos")
    return saved_paths


def download_bdd100k_subset(output_dir: str, num_images: int = 500) -> list[str]:
    """Download a subset of BDD100K dashcam images from HuggingFace.

    BDD100K has general driving scenes — good for learning dashcam style
    but doesn't contain crash-specific content.
    """
    from datasets import load_dataset

    print(f"Loading BDD100K from HuggingFace (first {num_images} images)...")

    ds = load_dataset(
        "dgural/bdd100k",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    for i, sample in enumerate(ds):
        if i >= num_images:
            break

        img = sample["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")

        img = center_crop_resize(img, 1024)

        filename = f"img_{i:04d}.png"
        path = os.path.join(output_dir, filename)
        img.save(path, "JPEG", quality=95)
        saved_paths.append(path)

        if (i + 1) % 50 == 0:
            print(f"  Saved {i + 1}/{num_images} images")

    print(f"Downloaded {len(saved_paths)} BDD100K images to {output_dir}")
    return saved_paths


def load_local_images(input_dir: str, output_dir: str, num_images: int = 500) -> list[str]:
    """Load and preprocess dashcam images from a local folder."""
    os.makedirs(output_dir, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted([
        f for f in Path(input_dir).iterdir()
        if f.suffix.lower() in extensions
    ])[:num_images]

    print(f"Found {len(image_files)} images in {input_dir}")

    saved_paths = []
    for i, img_path in enumerate(image_files):
        img = Image.open(img_path).convert("RGB")
        img = center_crop_resize(img, 1024)

        filename = f"img_{i:04d}.png"
        path = os.path.join(output_dir, filename)
        img.save(path)
        saved_paths.append(path)

    print(f"Processed {len(saved_paths)} images to {output_dir}")
    return saved_paths


def center_crop_resize(img: Image.Image, size: int) -> Image.Image:
    """Center-crop to square then resize to target size.

    Dashcam images are typically 1280x720 (16:9).
    We crop the center 720x720 square then resize to 1024x1024.
    """
    w, h = img.size
    crop_size = min(w, h)
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    img = img.crop((left, top, left + crop_size, top + crop_size))
    img = img.resize((size, size), Image.LANCZOS)
    return img


def auto_caption_blip(image_paths: list[str], device: str = "cuda") -> list[str]:
    """Generate captions for images using BLIP.

    BLIP is a lightweight vision-language model (~1GB VRAM).
    It generates natural language descriptions of images.

    We prefix all captions with "dashcam photo, " to anchor
    the LoRA to the dashcam domain.

    Args:
        image_paths: List of image file paths
        device: "cuda" or "cpu"

    Returns:
        List of caption strings
    """
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch

    print("Loading BLIP for auto-captioning...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    captions = []
    batch_size = 8

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]

        inputs = processor(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=3,
            )

        batch_captions = processor.batch_decode(outputs, skip_special_tokens=True)

        for caption in batch_captions:
            full_caption = f"dashcam photo, point of view from inside a car, {caption}"
            captions.append(full_caption)

        if (i + batch_size) % 50 == 0:
            print(f"  Captioned {min(i + batch_size, len(image_paths))}/{len(image_paths)}")

    # Clean up BLIP from GPU
    del model, processor
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"Generated {len(captions)} captions")
    return captions


def create_metadata(output_dir: str, image_paths: list[str], captions: list[str]):
    """Create metadata.jsonl for HuggingFace Dataset format.

    Each line: {"file_name": "img_0001.png", "text": "dashcam photo, ..."}
    """
    metadata_path = os.path.join(output_dir, "metadata.jsonl")

    with open(metadata_path, "w") as f:
        for path, caption in zip(image_paths, captions):
            entry = {
                "file_name": os.path.basename(path),
                "text": caption,
            }
            f.write(json.dumps(entry) + "\n")

    print(f"Saved metadata: {metadata_path} ({len(captions)} entries)")
    return metadata_path


def prepare_dataset(
    source: str = "dota",
    input_dir: str = "",
    output_dir: str = "data/dashcam_lora",
    num_images: int = 500,
    device: str = "cuda",
) -> str:
    """Full dataset preparation pipeline.

    Args:
        source: "dota" for DoTA anomaly frames, "nexar" for crash videos, "bdd100k" for general driving, "local" for local folder
        input_dir: Path to local images (only for source="local")
        output_dir: Where to save processed dataset
        num_images: Number of images to use
        device: Device for BLIP captioning

    Returns:
        Path to output directory (ready for training)
    """
    print(f"\n{'='*60}")
    print(f"PREPARING DASHCAM DATASET FOR LORA TRAINING")
    print(f"  Source: {source}")
    print(f"  Images: {num_images}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    # Step 1: Get images
    if source == "dota":
        image_paths = download_dota_frames(output_dir, num_images)
    elif source == "nexar":
        image_paths = download_nexar_crash_frames(output_dir, num_images)
    elif source == "bdd100k":
        image_paths = download_bdd100k_subset(output_dir, num_images)
    elif source == "local":
        if not input_dir:
            raise ValueError("input_dir required for source='local'")
        image_paths = load_local_images(input_dir, output_dir, num_images)
    else:
        raise ValueError(f"Unknown source: {source}. Use 'dota', 'nexar', 'bdd100k', or 'local'")

    # Step 2: Auto-caption
    captions = auto_caption_blip(image_paths, device=device)

    # Step 3: Create metadata
    create_metadata(output_dir, image_paths, captions)

    print(f"\nDataset ready at: {output_dir}")
    print(f"  {len(image_paths)} images + metadata.jsonl")
    print(f"\nNext step: run training with:")
    print(f"  python scripts/train_dashcam_lora.py --data_dir {output_dir}")

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dashcam dataset for LoRA training")
    parser.add_argument("--source", choices=["dota", "nexar", "bdd100k", "local"], default="dota")
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="data/dashcam_lora")
    parser.add_argument("--num_images", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    prepare_dataset(
        source=args.source,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_images=args.num_images,
        device=args.device,
    )
