"""Prepare dashcam image dataset for SDXL LoRA fine-tuning.

Two modes:
1. AUTO: Download dashcam images from HuggingFace BDD100K subset
2. LOCAL: Use your own folder of dashcam images

Then auto-caption each image using BLIP-2 and create
a metadata.jsonl file in HuggingFace Dataset format.

Output structure:
    data/dashcam_lora/
    ├── img_0001.png
    ├── img_0002.png
    ├── ...
    └── metadata.jsonl   ← {"file_name": "img_0001.png", "text": "a dashcam photo of..."}

Usage:
    # From HuggingFace (recommended for HPC):
    python scripts/prepare_dashcam_data.py --source hf --num_images 500

    # From local folder:
    python scripts/prepare_dashcam_data.py --source local --input_dir /path/to/dashcam_images

    # From Jupyter:
    from scripts.prepare_dashcam_data import prepare_dataset
    prepare_dataset(source="hf", num_images=500)
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image


def download_bdd100k_subset(output_dir: str, num_images: int = 500) -> list[str]:
    """Download a subset of BDD100K dashcam images from HuggingFace.

    BDD100K is the largest open driving dataset with 100K dashcam images.
    We download a small subset for LoRA training.

    Args:
        output_dir: Directory to save images
        num_images: Number of images to download (200-1000 recommended)

    Returns:
        List of saved image paths
    """
    from datasets import load_dataset

    print(f"Loading BDD100K from HuggingFace (first {num_images} images)...")

    # BDD100K on HuggingFace (dgural/bdd100k)
    ds = load_dataset(
        "dgural/bdd100k",
        split=f"train[:{num_images}]",
        trust_remote_code=True,
    )

    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    for i, sample in enumerate(ds):
        img = sample["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize to 1024x1024 for SDXL (crop center)
        img = center_crop_resize(img, 1024)

        filename = f"img_{i:04d}.png"
        path = os.path.join(output_dir, filename)
        img.save(path)
        saved_paths.append(path)

        if (i + 1) % 50 == 0:
            print(f"  Saved {i + 1}/{num_images} images")

    print(f"Downloaded {len(saved_paths)} BDD100K images to {output_dir}")
    return saved_paths


def load_local_images(input_dir: str, output_dir: str, num_images: int = 500) -> list[str]:
    """Load and preprocess dashcam images from a local folder.

    Args:
        input_dir: Directory containing dashcam images
        output_dir: Directory to save processed images
        num_images: Maximum number of images to use

    Returns:
        List of saved image paths
    """
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
    """Generate captions for images using BLIP-2.

    BLIP-2 is a lightweight vision-language model (~3GB VRAM).
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

        # Prefix with dashcam context
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
    source: str = "hf",
    input_dir: str = "",
    output_dir: str = "data/dashcam_lora",
    num_images: int = 500,
    device: str = "cuda",
) -> str:
    """Full dataset preparation pipeline.

    Args:
        source: "hf" for HuggingFace BDD100K, "local" for local folder
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
    if source == "hf":
        image_paths = download_bdd100k_subset(output_dir, num_images)
    elif source == "local":
        if not input_dir:
            raise ValueError("input_dir required for source='local'")
        image_paths = load_local_images(input_dir, output_dir, num_images)
    else:
        raise ValueError(f"Unknown source: {source}. Use 'hf' or 'local'")

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
    parser.add_argument("--source", choices=["hf", "local"], default="hf")
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
