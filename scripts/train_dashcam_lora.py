"""SDXL LoRA fine-tuning on dashcam images.

Trains a Low-Rank Adaptation (LoRA) on Stable Diffusion XL to learn
dashcam-specific visual characteristics:
- Wide-angle lens distortion
- Dashboard reflections
- Road-centric composition
- Weather/lighting patterns typical of dashcam footage

LoRA only trains ~4-50MB of additional parameters while the full
SDXL model (6.5GB) stays frozen. This is efficient enough for
a single A100 40GB GPU in ~1-2 hours.

Paper reference: LoRA (arxiv 2106.09685)

Usage:
    # From command line:
    python scripts/train_dashcam_lora.py --data_dir data/dashcam_lora --epochs 10

    # From Jupyter:
    from scripts.train_dashcam_lora import train_lora
    train_lora(data_dir="data/dashcam_lora", num_epochs=10)
"""

import os
import json
import math
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class DashcamDataset(Dataset):
    """Dataset that loads images + captions from metadata.jsonl.

    Expected format in data_dir:
        img_0001.png
        img_0002.png
        ...
        metadata.jsonl  ← {"file_name": "img_0001.png", "text": "dashcam photo, ..."}
    """

    def __init__(self, data_dir: str, resolution: int = 1024):
        self.data_dir = data_dir
        self.resolution = resolution

        # Load metadata
        metadata_path = os.path.join(data_dir, "metadata.jsonl")
        self.entries = []
        with open(metadata_path) as f:
            for line in f:
                self.entries.append(json.loads(line))

        print(f"Loaded {len(self.entries)} training samples from {data_dir}")

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # [-1, 1] range for diffusion
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_path = os.path.join(self.data_dir, entry["file_name"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return {"pixel_values": image, "caption": entry["text"]}


def train_lora(
    data_dir: str = "data/dashcam_lora",
    output_dir: str = "models/dashcam_lora",
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    lora_rank: int = 8,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    resolution: int = 1024,
    save_every_n_epochs: int = 5,
    seed: int = 42,
):
    """Train SDXL LoRA on dashcam images.

    This implements the standard diffusion model LoRA training loop:
    1. Load pretrained SDXL (frozen)
    2. Add LoRA adapters to UNet attention layers
    3. For each image: encode → add noise → predict noise → compute loss
    4. Only update LoRA weights (tiny fraction of total params)

    Args:
        data_dir: Path to dataset (images + metadata.jsonl)
        output_dir: Where to save LoRA weights
        num_epochs: Training epochs (10-20 for small datasets)
        learning_rate: LoRA learning rate (1e-4 is standard)
        lora_rank: LoRA rank (4-16, higher = more capacity but slower)
        batch_size: Per-GPU batch size (1 for A100 with SDXL)
        gradient_accumulation_steps: Effective batch = batch_size × this
        resolution: Image resolution (1024 for SDXL)
        save_every_n_epochs: Checkpoint frequency
        seed: Random seed for reproducibility
    """
    from diffusers import StableDiffusionXLPipeline, DDPMScheduler, AutoencoderKL
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    from peft import LoraConfig, get_peft_model
    import gc

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"SDXL LoRA TRAINING")
    print(f"{'='*60}")
    print(f"  Dataset:     {data_dir}")
    print(f"  Output:      {output_dir}")
    print(f"  Epochs:      {num_epochs}")
    print(f"  LR:          {learning_rate}")
    print(f"  LoRA rank:   {lora_rank}")
    print(f"  Batch size:  {batch_size} × {gradient_accumulation_steps} = {batch_size * gradient_accumulation_steps}")
    print(f"  Resolution:  {resolution}")
    print(f"  Device:      {device}")
    print(f"{'='*60}\n")

    # === Load SDXL components ===
    print("Loading SDXL components...")
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    # Load noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Load tokenizers
    tokenizer_1 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")

    # Load text encoders (frozen)
    text_encoder_1 = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_id, subfolder="text_encoder_2", torch_dtype=dtype
    ).to(device)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # Load VAE (frozen)
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=dtype
    ).to(device)
    vae.requires_grad_(False)

    # Load UNet (we'll add LoRA to this)
    print("Loading UNet...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=dtype, use_safetensors=True,
        variant="fp16" if device == "cuda" else None,
    )
    unet = pipe.unet.to(device)
    del pipe
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # === Add LoRA adapters ===
    print(f"Adding LoRA adapters (rank={lora_rank})...")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,  # alpha = rank is standard
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",  # self-attention
            "proj_in", "proj_out",  # projections
        ],
        lora_dropout=0.05,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Enable gradient checkpointing to save VRAM
    unet.enable_gradient_checkpointing()

    # === Dataset ===
    dataset = DashcamDataset(data_dir, resolution=resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # === Optimizer ===
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-2)

    # Cosine LR schedule
    total_steps = len(dataloader) * num_epochs // gradient_accumulation_steps
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # === Training loop ===
    print(f"\nStarting training ({total_steps} optimizer steps)...")

    global_step = 0
    best_loss = float("inf")

    for epoch in range(num_epochs):
        unet.train()
        epoch_loss = 0.0

        for step, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device, dtype=dtype)
            captions = batch["caption"]

            # 1. Encode images to latent space (VAE)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # 2. Encode text prompts (both CLIP encoders for SDXL)
            with torch.no_grad():
                tokens_1 = tokenizer_1(
                    captions, padding="max_length",
                    max_length=tokenizer_1.model_max_length,
                    truncation=True, return_tensors="pt"
                ).input_ids.to(device)
                tokens_2 = tokenizer_2(
                    captions, padding="max_length",
                    max_length=tokenizer_2.model_max_length,
                    truncation=True, return_tensors="pt"
                ).input_ids.to(device)

                enc_1 = text_encoder_1(tokens_1, output_hidden_states=True)
                enc_2 = text_encoder_2(tokens_2, output_hidden_states=True)

                # SDXL uses penultimate hidden states
                prompt_embeds = torch.cat([
                    enc_1.hidden_states[-2],
                    enc_2.hidden_states[-2],
                ], dim=-1)

                pooled_prompt_embeds = enc_2[0]

            # 3. Sample noise and timestep
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device
            ).long()

            # 4. Add noise to latents (forward diffusion)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 5. SDXL needs add_time_ids (original_size, crops, target_size)
            add_time_ids = torch.tensor(
                [[resolution, resolution, 0, 0, resolution, resolution]],
                dtype=dtype, device=device,
            ).repeat(latents.shape[0], 1)

            # 6. Predict noise with UNet (LoRA weights are trainable)
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids,
                },
            ).sample

            # 7. Compute loss (MSE between predicted and actual noise)
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            loss = loss / gradient_accumulation_steps

            loss.backward()
            epoch_loss += loss.item() * gradient_accumulation_steps

            # Gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        avg_loss = epoch_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch + 1}/{num_epochs}  |  Loss: {avg_loss:.6f}  |  LR: {current_lr:.2e}")

        # Save checkpoint
        if (epoch + 1) % save_every_n_epochs == 0 or (epoch + 1) == num_epochs:
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch{epoch + 1}")
            unet.save_pretrained(checkpoint_dir)
            print(f"  Saved checkpoint: {checkpoint_dir}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    # === Save final LoRA weights ===
    final_dir = os.path.join(output_dir, "final")
    unet.save_pretrained(final_dir)

    # Also save training config for reference
    config = {
        "data_dir": data_dir,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "resolution": resolution,
        "best_loss": best_loss,
        "total_steps": global_step,
        "num_images": len(dataset),
    }
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"  Best loss:    {best_loss:.6f}")
    print(f"  Total steps:  {global_step}")
    print(f"  LoRA saved:   {final_dir}")
    print(f"{'='*60}")
    print(f"\nTo use in pipeline:")
    print(f"  pipeline = CrashScenePipeline(lora_weights='{final_dir}')")

    # Clean up
    del unet, vae, text_encoder_1, text_encoder_2
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return final_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SDXL LoRA on dashcam images")
    parser.add_argument("--data_dir", type=str, default="data/dashcam_lora")
    parser.add_argument("--output_dir", type=str, default="models/dashcam_lora")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    args = parser.parse_args()

    train_lora(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        lora_rank=args.lora_rank,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
    )
