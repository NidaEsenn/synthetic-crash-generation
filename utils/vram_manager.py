import torch
import gc
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VRAMSnapshot:
    """Snapshot of GPU memory at a specific pipeline stage."""
    stage: str
    allocated_mb: float
    reserved_mb: float
    peak_mb: float


GPU_PROFILES = {
    "t4": 15360,       # 15 GB
    "a100_40": 40960,  # 40 GB
    "a100_80": 81920,  # 80 GB
    "v100": 16384,     # 16 GB
    "h200": 143360,    # 140 GB
}


class VRAMManager:
    """Manages GPU memory across multi-model pipeline stages.

    Supports configurable GPU profiles for accurate headroom reporting.
    Default: A100 40GB (Northeastern HPC). Use gpu_profile="t4" for Colab.
    """

    def __init__(self, device: Optional[str] = None, gpu_profile: str = "a100_40"):
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.gpu_profile = gpu_profile
        self.gpu_total_mb = GPU_PROFILES.get(gpu_profile, 40960)
        self.snapshots: list[VRAMSnapshot] = []
        self._active_models: dict[str, object] = {}

    def snapshot(self, stage: str) -> VRAMSnapshot:
        """Record current GPU memory usage."""
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            peak = torch.cuda.max_memory_allocated() / 1024**2
        else:
            allocated = reserved = peak = 0.0

        snap = VRAMSnapshot(
            stage=stage,
            allocated_mb=round(allocated, 1),
            reserved_mb=round(reserved, 1),
            peak_mb=round(peak, 1),
        )
        self.snapshots.append(snap)
        return snap

    def unload(self, name: str):
        """Unload a model and free GPU memory."""
        if name in self._active_models:
            del self._active_models[name]
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def unload_all(self):
        """Unload all tracked models."""
        self._active_models.clear()
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def register(self, name: str, model: object):
        """Track a loaded model for later cleanup."""
        self._active_models[name] = model

    def report(self) -> dict:
        """Generate a VRAM usage report across all pipeline stages."""
        peak_overall = max((s.peak_mb for s in self.snapshots), default=0.0)
        return {
            "device": self.device,
            "stages": [
                {
                    "stage": s.stage,
                    "allocated_mb": s.allocated_mb,
                    "reserved_mb": s.reserved_mb,
                    "peak_mb": s.peak_mb,
                }
                for s in self.snapshots
            ],
            "peak_overall_mb": peak_overall,
            "gpu_profile": self.gpu_profile,
            "gpu_total_mb": self.gpu_total_mb,
            "headroom_mb": round(self.gpu_total_mb - peak_overall, 1) if self.device == "cuda" else None,
        }

    def print_report(self):
        """Print a human-readable VRAM report."""
        r = self.report()
        print(f"\n--- VRAM Report ({r['device']}) ---")
        for s in r["stages"]:
            print(f"  [{s['stage']}] allocated={s['allocated_mb']}MB  peak={s['peak_mb']}MB")
        print(f"  Peak overall: {r['peak_overall_mb']}MB")
        if r["headroom_mb"] is not None:
            print(f"  {r['gpu_profile'].upper()} headroom: {r['headroom_mb']}MB / {r['gpu_total_mb']}MB")
        print("---\n")


if __name__ == "__main__":
    mgr = VRAMManager()
    print(f"Device: {mgr.device}")
    mgr.snapshot("init")
    mgr.print_report()
