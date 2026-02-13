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


class VRAMManager:
    """Manages GPU memory across multi-model pipeline stages.

    On a T4 (15GB usable VRAM), we can't load multiple large models
    simultaneously. This manager handles sequential loading/unloading
    and profiles memory at each stage for reporting.
    """

    def __init__(self, device: Optional[str] = None):
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

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
            "t4_headroom_mb": round(15360 - peak_overall, 1) if self.device == "cuda" else None,
        }

    def print_report(self):
        """Print a human-readable VRAM report."""
        r = self.report()
        print(f"\n--- VRAM Report ({r['device']}) ---")
        for s in r["stages"]:
            print(f"  [{s['stage']}] allocated={s['allocated_mb']}MB  peak={s['peak_mb']}MB")
        print(f"  Peak overall: {r['peak_overall_mb']}MB")
        if r["t4_headroom_mb"] is not None:
            print(f"  T4 headroom: {r['t4_headroom_mb']}MB remaining")
        print("---\n")


if __name__ == "__main__":
    mgr = VRAMManager()
    print(f"Device: {mgr.device}")
    mgr.snapshot("init")
    mgr.print_report()
