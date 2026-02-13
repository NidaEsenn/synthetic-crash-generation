import torch
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test imports
try:
    from diffusers import StableDiffusionPipeline
    print("✅ Diffusers OK")
except ImportError as e:
    print(f"❌ Diffusers: {e}")

try:
    import openai
    print("✅ OpenAI OK")
except ImportError as e:
    print(f"❌ OpenAI: {e}")

print("\nSetup complete!")