from pathlib import Path

import comfy.utils
import torch
from comfy import model_management, model_patcher

from .model import Qwen21Upscaler


class Qwen21LatentUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": (
                    "LATENT",
                    {"tooltip": "Clean Qwen Image 2.1 latents."},
                )
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    CATEGORY = "latent/upscaling"
    DESCRIPTION = "Learned 2x upscale for clean Qwen Image 2.1 latents. Follow with a sampler to add detail."

    def upscale(self, latent):
        samples = latent["samples"]
        if samples.ndim != 4 or samples.shape[1] != 64:
            raise ValueError("Expected Qwen Image 2.1 latents with 64 channels.")
        if "noise_mask" in latent:
            raise ValueError("Masked latents are not supported.")

        network = Qwen21Upscaler()
        path = Path(__file__).parent / "models" / "qwen21-latent-2x-v1.safetensors"
        network.load_state_dict(
            comfy.utils.load_torch_file(str(path), safe_load=True), strict=True
        )
        network.eval()
        device = model_management.get_torch_device()
        patcher = model_patcher.CoreModelPatcher(
            network,
            load_device=device,
            offload_device=model_management.unet_offload_device(),
        )
        model_management.load_models_gpu(
            [patcher],
            memory_required=128 * 1024**2 + samples.numel() * 256,
            force_full_load=True,
        )
        result = latent.copy()
        result["samples"] = network(samples.to(device=device, dtype=torch.float32)).to(
            device=samples.device, dtype=samples.dtype
        )
        return (result,)


NODE_CLASS_MAPPINGS = {"Qwen21LatentUpscale": Qwen21LatentUpscale}
NODE_DISPLAY_NAME_MAPPINGS = {"Qwen21LatentUpscale": "Qwen Image 2.1 Latent Upscale 2x"}
