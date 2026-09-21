# Qwen Image 2.1 Latent Upscale

A ComfyUI node and a 2× upscaling model for Qwen Image 2.1.

## Install

Copy this folder into `ComfyUI/custom_nodes/` and restart. The upscaler model is included; no extra packages are needed. Update ComfyUI and install the [Qwen Image 2.1 models](https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1) first.

## Use

Load the [example workflow](workflows/Qwen21LatentUpscale.json) and select your Qwen models. It samples at 1 MP, upscales clean latents, then adds detail at 4 MP (1728 × 2304).

Adjust **steps** and **denoise** on the final KSampler for more detail or larger changes.

## Credits

Architecture and initial weights adapted from [LoganBooker/SesquiLSR](https://github.com/LoganBooker/SesquiLSR) (MIT). Built for [ComfyUI](https://github.com/Comfy-Org/ComfyUI) and [Qwen Image 2.1](https://github.com/QwenLM/Qwen-Image-2.1).
