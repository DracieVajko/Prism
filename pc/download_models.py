#!/usr/bin/env python3
"""Download diffusion models into DiffusionServer/models/ (resumable).

Presets (all from public HuggingFace repos, no login needed):
  odysseus-default : all 4 models below (~19 GB)
  sd15             : DreamShaper_8 + AbsoluteReality_1.8 (~5 GB)
  sdxl             : AnimagineXL_3.1 + JuggernautXL_v9 (~14 GB)

Usage:
  python download_models.py --list
  python download_models.py --preset odysseus-default
  python download_models.py DreamShaper_8 JuggernautXL_v9
  python download_models.py --civitai 12345 --target models/MyModel --name model.safetensors --civitai-token ABC
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent / "models"

# Cancellation flags (model_id -> set by model_install_manager.cancel)
_CANCEL: set = set()


class CancelledError(Exception):
    pass


def _check_cancel(task_id: str):
    if task_id in _CANCEL:
        raise CancelledError(f"cancelled: {task_id}")

TASKS = [
    {"id": "DreamShaper_8", "kind": "snapshot", "repo": "Lykon/dreamshaper-8",
     "target": "DreamShaper_8", "min_gb": 2.0,
     "allow": ["*.fp16.safetensors", "*.json", "*.txt", "*.model",
               "tokenizer/*", "scheduler/*", "feature_extractor/*"],
     "note": "official diffusers repo, fp16 weights + configs (no single-file on HF)",
     "license": "CreativeML Open RAIL-M (see repo card)"},
    {"id": "AbsoluteReality_1.8", "kind": "file",
     "repo": "Lykon/AbsoluteReality",
     "file": "AbsoluteReality_1.8.1_pruned.safetensors",
     "target": "AbsoluteReality_1.8", "min_gb": 1.5,
     "license": "CreativeML Open RAIL-M (see repo card)"},
    {"id": "AnimagineXL_3.1", "kind": "file", "repo": "cagliostrolab/animagine-xl-3.1",
     "file": "animagine-xl-3.1.safetensors", "target": "AnimagineXL_3.1", "min_gb": 5.5,
     "license": "Fair AI Public License 1.0-SD (see repo card)"},
    {"id": "JuggernautXL_v9", "kind": "file", "repo": "RunDiffusion/Juggernaut-XL-v9",
     "file": "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
     "target": "JuggernautXL_v9", "min_gb": 5.5,
     "license": "CreativeML Open RAIL-M (see repo card)"},
    {"id": "SDXL_Turbo", "kind": "file", "repo": "stabilityai/sdxl-turbo",
     "file": "sd_xl_turbo_1.0_fp16.safetensors",
     "target": "SDXL_Turbo", "min_gb": 6.0,
     "single_file_config": "stabilityai/sdxl-turbo",
     "pipeline": "StableDiffusionXLPipeline",
     "license": "OpenRAIL++-M (see repo card)"},
    {"id": "NoobAI-XL-Vpred", "kind": "file", "repo": "Laxhar/noobai-XL-Vpred-1.0",
     "file": "NoobAI-XL-Vpred-v1.0.safetensors",
     "target": "NoobAI-XL-Vpred", "min_gb": 6.5,
     "single_file_config": "stabilityai/stable-diffusion-xl-base-1.0",
     "pipeline": "StableDiffusionXLPipeline",
     "license": "Fair AI Public License 1.0-SD (see repo card)"},
    {"id": "amazing-logos-v3", "kind": "snapshot", "repo": "iamkaikai/amazing-logos-v3",
     "target": "amazing-logos-v3", "min_gb": 2.0,
     "allow": ["model_index.json", "unet/config.json", "unet/diffusion_pytorch_model.bin",
               "vae/config.json", "vae/diffusion_pytorch_model.bin",
               "text_encoder/config.json", "text_encoder/pytorch_model.bin",
               "tokenizer/*", "scheduler/scheduler_config.json",
               "feature_extractor/preprocessor_config.json",
               "safety_checker/config.json", "safety_checker/pytorch_model.bin"],
     "note": "SD1.5 fine-tuned on 200k logos, 512px, prompt format: 'Simple elegant logo for X'",
     "license": "CreativeML Open RAIL-M (see repo card)"},
]

PRESETS = {
    "odysseus-default": [t["id"] for t in TASKS],
    "sd15": ["DreamShaper_8", "AbsoluteReality_1.8"],
    "sdxl": ["AnimagineXL_3.1", "JuggernautXL_v9"],
    "anime": ["AnimagineXL_3.1", "NoobAI-XL-Vpred"],
    "logo": ["amazing-logos-v3"],
    "sdxl-turbo": ["SDXL_Turbo"],
    "upscale": ["RealESRGAN_x4v3"],
    "swinir": ["swinir-realsr-x4-upscale"],
    "zimage": ["z-image-turbo"],
    "zimage-fp8": ["z-image-turbo-fp8"],
    "colorize": ["ddcolor-full", "ddcolor-artistic", "deoldify-artistic"],
    "flux-kontext": ["flux-kontext-dev"],
    "flux-kontext-fp8": ["flux-kontext-dev-fp8"],
    "ocr": ["nemotron-ocr-0", "granite-docling-258m", "qwen3-vl-2b"],
    "vlm": ["qwen3-vl-2b"],
}

# PC caption/chat use Qwen3-VL-2B (listed under OCR_TASKS so one copy serves
# OCR, description, caption and vision chat). SmolVLM was removed from the PC
# catalog after Qwen won the comparison; Android keeps its own SmolVLM2 runtime.
VLM_TASKS = []

# OCR: Xiaomi Nemotron-3.5-OCR-0 (Qwen3.5-0.8B based, ~1.67 GB)
OCR_TASKS = [
    {"id": "nemotron-ocr-0", "kind": "snapshot", "repo": "xiaomi-research/Xiaomi-OCR-0",
     "target": "nemotron-ocr-0", "min_gb": 1.6,
     "allow": ["*.safetensors", "*.json", "tokenizer/*", "tokenizer_config.json",
               "*.model", "*.txt", "chat_template.jinja", "generation_config.json"],
     "license": "Apache-2.0 (see repo card)",
     "note": "OCR model based on Qwen3.5-0.8B. Requires transformers>=4.51."},
    {"id": "granite-docling-258m", "kind": "snapshot",
     "repo": "ibm-granite/granite-docling-258M",
     "target": "granite-docling-258m", "min_gb": 0.45,
     "allow": ["*.json", "*.safetensors", "*.model", "*.txt",
               "tokenizer/*", "chat_template*"],
     "license": "Apache-2.0",
     "note": "Granite-Docling 258M: tiny, fast PDF/page -> DocTags/Markdown "
             "(tables, formulas, layout). CPU-friendly."},
    {"id": "ovis-ocr2", "kind": "snapshot",
     "repo": "ATH-MaaS/OvisOCR2", "target": "ovis-ocr2", "min_gb": 1.6,
     "allow": ["*.json", "*.safetensors", "*.txt", "*.model",
               "tokenizer/*", "chat_template*", "preprocessor_config.json"],
     "license": "Apache-2.0",
     "note": "OvisOCR2 853M: 96.58 OmniDocBench v1.6, layout/tables/formulas."},
    {"id": "glm-ocr", "kind": "snapshot",
     "repo": "zai-org/GLM-OCR", "target": "glm-ocr", "min_gb": 2.5,
     "allow": ["*.json", "*.safetensors", "*.txt", "*.model",
               "tokenizer/*", "chat_template*", "preprocessor_config.json",
               "processor_config.json"],
     "license": "MIT",
     "note": "GLM-OCR 0.9B: 94.62 OmniDocBench, tables, formulas, 8 languages."},
    {"id": "paddleocr-vl-1.6", "kind": "snapshot",
     "repo": "PaddlePaddle/PaddleOCR-VL-1.6", "target": "paddleocr-vl-1.6",
     "min_gb": 1.8,
     "allow": ["*.json", "*.safetensors", "*.txt", "*.model", "*.py",
               "*.jinja", "tokenizer*", "chat_template*",
               "preprocessor_config.json", "processor_config.json"],
     "license": "Apache-2.0",
     "note": "PaddleOCR-VL 1.6 (0.96B): document parsing, 109 languages. "
             "Loaded through the native transformers class (no remote code)."},
    {"id": "lightonocr-2-1b", "kind": "snapshot",
     "repo": "lightonai/LightOnOCR-2-1B", "target": "lightonocr-2-1b",
     "min_gb": 1.9,
     "allow": ["*.json", "*.safetensors", "*.txt", "*.model",
               "tokenizer/*", "chat_template*", "processor_config.json",
               "preprocessor_config.json"],
     "license": "Apache-2.0",
     "note": "LightOnOCR 2 1B: fast end-to-end page OCR, 2 GB."},
    {"id": "qianfan-ocr", "kind": "snapshot",
     "repo": "baidu/Qianfan-OCR", "target": "qianfan-ocr", "min_gb": 9.0,
     "allow": ["*.json", "*.safetensors", "*.txt", "*.model",
               "tokenizer/*", "chat_template*", "preprocessor_config.json",
               "processor_config.json"],
     "license": "Apache-2.0",
     "note": "Qianfan-OCR 4.7B (192 languages, KIE, layout). ~9.5 GB; needs offload."},
    {"id": "qwen3-vl-2b", "kind": "snapshot",
     "repo": "Qwen/Qwen3-VL-2B-Instruct",
     "target": "qwen3-vl-2b", "min_gb": 4.0,
     "allow": ["*.json", "*.safetensors", "*.txt", "*.model",
               "tokenizer/*", "chat_template*", "preprocessor_config.json",
               "video_preprocessor_config.json"],
     "license": "Apache-2.0",
     "note": "Qwen3-VL 2B: OCR in 32 languages plus object recognition and "
             "image description. Fits a 6 GB GPU."},
]

# Optional ASR engines (transformers 5.x classes, generic hf_asr.py).
# Nemotron ASR was removed: the 0.6B RNNT generation path is broken in
# transformers 5.21 (empty transcripts) and the 30B variants do not fit this
# class of machine. Nemotron OCR is a separate, working model.
AUDIO_TASKS = [
    {"id": "cohere-transcribe", "kind": "snapshot",
     "repo": "CohereLabs/cohere-transcribe-03-2026",
     "target": "cohere-transcribe", "min_gb": 3.9,
     "allow": ["*.json", "*.safetensors", "*.txt", "*.model",
               "tokenizer/*", "preprocessor_config.json"],
     "license": "Apache-2.0",
     "note": "Cohere Transcribe (14 languages, lowest word error)."},
    {"id": "granite-speech-470m", "kind": "snapshot",
     "repo": "ibm-granite/granite-speech-5.0-470m-turboctc",
     "target": "granite-speech-470m", "min_gb": 0.9,
     "allow": ["*.json", "*.safetensors", "*.txt", "*.model",
               "tokenizer/*", "preprocessor_config.json"],
     "license": "Apache-2.0",
     "note": "Granite Speech 5.0 470M CTC (7 languages, fast, ~1 s)."},
    {"id": "granite-speech-2b", "kind": "snapshot",
     "repo": "ibm-granite/granite-speech-5.0-2b",
     "target": "granite-speech-2b", "min_gb": 4.8,
     "allow": ["*.json", "*.safetensors", "*.txt", "*.model",
               "tokenizer/*", "preprocessor_config.json"],
     "license": "Apache-2.0",
     "note": "Granite Speech 5.0 2B (7 languages, higher accuracy)."},
    {"id": "parakeet-v3", "kind": "snapshot",
     "repo": "nvidia/parakeet-tdt-0.6b-v3", "target": "parakeet-v3",
     "min_gb": 5.4,
     "allow": ["*.json", "*.safetensors", "*.txt", "*.model",
               "tokenizer/*", "preprocessor_config.json"],
     "license": "CC-BY-4.0",
     "note": "Parakeet TDT 0.6B v3 (25 European languages, auto language)."},
    {"id": "parakeet-v2", "kind": "snapshot",
     "repo": "nvidia/parakeet-tdt-0.6b-v2", "target": "parakeet-v2",
     "min_gb": 2.4,
     "allow": ["*.json", "*.safetensors", "*.txt", "*.model",
               "tokenizer/*", "preprocessor_config.json"],
     "license": "CC-BY-4.0",
     "note": "Parakeet TDT 0.6B v2 (English)."},
    {"id": "parakeet-unified-en", "kind": "snapshot",
     "repo": "nvidia/parakeet-unified-en-0.6b",
     "target": "parakeet-unified-en", "min_gb": 2.4,
     "allow": ["*.json", "*.safetensors", "*.txt", "*.model",
               "tokenizer/*", "preprocessor_config.json"],
     "license": "see repo card",
     "note": "Parakeet Unified English (streaming, English)."},
    {"id": "nb-whisper-medium", "kind": "snapshot",
     "repo": "NbAiLab/nb-whisper-medium", "target": "nb-whisper-medium",
     "min_gb": 2.9,
     "allow": ["model.safetensors", "config.json", "generation_config.json",
               "tokenizer.json", "tokenizer_config.json",
               "preprocessor_config.json", "vocab.json", "merges.txt"],
     "license": "Apache-2.0",
     "note": "NB-Whisper Medium (Norwegian, National Library)."},
    {"id": "ivrit-whisper-turbo", "kind": "snapshot",
     "repo": "ivrit-ai/whisper-large-v3-turbo-ct2",
     "target": "ivrit-whisper-turbo", "min_gb": 1.5,
     "allow": ["model.bin", "config.json", "tokenizer.json",
               "vocabulary.json", "preprocessor_config.json"],
     "license": "Apache-2.0",
     "note": "ivrit.ai Whisper large-v3-turbo (Hebrew, CT2, ~2 s)."},
  ]

# Restoration / colorization (ONNX, CPU or GPU via onnxruntime).
# DDColor: state-of-the-art B&W photo colorization (ICCV 2023, Apache-2.0).
# SwinIR: super-resolution / restoration (ICCV 2021, Apache-2.0).
RESTORATION_TASKS = [
    {"id": "ddcolor-full", "kind": "file",
     "repo": "Diogo122333/ddcolor-512-fp16",
     "file": "ddcolor-512-fp16.onnx",
     "target": "ddcolor-full", "min_gb": 0.40,
     "license": "Apache-2.0 (DDColor upstream; community ONNX export)",
     "note": "DDColor full (convnext-l) colorization, ~435 MB fp16, 512x512. "
             "Uses the same ab contract as tiny; better chroma, no green cast "
             "on the test slide."},
    {"id": "ddcolor-artistic", "kind": "file",
     "repo": "facefusion/models-3.0.0",
     "file": "ddcolor_artistic.onnx",
     "target": "ddcolor-artistic", "min_gb": 0.85,
     "license": "Apache-2.0 (DDColor artistic checkpoint, FaceFusion repack)",
     "note": "DDColor artistic (~980 MB ONNX): richer, less literal chroma than "
             "ddcolor-full. Best-looking result on the B&W street test."},
    {"id": "deoldify-artistic", "kind": "file",
     "repo": "facefusion/models-3.0.0",
     "file": "deoldify_artistic.onnx",
     "target": "deoldify-artistic", "min_gb": 0.2,
     "license": "MIT (DeOldify, FaceFusion ONNX repack)",
     "note": "DeOldify Artistic (~250 MB ONNX): strongest saturation, keeps the "
             "original luminance. Fastest of the three colorizers."},
    {"id": "swinir-realsr-x4", "kind": "file",
     "repo": "Heliosoph/swinir-onnx",
     "file": "swinir_realsr_x4.onnx",
     "target": "swinir-realsr-x4", "min_gb": 0.1,
     "license": "Apache-2.0 (see JingyunLiang/SwinIR)",
     "note": "SwinIR real-world super-resolution x4 (ICCV 2021, ~122 MB, "
             "64x64->256x256). ONNX, runs on CPU/CUDA via onnxruntime. "
             "Apache-2.0."},
    # FLUX.1-Kontext-dev: instruction-based restoration/editing. Grouped with
    # restoration (not upscale) so it never appears in the upscale dropdown.
    # Gated repo: set the HF token in Settings first. Needs >= 64 GB RAM.
    {"id": "flux-kontext-dev", "kind": "snapshot",
     "repo": "black-forest-labs/FLUX.1-Kontext-dev",
     "target": "flux-kontext-dev", "min_gb": 28.0,
     "allow": ["*.json", "*.safetensors", "*.txt", "tokenizer/*",
               "scheduler/*", "text_encoder/*", "text_encoder_2/*",
               "transformer/*", "vae/*"],
     "license": "FLUX.1 [dev] Non-Commercial License (gated)",
     "note": "FLUX.1-Kontext-dev bf16 (~31.8 GB) restoration/editing. Gated: set "
             "the HF token in Settings first. Needs >= 64 GB RAM to load."},
    # Ungated FP8 mirror: same FluxKontextPipeline, ~21.8 GB. Still needs an
    # Ada+ GPU for FP8 matmul; on Ampere torch raises a dtype error.
    {"id": "flux-kontext-dev-fp8", "kind": "snapshot",
     "repo": "AlekseyCalvin/Flux_Kontext_Dev_fp8_scaled_diffusers",
     "target": "flux-kontext-dev-fp8", "min_gb": 18.0,
     "allow": ["*.json", "*.safetensors", "*.txt", "tokenizer/*",
               "scheduler/*", "text_encoder/*", "text_encoder_2/*",
               "transformer/*", "vae/*"],
     "license": "FLUX.1 [dev] Non-Commercial License",
     "note": "FLUX.1-Kontext-dev FP8 scaled (~21.8 GB) restoration/editing. "
             "Same pipeline as bf16; requires FP8-capable GPU."},
]

# Upscale weights (small, also downloaded to the mobile package via ModelDownloader).
# RealESRGAN_x4plus.pth: x4, general photos (BSD-3-Clause, xinntao/Real-ESRGAN,
# stable GitHub release asset — no official HF repo for this weight).
UPSCALE_TASKS = [
    {"id": "RealESRGAN_x4v3", "kind": "url",
     "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
     "file": "RealESRGAN_x4plus.pth", "target": "RealESRGAN_x4v3",
     "min_gb": 0.05,
     "license": "BSD-3-Clause (see repo card)"},
    # Same ONNX file as the restoration entry, exposed under Upscale as a
    # deterministic engine that is gentler than ESRGAN on photos and text.
    {"id": "swinir-realsr-x4-upscale", "kind": "file",
     "repo": "Heliosoph/swinir-onnx",
     "file": "swinir_realsr_x4.onnx",
     "target": "swinir-realsr-x4", "min_gb": 0.1,
     "license": "Apache-2.0 (see JingyunLiang/SwinIR)",
     "note": "SwinIR real-world SR x4 (Apache-2.0, ~122 MB, tiled): faithful "
             "texture, less crunchy than ESRGAN. Fits 4 GB GPUs."},
    # Generative refine: Real-ESRGAN base pass -> Z-Image Turbo img2img.
    {"id": "z-image-turbo", "kind": "snapshot",
     "repo": "Tongyi-MAI/Z-Image-Turbo",
     "target": "z-image-turbo", "min_gb": 30.0,
     "allow": ["*.json", "*.safetensors", "*.txt", "tokenizer/*",
               "scheduler/*", "text_encoder/*", "transformer/*", "vae/*"],
     "license": "Apache-2.0 (Tongyi-MAI, commercial use allowed)",
     "note": "Z-Image Turbo 6B + Qwen3-4B encoder + FLUX VAE (~33 GB). "
             "Generative img2img refine; runs on 6 GB VRAM with CPU offload."},
    # FP8 quantized Z-Image Turbo: same diffusers ZImagePipeline, ~40% smaller.
    # Near-bf16 quality; fits smaller RAM. Transformer 6.15 GB instead of 12 GB.
    {"id": "z-image-turbo-fp8", "kind": "snapshot",
     "repo": "ykarout/Z-Image-Turbo-FP8-Full",
     "target": "z-image-turbo-fp8", "min_gb": 14.0,
     # E4M3FN only (skip the alternate e5m2 transformer, same 6 GB again).
     "allow": ["*.json", "*.txt", "tokenizer/*", "scheduler/*",
               "text_encoder/*", "transformer/config.json",
               "transformer/diffusion_pytorch_model.safetensors", "vae/*"],
     "license": "Apache-2.0 (quantization of Tongyi-MAI/Z-Image-Turbo)",
     "note": "Z-Image Turbo FP8 (E4M3FN transformer, 6.15 GB) + Qwen3-4B "
             "encoder + VAE (~20.5 GB). Same ZImagePipeline; loads on less RAM."},
]

FAILED = []


# ---------------------------------------------------------------------------
# Post-download preparation
#
# A downloaded file is not necessarily a *usable* model. Some entries need a
# build step (for example Real-ESRGAN ships a .pth that the ONNX engine needs
# as .onnx). `prepare_task` runs that step automatically after the download, so
# "press Download" is enough for the model to work immediately.
#
# When the preparation succeeds a `.READY` manifest is written next to the
# weights. `is_task_installed` treats a missing `.READY` as "not ready" only
# when the task declares a prepare hook, so existing installs keep working
# until they are re-prepared.
# ---------------------------------------------------------------------------

READY_FILE = ".READY"


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _prepare_realesrgan(task) -> str:
    """Make the weights usable by the engine that will actually serve them.

    With torch installed the .pth is enough (`realesrgan-torch`). Without torch
    the ONNX file is required, so it is built here from the same .pth - the
    download alone would leave the model unusable.
    """
    target = BASE / task["target"]
    pth = target / "RealESRGAN_x4plus.pth"
    onnx = target / "RealESRGAN_x4plus.onnx"
    if not pth.is_file():
        raise RuntimeError("RealESRGAN .pth missing after download")
    if onnx.is_file() and onnx.stat().st_size > 10 * 1024 ** 2:
        return "pth + onnx present"
    if _torch_available():
        return "pth only (torch engine)"
    script = Path(__file__).resolve().parent / "tools" / "convert_esrgan_onnx.py"
    if not script.is_file():
        raise RuntimeError(f"converter not found: {script}")
    print(f"[{task['id']}] torch missing: building ONNX from .pth...", flush=True)
    import subprocess
    proc = subprocess.run([sys.executable, str(script)],
                          cwd=str(Path(__file__).resolve().parent),
                          capture_output=True, text=True)
    if proc.returncode != 0 or not onnx.is_file():
        tail = (proc.stderr or proc.stdout or "")[-400:]
        raise RuntimeError(f"ONNX build failed: {tail}")
    print(f"[{task['id']}] ONNX built ({onnx.stat().st_size / 1e6:.0f} MB)", flush=True)
    return "onnx built"


PREPARE = {
    "RealESRGAN_x4v3": _prepare_realesrgan,
}


def _verify_task(task) -> None:
    """Fail loudly when the freshly downloaded files cannot serve requests."""
    target = BASE / task["target"]
    kind = task.get("kind")
    if kind in ("file", "url"):
        f = target / task["file"]
        if not f.is_file():
            raise RuntimeError(f"expected {f} after download")
        if f.stat().st_size / 1e9 < task.get("min_gb", 0.0):
            raise RuntimeError(
                f"{f.name} is {f.stat().st_size / 1e9:.2f} GB, "
                f"expected >= {task['min_gb']} GB")
        return
    size = dir_size_gb(target)
    if size < task.get("min_gb", 0.0):
        raise RuntimeError(f"downloaded only {size:.2f} GB, "
                           f"expected >= {task['min_gb']} GB")


def _write_ready(task) -> None:
    target = BASE / task["target"]
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": task["id"],
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "kind": task.get("kind", ""),
        "min_gb": task.get("min_gb", 0.0),
        "size_gb": round(dir_size_gb(target), 2),
    }
    (target / READY_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare_task(task) -> str:
    """Verify a downloaded model and run its build hook, then mark it READY.

    Returns the hook result ("" when the task has no hook). Raises on any
    failure so the caller can report the model as unusable instead of pretending
    the install succeeded.
    """
    _verify_task(task)
    hook = PREPARE.get(task["id"])
    detail = hook(task) if hook else ""
    _write_ready(task)
    if detail:
        print(f"[{task['id']}] prepared: {detail}", flush=True)
    return detail


def _ready_realesrgan(task) -> bool:
    target = BASE / task["target"]
    pth = target / "RealESRGAN_x4plus.pth"
    onnx = target / "RealESRGAN_x4plus.onnx"
    has_pth = pth.is_file() and pth.stat().st_size > 20 * 1024 ** 2
    has_onnx = onnx.is_file() and onnx.stat().st_size > 10 * 1024 ** 2
    if _torch_available():
        return has_pth
    return has_onnx


# Artifact-level readiness. Preferred over the .READY manifest so an install
# made before the prepare hooks existed is still recognised as usable.
READY_CHECKS = {
    "RealESRGAN_x4v3": _ready_realesrgan,
}


def is_prepared(task) -> bool:
    """True when the model can serve requests right now."""
    if task["id"] not in PREPARE:
        return True
    check = READY_CHECKS.get(task["id"])
    if check is not None:
        try:
            if check(task):
                return True
        except Exception:
            pass
    return (BASE / task["target"] / READY_FILE).is_file()


def run_with_retries(fn, label, tries=5):
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:
            print(f"[{label}] attempt {attempt}/{tries} failed: {str(e)[:200]}", flush=True)
            if attempt == tries:
                raise
            time.sleep(10 * attempt)


PARTIAL_SUFFIXES = (".incomplete", ".part", ".lock")


def dir_size_gb(path: Path) -> float:
    """Bytes of finished files only; partial downloads must not count as installed."""
    return sum(f.stat().st_size for f in path.rglob("*")
               if f.is_file() and not f.name.endswith(PARTIAL_SUFFIXES)) / 1e9


def download_url(task):
    import requests
    target = BASE / task["target"]
    target.mkdir(parents=True, exist_ok=True)
    dest = target / task["file"]
    if dest.exists() and dest.stat().st_size / 1e9 >= task["min_gb"]:
        print(f"[{task['id']}] exists, skipping ({dest.stat().st_size / 1e9:.2f} GB)", flush=True)
        return
    print(f"[{task['id']}] downloading {task['url']} ...", flush=True)

    def _dl():
        with requests.get(task["url"], stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    _check_cancel(task["id"])
                    if chunk:
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r  {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
            tmp.rename(dest)
        print(f"\n[{task['id']}] DONE ({dest.stat().st_size / 1e6:.1f} MB in {dest})", flush=True)

    run_with_retries(_dl, task["id"])


def download_hf(task):
    from huggingface_hub import hf_hub_download
    target = BASE / task["target"]
    target.mkdir(parents=True, exist_ok=True)
    if dir_size_gb(target) >= task["min_gb"]:
        print(f"[{task['id']}] exists, skipping ({dir_size_gb(target):.2f} GB)", flush=True)
        return
    if task["kind"] == "file":
        print(f"[{task['id']}] downloading {task['file']} ...", flush=True)

        def _dl():
            p = hf_hub_download(task["repo"], task["file"], local_dir=str(target), revision="main")
            got, dest = Path(p), target / task["file"]
            if got != dest:
                if dest.exists():
                    dest.unlink()
                got.rename(dest)
            return dest

        run_with_retries(_dl, task["id"])
    else:
        import fnmatch

        from huggingface_hub import list_repo_files
        print(f"[{task['id']}] snapshot {task['repo']} -> {target} ...", flush=True)
        allow = task.get("allow") or ["*"]
        all_files = run_with_retries(lambda: list_repo_files(task["repo"]), task["id"])
        files = [f for f in all_files if any(fnmatch.fnmatch(f, pat) for pat in allow)]
        print(f"[{task['id']}] {len(files)} files", flush=True)
        for rel in files:
            _check_cancel(task["id"])
            run_with_retries(
                lambda rel=rel: hf_hub_download(task["repo"], rel, local_dir=str(target), revision="main"),
                task["id"],
            )
    size = dir_size_gb(target)
    if size < task["min_gb"]:
        raise RuntimeError(f"downloaded only {size:.2f} GB, expected >= {task['min_gb']} GB")
    print(f"[{task['id']}] DONE ({size:.2f} GB in {target})", flush=True)


def download_civitai(version_id: str, target_dir: str, filename: str, token: str):
    """Download a Civitai model version by numeric version ID.

    Find the ID on the model page URL (.../models/<modelId>?modelVersionId=<ID>).
    Token: Civitai account Settings -> API keys (required by Civitai for downloads).
    """
    import requests
    target = BASE / target_dir
    target.mkdir(parents=True, exist_ok=True)
    dest = target / filename
    if dest.exists() and dest.stat().st_size > 100 * 1024 * 1024:
        print(f"[civitai:{version_id}] exists, skipping ({dest.stat().st_size / 1e9:.2f} GB)", flush=True)
        return
    url = f"https://civitai.com/api/download/models/{version_id}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    print(f"[civitai:{version_id}] downloading -> {dest} ...", flush=True)
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {done / 1e9:.2f}/{total / 1e9:.2f} GB", end="", flush=True)
    print(f"\n[civitai:{version_id}] DONE ({dest.stat().st_size / 1e9:.2f} GB)", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Download models into DiffusionServer/models/")
    ap.add_argument("models", nargs="*", help="model IDs (default: preset)")
    ap.add_argument("--preset", default="odysseus-default", choices=list(PRESETS),
                    help="download preset")
    ap.add_argument("--list", action="store_true", help="list models/presets and exit")
    ap.add_argument("--civitai", default="", help="Civitai model VERSION id to download")
    ap.add_argument("--target", default="", help="target subdir under models/ (civitai)")
    ap.add_argument("--name", default="", help="output filename (civitai)")
    ap.add_argument("--civitai-token", default=os.environ.get("CIVITAI_TOKEN", ""),
                    help="Civitai API token (or CIVITAI_TOKEN env)")
    args = ap.parse_args()

    # Pick up HF_TOKEN / PRISM_LLM_* saved from the WebUI Settings tab so a
    # gated repo works from the CLI without exporting anything manually.
    try:
        from unified import secrets as _secrets
        _secrets.load_and_apply()
    except Exception:
        pass

    if args.list:
        print("Models:", flush=True)
        for t in (TASKS + UPSCALE_TASKS + VLM_TASKS + OCR_TASKS +
                  AUDIO_TASKS + RESTORATION_TASKS):
            print(f"  {t['id']}: {t.get('repo', 'civitai')} [{t.get('license', '')}]", flush=True)
        print("Presets:", flush=True)
        for name, ids in PRESETS.items():
            print(f"  {name}: {', '.join(ids)}", flush=True)
        return

    if args.civitai:
        if not args.target or not args.name:
            ap.error("--civitai needs --target and --name")
        try:
            download_civitai(args.civitai, args.target, args.name, args.civitai_token)
        except Exception as e:  # noqa: BLE001
            print(f"[civitai:{args.civitai}] FAILED: {e}", flush=True)
            sys.exit(1)
        print("ALL DONE", flush=True)
        return

    ids = args.models or PRESETS[args.preset]
    all_tasks = (TASKS + UPSCALE_TASKS + VLM_TASKS + OCR_TASKS +
                 AUDIO_TASKS + RESTORATION_TASKS)
    unknown = [i for i in ids if i not in {t["id"] for t in all_tasks}]
    if unknown:
        print(f"Unknown model IDs: {', '.join(unknown)}", flush=True)
        sys.exit(2)
    BASE.mkdir(exist_ok=True)
    for t in all_tasks:
        if t["id"] not in ids:
            continue
        try:
            if t.get("kind") == "url":
                download_url(t)
            else:
                download_hf(t)
            # Build whatever the runtime needs (e.g. ONNX export) and mark ready.
            prepare_task(t)
        except Exception as e:  # noqa: BLE001
            print(f"[{t['id']}] FAILED: {e}", flush=True)
            FAILED.append(t["id"])
    if FAILED:
        print("FAILED:", ", ".join(FAILED), flush=True)
        sys.exit(1)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
