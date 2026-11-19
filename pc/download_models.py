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
]

PRESETS = {
    "odysseus-default": [t["id"] for t in TASKS],
    "sd15": ["DreamShaper_8", "AbsoluteReality_1.8"],
    "sdxl": ["AnimagineXL_3.1", "JuggernautXL_v9"],
    "sdxl-turbo": ["SDXL_Turbo"],
    "upscale": ["RealESRGAN_x4v3"],
    "swinir": ["swinir-realsr-x4-upscale"],
    "zimage": ["z-image-turbo"],
    "ocr": ["nemotron-ocr-0", "granite-docling-258m", "qwen3-vl-2b"],
    "vlm": ["SmolVLM_500M"],
}

# VLM (V2): HuggingFaceTB SmolVLM-500M-Instruct (~1 GB, Apache-2.0).
# Snapshot like SD models; the server uses transformers AutoModelForImageTextToText.
VLM_TASKS = [
    {"id": "SmolVLM_500M", "kind": "snapshot", "repo": "HuggingFaceTB/SmolVLM-500M-Instruct",
     "target": "SmolVLM_500M", "min_gb": 0.8,
     "allow": ["*.safetensors", "*.json", "tokenizer/*", "preprocessor_config.json",
               "*.model", "*.txt", "chat_template.jinja"],
     "license": "Apache-2.0 (see repo card)"},
]

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
    {"id": "paddleocr-vl-1.6", "kind": "snapshot",
     "repo": "PaddlePaddle/PaddleOCR-VL-1.6", "target": "paddleocr-vl-1.6",
     "min_gb": 1.8,
     "allow": ["*.json", "*.safetensors", "*.txt", "*.model",
               "tokenizer/*", "chat_template*", "preprocessor_config.json"],
     "license": "Apache-2.0",
     "note": "PaddleOCR-VL 1.6 (0.96B): document parsing, 109 languages."},
    {"id": "glm-ocr", "kind": "snapshot",
     "repo": "zai-org/GLM-OCR", "target": "glm-ocr", "min_gb": 2.5,
     "allow": ["*.json", "*.safetensors", "*.txt", "*.model",
               "tokenizer/*", "chat_template*", "preprocessor_config.json"],
     "license": "MIT",
     "note": "GLM-OCR 1.33B: tables, formulas, 8 languages, MIT."},
    {"id": "qianfan-ocr", "kind": "snapshot",
     "repo": "baidu/Qianfan-OCR", "target": "qianfan-ocr", "min_gb": 9.0,
     "allow": ["*.json", "*.safetensors", "*.txt", "*.model",
               "tokenizer/*", "chat_template*", "preprocessor_config.json"],
     "license": "Apache-2.0",
     "note": "Qianfan-OCR 4.7B (192 languages, KIE, layout). ~9.5 GB; needs offload."},
    {"id": "firered-ocr", "kind": "snapshot",
     "repo": "FireRedTeam/FireRed-OCR", "target": "firered-ocr", "min_gb": 4.0,
     "allow": ["*.json", "*.safetensors", "*.txt", "*.model",
               "tokenizer/*", "chat_template*", "preprocessor_config.json"],
     "license": "Apache-2.0",
     "note": "FireRed-OCR: Qwen3-VL-2B fine-tuned for document parsing."},
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

# Audio: NVIDIA Nemotron 3.5 ASR. All three are optional downloads; the 0.6B
# runs comfortably on CPU, the 30B-A4B variants need far more memory than a
# 6-8 GB GPU (see the note on each entry) and are offered for completeness.
AUDIO_TASKS = [
    {"id": "nemotron-asr-0.6b", "kind": "snapshot",
     "repo": "nvidia/nemotron-3.5-asr-streaming-0.6b",
     "target": "nemotron-asr-0.6b", "min_gb": 2.4,
     "allow": ["config.json", "generation_config.json", "processor_config.json",
               "tokenizer.json", "tokenizer_config.json", "model.safetensors"],
     "note": "Nemotron 3.5 ASR 0.6B (40 languages, punctuated) - 2.55 GB, runs on CPU",
     "license": "OpenMDW-1.1 (see repo card)"},
    {"id": "nemotron-transcribe-30b-a4b-nvfp4", "kind": "snapshot",
     "repo": "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4",
     "target": "nemotron-transcribe-30b-a4b-nvfp4", "min_gb": 20.0,
     "allow": ["*.json", "*.safetensors", "*.jinja", "*.txt", "tokenizer/*",
               "llm_backbone/*"],
     "note": "Nemotron 3.5 Transcribe 30B-A4B NVFP4 - 21.6 GB; needs a Blackwell GPU or ~24 GB VRAM",
     "license": "OpenMDW-1.1 (see repo card)"},
    # Optional ASR engines (transformers 5.x classes, generic hf_asr.py).
    {"id": "qwen3-asr-0.6b", "kind": "snapshot",
     "repo": "Qwen/Qwen3-ASR-0.6B", "target": "qwen3-asr-0.6b",
     "min_gb": 1.7,
     "allow": ["*.json", "*.safetensors", "*.txt", "*.model",
               "tokenizer/*", "chat_template*", "preprocessor_config.json"],
     "license": "Apache-2.0",
     "note": "Qwen3-ASR 0.6B (30 languages, Chinese + European). ~2 s"},
    {"id": "qwen3-asr-1.7b", "kind": "snapshot",
     "repo": "Qwen/Qwen3-ASR-1.7B-hf", "target": "qwen3-asr-1.7b",
     "min_gb": 3.9,
     "allow": ["*.json", "*.safetensors", "*.txt", "*.model",
               "tokenizer/*", "chat_template*", "preprocessor_config.json"],
     "license": "Apache-2.0",
     "note": "Qwen3-ASR 1.7B (30 languages, higher accuracy). ~2 s"},
    {"id": "qwencleo-asr-1.7b", "kind": "snapshot",
     "repo": "mohammedaly22/QwenCleo-ASR", "target": "qwencleo-asr-1.7b",
     "min_gb": 3.9,
     "allow": ["*.json", "*.safetensors", "*.txt", "*.model",
               "tokenizer/*", "chat_template*", "preprocessor_config.json"],
     "license": "Apache-2.0",
     "note": "QwenCleo ASR 1.7B (Egyptian Arabic + English)."},
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
# GFPGAN: face restoration (CVPR 2021, Apache-2.0).
# SwinIR: super-resolution / restoration (ICCV 2021, Apache-2.0).
RESTORATION_TASKS = [
    {"id": "ddcolor-tiny", "kind": "file",
     "repo": "edgetools/ddcolor",
     "file": "ddcolor-tiny-fp16.onnx",
     "target": "ddcolor-tiny", "min_gb": 0.12,
     "license": "Apache-2.0 (see piddnad/DDColor)",
     "note": "DDColor-tiny colorization (ICCV 2023, ~130 MB fp16, 512x512). "
             "ONNX, runs on CPU/CUDA via onnxruntime. Apache-2.0."},
    {"id": "gfpgan-v1.4", "kind": "file",
     "repo": "sidkonduru/gfpgan-v1.4-onnx",
     "file": "gfpgan14-fp16.onnx",
     "target": "gfpgan-v1.4", "min_gb": 0.15,
     "license": "Apache-2.0 (see TencentARC/GFPGAN)",
     "note": "GFPGAN v1.4 face restoration (CVPR 2021, ~170 MB fp16, 512x512). "
             "ONNX, runs on CPU/CUDA via onnxruntime. Apache-2.0."},
    {"id": "swinir-realsr-x4", "kind": "file",
     "repo": "Heliosoph/swinir-onnx",
     "file": "swinir_realsr_x4.onnx",
     "target": "swinir-realsr-x4", "min_gb": 0.1,
     "license": "Apache-2.0 (see JingyunLiang/SwinIR)",
     "note": "SwinIR real-world super-resolution x4 (ICCV 2021, ~122 MB, "
             "64x64->256x256). ONNX, runs on CPU/CUDA via onnxruntime. "
             "Apache-2.0."},
    {"id": "instruct-pix2pix", "kind": "snapshot",
     "repo": "timbrooks/instruct-pix2pix",
     "target": "instruct-pix2pix", "min_gb": 3.0,
     "allow": ["*.safetensors", "*.json", "tokenizer/*", "scheduler/*",
               "feature_extractor/*", "model_index.json", "*.txt"],
     "license": "MIT (see timbrooks/instruct-pix2pix)",
     "note": "InstructPix2Pix (SD 1.5 based, ~5.8 GB, MIT). Instruction-based "
             "image restoration: restore, repair, colorize, reconstruct. "
             "Requires torch + diffusers on PC."},
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
]

FAILED = []


def run_with_retries(fn, label, tries=5):
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - network flakes, retry
            print(f"[{label}] attempt {attempt}/{tries} failed: {str(e)[:200]}", flush=True)
            if attempt == tries:
                raise
            time.sleep(10 * attempt)


def dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


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
    from huggingface_hub import hf_hub_download, snapshot_download
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
        except Exception as e:  # noqa: BLE001
            print(f"[{t['id']}] FAILED: {e}", flush=True)
            FAILED.append(t["id"])
    if FAILED:
        print("FAILED:", ", ".join(FAILED), flush=True)
        sys.exit(1)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
