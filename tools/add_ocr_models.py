content = open('D:\\AI\\DiffusionServer\\pc\\download_models.py', 'r').read()

old = '''VLM_TASKS = [
    {"id": "SmolVLM_500M", "kind": "snapshot", "repo": "HuggingFaceTB/SmolVLM-500M-Instruct",
     "target": "SmolVLM_500M", "min_gb": 0.8,
     "allow": ["*.safetensors", "*.json", "tokenizer/*", "preprocessor_config.json",
               "*.model", "*.txt", "chat_template.jinja"],
     "license": "Apache-2.0 (see repo card)"},
]

# Audio: NVIDIA Nemotron 3.5 ASR'''

new = '''VLM_TASKS = [
    {"id": "SmolVLM_500M", "kind": "snapshot", "repo": "HuggingFaceTB/SmolVLM-500M-Instruct",
     "target": "SmolVLM_500M", "min_gb": 0.8,
     "allow": ["*.safetensors", "*.json", "tokenizer/*", "preprocessor_config.json",
               "*.model", "*.txt", "chat_template.jinja"],
     "license": "Apache-2.0 (see repo card)"},
]

# OCR: Xiaomi Nemotron-3.5-OCR-0 (Qwen3.5-0.8B based, ~1.67 GB)
OCR_TASKS = [
    {"id": "nemotron-ocr-0", "kind": "snapshot", "repo": "xiaomi-research/Xiaomi-OCR-0",
     "target": "nemotron-ocr-0", "min_gb": 2.0,
     "allow": ["*.safetensors", "*.json", "tokenizer/*", "tokenizer_config.json",
               "*.model", "*.txt", "chat_template.jinja", "generation_config.json"],
     "license": "Apache-2.0 (see repo card)",
     "note": "OCR model based on Qwen3.5-0.8B. Requires transformers>=4.51."},
]

# Audio: NVIDIA Nemotron 3.5 ASR'''

content = content.replace(old, new)

open('D:\\AI\\DiffusionServer\\pc\\download_models.py', 'w').write(content)
print("Done!")