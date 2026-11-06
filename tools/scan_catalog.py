import re
from pathlib import Path

s = Path(r"D:\AI\DiffusionServer\pc\download_models.py").read_text(encoding="utf-8")
for m in re.finditer(r"^(TASKS|VLM_TASKS|UPSCALE_TASKS|AUDIO_TASKS)\s*=", s, re.M):
    line = s[:m.start()].count("\n") + 1
    print("line", line, ":", m.group(1))

print("\n--- kind occurrences ---")
for m in re.finditer(r'"kind":\s*"(\w+)"', s):
    line = s[:m.start()].count("\n") + 1
    print(" line", line, m.group(1))
