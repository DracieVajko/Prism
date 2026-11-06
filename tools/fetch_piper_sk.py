import urllib.request
from pathlib import Path

OUT = Path(r"D:\AI\tools\sk_samples")
OUT.mkdir(parents=True, exist_ok=True)

BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/sk/sk_SK/lili/medium/"
for name, dest in [("MODEL_CARD", "MODEL_CARD.txt"),
                   ("samples/speaker_0.mp3", "speaker_0.mp3")]:
    try:
        with urllib.request.urlopen(BASE + name, timeout=120) as r:
            data = r.read()
        p = OUT / dest
        p.write_bytes(data)
        print("ok", dest, len(data), "bytes")
    except Exception as e:
        print("FAIL", name, str(e)[:150])

card = OUT / "MODEL_CARD.txt"
if card.exists():
    print("\n--- MODEL_CARD ---")
    print(card.read_text(encoding="utf-8", errors="replace")[:1500])
