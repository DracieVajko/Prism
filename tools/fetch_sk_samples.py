import json
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(r"D:\AI\tools\sk_samples")
OUT.mkdir(parents=True, exist_ok=True)

url = ("https://datasets-server.huggingface.co/rows?dataset="
       + urllib.parse.quote("google/fleurs", safe="")
       + "&config=sk_sk&split=test&offset=0&length=4")
print("GET", url)
with urllib.request.urlopen(url, timeout=120) as r:
    data = json.load(r)

rows = data.get("rows", [])
print("rows:", len(rows))
for i, row in enumerate(rows):
    rec = row["row"]
    audio = rec.get("audio") or []
    if isinstance(audio, list) and audio:
        audio = audio[0]
    src = audio.get("src") if isinstance(audio, dict) else None
    transcript = rec.get("transcription") or rec.get("raw_transcription") or ""
    if not src:
        print(i, "no audio src; keys:", list(rec)[:10])
        continue
    dest = OUT / f"sk_fleurs_{i}.wav"
    with urllib.request.urlopen(src, timeout=120) as r2:
        dest.write_bytes(r2.read())
    print(f"  {dest.name}  {dest.stat().st_size} bytes  ref={transcript[:80]!r}")
