"""Prove whether the earlier 'mangled diacritics' were real or a console artifact.

The old log line was written through PowerShell redirection on a cp1250 console.
If decoding those bytes as cp1250 yields correct Slovak, the model output was
always fine and only the console mangled it.
"""
from pathlib import Path

log = Path(r"D:\AI\tools\gemma_final.log")
if not log.exists():
    log = Path(r"C:\Users\DracieVajko\AppData\Local\Temp\gemma_final.log")

raw = log.read_bytes()
# take every line that contains cleanup output
lines = [ln for ln in raw.split(b"\n") if b"cleanup" in ln]
print("lines:", len(lines))
for ln in lines[:3]:
    print("\nRAW BYTES:", ln[:160])
    for enc in ("utf-8", "cp1250", "cp852", "cp437"):
        try:
            print(f"  {enc:>6}: {ln.decode(enc, 'strict')[:150]}")
        except UnicodeDecodeError as e:
            print(f"  {enc:>6}: <decode error {e.start}>")
