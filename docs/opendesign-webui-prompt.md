# Prompt for OpenDesign — redesign Lenswright WebUI

Paste this whole file into OpenDesign. Attach the two screenshots from the
current WebUI (Generate tab and Transcribe tab with the Live block) if they are
available in your client; the prompt is self-contained without them.

---

You are redesigning the **Lenswright PC WebUI**. Produce a new, production-ready
design and the minimal code change that implements it. Do not invent a product
that is not there.

## What exists today (the reference I showed you)

- Live server: `http://127.0.0.1:8101/webui` (FastAPI serves a single static file).
- Source: `D:\AI\Lenswright\pc\webui_unified.html` — plain HTML/CSS/vanilla JS,
  no framework, no build step, 133 element ids, 10 tabs.
- Header: `Lenswright  server OK · img models: 7 · loaded: DreamShaper_8 · audio: small`,
  plus `Refresh` and a `Light`/`Dark` toggle.
- Tabs, in order: Generate, Transcribe, Upscale, Caption, Inpaint, Edit, Tools,
  Outputs, Models, Settings.
- Generate tab: image-model select, prompt, negative prompt, resolution
  (256/512/768/custom), quality (low/medium/high), guidance, repeats, green
  Generate button, and a `copy`/`clear` console on the right.
- Transcribe tab: file picker, Whisper model select with per-model install
  state, language, format, 30 s split, Silero VAD, post-process and target
  language, green Transcribe button, then the new **Live transcription
  (localhost only)** block: two source checkboxes (Microphone, Shared tab
  audio), language select, Start live / Stop live, status line, microphone and
  device tail previews, and links to the two transcript files.
- The UI currently starts with `connecting...` and `(loading…)` before models
  arrive; after load the model select shows e.g. `DreamShaper_8 (local)`.
- The left form column is narrow (~500 px) and the right console column is
  mostly empty. This imbalance is visible in the screenshots.
- The Transcribe tab says `small is not installed (495 MB) – Download it…`
  while the header says `audio: small`. Those are different concepts
  (configured default vs installed weights) but read as a contradiction.

## Requirements (non-negotiable)

1. **Preserve every behavior and every endpoint.** All 10 tabs, every button,
   input and selector stay reachable and functional. Do not remove the Live
   transcription block or its localhost-only guard. Do not change the API
   contract in `pc/docs/UNIFIED-CONTRACT.md`.
2. **English-only UI text**, comments and log messages (project rule).
3. **No new dependency** unless marked `NEW DEPENDENCY` and justified. Prefer
   plain CSS/JS in the same single file; keep the no-build-step property.
4. **Stay in the repo's conventions**: one file, existing id naming, inline
   `<style>`/`<script>`, existing colors/theme variables. Minimal diffs only —
   show changed lines, not whole files.
5. **Honest states.** Never present a configured-but-not-installed model as
   ready; never show a fake success. Empty OCR/live output must be visibly
   different from a completed result.

## What to fix and add (start here, then audit the rest yourself)

- **Layout:** rebalance the form/console columns so the right side is useful at
  desktop widths; give the console a real empty state and sensible minimum
  height; keep the form readable at narrow widths.
- **Loading:** replace `connecting...` / `(loading…)` with explicit skeleton or
  status states; show installed/not-installed and loaded/unloaded distinctly.
- **Header:** separate "configured audio model" from "installed/loaded"; make
  `audio: small` and `small is not installed` impossible to misread.
- **Live block:** surface per-source skip reasons (mic denied, no tab audio,
  MediaProjection unavailable); make Idle / Starting / Running / Error
  unmistakable; show mic and device tails with timestamps; keep the separate
  transcript-file links prominent; disable Stop while idle and Start while
  running; keep the localhost-only explanation.
- **Accessibility:** keyboard path for every control, visible focus, ARIA
  roles/labels on tabs and status, WCAG 2.1 AA contrast, text scaling, and
  `prefers-reduced-motion` for tab/modal/state transitions.
- **Audit every tab:** Upscale, Caption, Inpaint, Edit, Tools, Outputs, Models
  and Settings must have real empty, loading, success and error states; no dead
  controls; model selectors populated from `/v1/models`; outputs list and
  download links working.
- **Micro-interactions:** tab switches, button states and status changes get
  short, restrained motion (with reduced-motion fallback); disabled states must
  be obvious without relying on color alone.

## Deliverables

Follow the output format in
`C:\Users\DracieVajko\Downloads\PC_Commands\AI\MySkillInstructionPromptsPresets\prompts\system\opendesign-ui-code-audit.md`
(sections 0–7: Context, Findings table, Visual polish and micro-interactions,
Functionality and logical integrity, Architecture and code quality, Refactoring
snippets as minimal diffs, Prioritized roadmap, Verification).

- Cite evidence: `pc/webui_unified.html:<line>` or a named element in the
  screenshots. Mark inferred statements as inferred; write `not verified` and
  how to verify where you cannot confirm something.
- Severity: critical / high / medium / low. Say plainly when something is only
  cosmetic.
- Include a concrete testing plan: start the server with
  `cd pc && python -m uvicorn unified_server:app --host 127.0.0.1 --port 8101`,
  open `/webui`, click through all 10 tabs, check the browser console, and run
  `cd pc && python -m pytest tests/test_webui_live.py -q`.

Do not claim measurements you did not observe. Do not paste entire files.
