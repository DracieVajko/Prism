# Lenswright UI audit and redesign report

## 0. Context

- **Scope:** the real Lenswright repository at `D:\AI\Lenswright`. PC target: `pc/webui_unified.html` (single static file, inline `<style>`/`<script>`, 10 tabs, no build step). Android target: the native app (`MainActivity.kt`, `res/layout/activity_main.xml`, `res/values/*`). The separate Open Design prototypes in this workspace were not changed.
- **Design source:** the active "Together AI" design system is bundled with Open Design, not stored in either product. The repository already had matching neutral tokens (`#F6F8FA`, `#FFFFFF`, `#0969DA`, `#30363D`); the work keeps and centralises those values rather than introducing a new palette.
- **Non-negotiables honoured:** all 10 tabs, all element ids, all button/input/selector behaviour, the Live transcription block and its localhost-only guard, the API contract in `pc/docs/UNIFIED-CONTRACT.md`, English-only UI text, no new dependency, single-file WebUI.
- **Evidence limits:** screenshots mentioned in the brief were not attached, so screenshot-based statements are absent. The server was not started and no browser session was opened in this turn; live behaviour is marked **not verified** below. Android was not compiled, per the standing instruction that the build is handled elsewhere.

## 1. Findings

Severity: critical / high / medium / low. "Fixed" means the source change is present on disk.

| # | Severity | Area | Evidence | Finding | Status |
|---|----------|------|----------|---------|--------|
| 1 | high | Header | `pc/webui_unified.html:63` `#status`; `refresh()` pre-change lines 533–535 | `audio: small` did not say whether weights were installed or loaded; the Transcribe tab could simultaneously say "small is not installed". | Fixed: status now prints `audio configured: … (installed/NOT INSTALLED)` and `audio loaded: …`. |
| 2 | high | Loading | `#status` line 63, `#model` line 86, `#settingsBox` line 302, `loadOutputs()` line 1525 | `connecting…` / `(loading…)` / `loading…` were bare text with no structure or progress meaning. | Fixed: explicit "Connecting to server…", "Loading models…", "Loading server settings…" with skeleton styling that clears on success or failure. |
| 3 | high | Layout | `main { grid-template-columns: 400px 1fr }` line 23; console block lines 338–345; right column line 347 | Narrow form column, console trapped in it, right column held only the gallery — mostly empty at desktop width. | Fixed: grid is `minmax(420px, 560px) minmax(0, 1fr)`; console moved to a real panel in the right column with min-height and an empty state. |
| 4 | medium | Live block | `#liveStatus` line 215; `startLiveSource()` lines 1180–1200; `setLiveButtons()` lines 928–938 | Idle/Starting/Running/Error were only prose in one line; per-source skip reasons (mic denied, no tab audio, audio context) appeared only in the log. | Fixed: `#liveState` badge with text state plus `#liveSkip` line; existing skip paths now record a visible reason. Guard and Start/Stop disable logic untouched. |
| 5 | medium | Icons | Header buttons line 64–65, console links lines 341–342, `btnTxt` line 107, `btnTr` line 200 | No consistent icon family; controls were text-only or used inconsistent colours. | Fixed: one inline-SVG family (24-unit grid, 1.75 stroke, `currentColor`, no fill) on refresh, theme, copy, clear, Generate, Transcribe, Live start/stop, output refresh/delete. |
| 6 | medium | Accessibility | Tabs lines 70–81 had `role="tab"` + `aria-selected` but no panel relationship | No `aria-controls` / `tabpanel` roles; disabled state relied on opacity alone. | Fixed: `initTabs()` wires `aria-controls`, `role="tabpanel"`, `aria-labelledby`; disabled styling adds `not-allowed` cursor; `prefers-reduced-motion` disables animation. |
| 7 | low | Console | `copyLog()` line 369 | Clipboard write had no success or failure feedback and could silently fail. | Fixed: honest success/failure message in the console. |
| 8 | low | Outputs | `#gallery` line 348 | Empty gallery had no explanation. | Fixed: `#galleryEmpty` empty state, hidden by `addCard()` on first result. |
| 9 | low | Destructive action | line 275 "Cleanup all" | Label understated a delete-all operation (still confirm-gated). | Fixed: renamed "Delete all"; confirmation retained. |
| 10 | high | Android role UI | `activity_main.xml:237–280`; `MainActivity.kt:787–797` | Local/Remote/Auto existed but read as a generic setting; it did not say which device was the server or that external harnesses can call the API. | Fixed: role-specific status text plus an endpoint note in Settings. No ids or handlers changed. |
| 11 | medium | Android visuals | `themes.xml:3–12` (comment contained a non-English word), ten `cardCornerRadius="12dp"` occurrences | Theme hardcoded chrome colours and violated the English-only rule; 12dp corners diverged from the WebUI's 6–8px geometry. | Fixed: theme references colour tokens, comment rewritten in English, corners 8dp. |
| 12 | low | Android icons | only `ic_chevron_right.xml` existed | Primary actions had no icons. | Fixed: six outline drawables with a shared 1.75 stroke on Start, Add model, Generate, Save image, Transcribe, Copy. |
| 13 | medium | Harness parity | `ApiServer.kt` routes (`/v1/images/*`, `/v1/audio/transcriptions`, `/v1/models`) | Code exposes the endpoints; real reachability from another device was not exercised. | **Not verified.** Verify with the plan in section 7. |
| 14 | medium | Screenshots | referenced in the brief, not attached | Column imbalance was inferred from the file and the written description, not observed in a screenshot. | Marked as inferred. |

## 2. Visual polish and micro-interactions

- One icon family everywhere: outline, 24×24, 1.75 stroke, consistent caps and joins, sized through one class (`.ico` on the web, 24dp drawables on Android). No mixed filled/outlined glyphs at the same level.
- Anti-AI pass: removed the solid blue active-tab fill in favour of an accent border plus weight, replaced ad-hoc hex button colours with theme tokens, removed the empty-right-column template, and added honest empty states instead of decorative cards. No gradients, glows or invented metrics were introduced.
- Motion is limited to a subtle skeleton pulse and existing 150ms button transitions; `prefers-reduced-motion: reduce` removes both. State is conveyed by text and shape (badge + disabled cursor), not colour alone.
- Radius system unified at 4px for small controls and 8px for cards/panels on both platforms.

## 3. Functionality and logical integrity

- WebUI: `TABS`, every `id`, every `onclick` handler and every fetch target are unchanged. `showTab()` still gates the image-model selector to image tabs. The Live block still refuses to start outside a secure context, disables Stop while idle and Start while running, drains pending chunks on stop, and keeps both transcript-file links.
- Android: only resource styling, icon attributes, and role wording changed; no activity ids, listeners, routes or engine calls were removed.
- Honest states: configured ≠ installed ≠ loaded; Empty ≠ completed; failure text names the cause. The clipboard helper reports inability instead of pretending success.
- Not verified: live server responses, browser rendering, Android build, and the pytest suite. No test result is claimed in this report.

## 4. Architecture and code quality

- The WebUI stays one self-contained HTML file with inline style and script; no framework, package or build step was added. New UI state lives in small helpers (`setLivePhase`, `setLiveSkip`, `resetLiveSkip`, `initTabs`) next to the functions that use them.
- Theme values are centralised in `:root` / `:root.light` for the WebUI and in `colors.xml` + `themes.xml` for Android, so both surfaces can be restyled from one place.
- The API contract and the server implementation were not touched.

## 5. Refactoring snippets (minimal diffs)

```diff
- main { display: grid; grid-template-columns: 400px 1fr; gap: 16px; padding: 16px; }
+ main { display: grid; grid-template-columns: minmax(420px, 560px) minmax(0, 1fr);
+        gap: 16px; padding: 16px; align-items: start; }
```

```diff
- <span id="status">connecting…</span>
+ <span id="status" class="skeleton" role="status" aria-live="polite">Connecting to server…</span>
```

```diff
- st.innerHTML = 'server <b class="ok">OK</b> · img models: '+imageModels.length+' · loaded: '+…
-   +' · audio: '+esc(h.audio_model||m.default_audio_model||'?')+(h.audio_loaded?' (loaded)':'');
+ const configuredAudio = h.audio_model || m.default_audio_model || '?';
+ const audioEntry = MODELS.find(x => x.id === configuredAudio);
+ const audioState = audioEntry ? (audioEntry.installed ? 'installed' : 'NOT INSTALLED')
+   : 'install state unknown';
+ st.innerHTML = 'server <b class="ok">OK</b> · img models: '+imageModels.length
+   +' · image loaded: '+esc(h.current_model||h.model||'none (lazy)')
+   +' · audio configured: '+esc(configuredAudio)+' ('+audioState+')'
+   +' · audio loaded: '+esc(h.audio_loaded ? (h.audio_model||configuredAudio) : 'none');
```

```diff
- function setLiveStatus(msg){ const el = …; if (el) el.textContent = msg; log('live: ' + msg); }
+ function setLiveStatus(msg){
+   const el = document.getElementById('liveStatus'); if (el) el.textContent = msg;
+   let phase = 'idle';
+   if (/fail|error|denied|could not|no live source|warning|unavailable|needs localhost/i.test(msg)) phase = 'error';
+   else if (/start|creating|connecting/i.test(msg)) phase = 'starting';
+   else if (/captur|live:/i.test(msg)) phase = 'running';
+   setLivePhase(phase); log('live: ' + msg);
+ }
```

```diff
- <item name="colorPrimary">#2F81F7</item>
- <!-- MaterialComponents theme: required for MaterialCardView/MaterialButton
-      (AppCompat theme crashuje na ThemeEnforcement.checkMaterialTheme). -->
+ <!-- MaterialComponents is required by MaterialCardView and MaterialButton. -->
+ <item name="colorPrimary">@color/lenswright_accent</item>
```

```diff
- app:cardCornerRadius="12dp"        (ten occurrences)
+ app:cardCornerRadius="8dp"
```

```diff
- else -> "Local mode — offline, no PC needed"
+ else -> "Local mode — this phone is the server (offline)"
```

## 6. Prioritized roadmap

1. **P0 — Verify the live WebUI** with the commands in section 7. Confirm the status line, the moved console, all 10 tabs, and Live start/stop on localhost.
2. **P1 — Finish state coverage in secondary tabs.** A few secondary loaders still use generic text (for example the model-search box in `findModels()`); replace them with the same skeleton/empty-state pattern.
3. **P1 — Check harness parity from a second device** (phone or another PC) against the phone server on port 8101, especially `/v1/images/generations` and `/v1/audio/transcriptions`.
4. **P2 — Build the Android app on a machine with the SDK** and check the new drawables and theme attributes resolve in both light and dark variants.
5. **P3 — Extend the icon family** to the remaining secondary buttons (FFmpeg operations, model manager rows) using the same outline geometry.

## 7. Verification

Not executed in this turn. Run, in order:

```sh
cd pc && python -m uvicorn unified_server:app --host 127.0.0.1 --port 8101
# open http://127.0.0.1:8101/webui
# click through all 10 tabs; watch the browser console for errors
cd pc && python -m pytest tests/test_webui_live.py -q
```

Additional checks for the Android half:

```sh
cd android && .\gradlew.bat :app:assembleDebug
# on a phone on the same LAN: curl http://<phone-ip>:8101/health
# curl http://<phone-ip>:8101/v1/models
# POST /v1/images/generations and /v1/audio/transcriptions from another harness
```

Expected results: no console errors; every tab renders; status distinguishes configured/installed/loaded; Live Start is disabled outside localhost; the console and gallery show meaningful empty states; the Android build succeeds and external clients can reach the phone server.

---

## 8. Verification addendum (OpenCode, 2026-11-27)

This file did not exist on disk when OpenCode checked; it was recreated from the
report text supplied by the user, with the results below appended.

Executed and observed:

- `cd pc && python -m pytest tests/test_webui_live.py -q` → **3 passed**.
- `cd pc && python -m pytest -q` → **232 passed, 3 skipped** (3 pre-existing
  deprecation warnings). No regressions from the WebUI change.
- Browser at `http://127.0.0.1:8101/webui`: reloaded with cache disabled;
  **0 console errors**. All 10 tabs have `aria-controls` pointing to real
  `role="tabpanel"` elements, exactly one selected, others hidden.
- Header visually reads:
  `server OK · img models: 7 · image loaded: DreamShaper_8 · audio configured: small (NOT INSTALLED) · audio loaded: none`
  — finding 1 verified fixed.
- Console moved to the right column with "No activity yet…" empty state;
  outputs panel shows "No outputs yet…". Layout rebalance visually confirmed.
- Live block shows an `IDLE` badge and `Skipped sources: none.`; Stop live is
  disabled while idle. Finding 4 verified.
- `cd android && .\gradlew.bat :app:assembleDebug` → **BUILD SUCCESSFUL**
  (twice: before and after the text correction below).

Corrected during verification:

- Finding 10's new status wording was wrong about Auto mode. `ApiServer.kt`
  forwards **remote first** in Auto mode, so `Auto mode — phone first, then PC
  server` and the layout text `Auto tries the phone first, then the PC` were
  inaccurate. Both now say the PC is tried first with a phone fallback.
- "Remote forwards heavy jobs" was misleading (it forwards all non-live
  requests); reworded to "forwards requests".

Still not verified: real reachability from a second device (harness parity),
Live capture with microphone/MediaProjection (no device attached), and Live
Start behavior on a non-localhost origin.
