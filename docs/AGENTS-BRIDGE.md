# Agent Bridge (MCP) — spec and plan

Status: proposed, not implemented. Decision log at the end.

## Problem

Prism can already generate images, upscale, OCR, transcribe, describe images and
run FFmpeg over an OpenAI-compatible HTTP API. Agent harnesses (Cindy, Claude
Code, Codex, OpenCode) cannot use any of it without hand-written `curl` calls,
because the API was shaped for a WebUI:

- every media endpoint takes a multipart upload and **returns raw base64 in the
  JSON body** (`unified_server.py:1035`, `:3194`), so an agent would paste
  megabytes of base64 into its context;
- long work (video upscale up to 900 frames, model installs) runs
  synchronously inside the request, so a client timeout of 60 s aborts it;
- there is no capability discovery an agent can trust, so it must guess model
  ids and gets `503 will-not-load` as a bare error.

Result: the server is capable, the agents are blind. Solved means: any harness,
pointed at one URL, can ask "what can you do right now", run any capability by
file path, get a file path back, and poll long jobs.

## Goals

1. One MCP endpoint, `http://<host>:8101/mcp`, speaking streamable HTTP, served
   by the existing Prism process (single source of truth for the contract).
2. Tool surface an agent can operate without guessing: capability discovery,
   path-in/path-out, honest errors, async jobs with progress.
3. First-class setup for four harnesses with one installer and one `doctor`
   check.
4. One skill document teaching *when* to use which capability, installable into
   every harness.
5. No capability reachable by an agent that is not also reachable from the
   WebUI. No new model engines in this project.

## Non-goals

- Video generation (text-to-video / image-to-video). Prism has no such engine;
  only `/v1/videos/upscale`. Tracked as a separate later phase with a ComfyUI
  backend, not part of this project's core.
- Routing to the other tools under `D:\AI` (MoneyPrinterTurbo, OpenMotion,
  AI_VideoEdit). Architecturally the same place, deliberately later.
- Android as an MCP *server*. Android stays an API-compatible Prism node; it
  may become an MCP *client* target later.
- Replacing the REST API, the WebUI, or the OpenAI compatibility layer.
- Multi-user, roles, or internet-facing deployment.

## Constraints

- `pc/AGENTS.md` rules apply: English only in code, docs, logs and commits; the
  two-product architecture is locked; honest `503` instead of faked output;
  checklist updated after every meaningful change.
- One resident heavy model. An agent must not silently evict the model a user
  is using in the WebUI.
- LAN-only, no authentication today. Anything an agent can reach over the
  network must be gated before LAN or Tailscale exposure.
- Python 3.12 venv at `pc/.venv` (or repo `.venv`); `mcp` is not installed yet.

## Verified facts

| Fact | Source |
|---|---|
| FastAPI app with an existing `lifespan` | `pc/unified_server.py:133`, `:159` |
| ~35 routes incl. `/v1/jobs`, `/v1/outputs/{rel}`, `/v1/models/catalog` | `pc/unified_server.py` route decorators |
| Media endpoints return base64 (`image`, `file`) | `:1035`, `:3194` |
| `_job_start()` is called only for transcription | `:2037` |
| Video upscale is synchronous, no job record | `:1051` |
| `ffmpeg/run` accepts free-form flags, blocks `-i` and shell metacharacters | `:3168` |
| No MCP anywhere in the repo | grep `mcp` in `pc/`, `docs/` |
| No video-generation route exists | route list |
| Cindy, Claude Code, Codex, OpenCode all support streamable HTTP MCP | harness configs on this machine |

## Approach

Mount a FastMCP app on the existing FastAPI process at `/mcp`, whose tools call
Prism's internal functions directly instead of looping back through HTTP. That
removes base64 from the path entirely (internal calls take and return file
paths), keeps one process and one service, and leaves `unified_server.py`'s
route behaviour unchanged.

The one technical unknown is lifespan composition: `FastMCP.streamable_http_app()`
manages its own session manager, and mounting a sub-app does not compose
lifespans. The existing `lifespan` must run it through an `AsyncExitStack`.
Prove this in Phase 0; if it fights back, fall back to serving the same app on
`:8103` from the same process (`Start-Prism.bat` gains one line) — the tool
layer does not change.

Blocking endpoints are `def`, not `async def`, so tool handlers must dispatch
them through `anyio.to_thread.run_sync`; otherwise an agent's upscale stalls the
event loop that also serves the WebUI.

## Tool surface

Eleven tools, grouped by artifact type with an `action` enum. Flat 1:1 mapping
of ~35 endpoints is rejected: agents pick wrong tools from large undifferentiated
sets, and most endpoints are one `action` value.

| Tool | Actions / purpose | Notes |
|---|---|---|
| `prism_status` | health, device, resident model, queue depth, free disk | cheap, safe to call often |
| `prism_models` | catalog with install state, size, licence, filter by kind | agents must not guess ids |
| `prism_model_install` | download / verify / cancel | consent-gated, multi-GB |
| `prism_image` | generate, edit, inpaint, outpaint, harmonize, colorize, restore, face-restore, upscale | path in, path out |
| `prism_video` | upscale, convert, trim, compress, extract-audio, info | upscale is async |
| `prism_audio` | transcribe, translate, speech, convert | `language` explicit |
| `prism_document` | ocr, extract (PDF/DOCX to text/Markdown) | |
| `prism_vision` | caption, describe, chat | VLM |
| `prism_text` | post-processing via `/v1/text/process` | |
| `prism_job` | list, status, cancel, wait | `wait` emits progress |
| `prism_artifact` | metadata, thumbnail, preview URL for an output | image block only when small |

Invariants the code enforces, not the prompt:

1. Never return base64 in a tool result. Return `{path, url, bytes, kind}` from
   the existing `_store_output()` / `_store_b64()` metadata.
2. Anything expected to exceed ~10 s returns a job id immediately.
3. A model download requires an explicit `confirm=true` parameter.
4. `503 will-not-load` becomes a typed error with the server's `hint` field
   preserved.
5. File arguments must resolve inside configured roots (`PRISM_MCP_ROOTS`).

## Harness configuration

One endpoint, four formats:

| Harness | Where |
|---|---|
| Cindy / Pi | `pi mcp add prism --url http://127.0.0.1:8101/mcp` (or `~/.pi/agent/mcp.json`) |
| Claude Code | `claude mcp add --transport http prism http://127.0.0.1:8101/mcp` |
| Codex | `~/.codex/config.toml` → `[mcp_servers.prism]`, `url = ...` |
| OpenCode | `~/.config/opencode/opencode.jsonc` → `mcp.prism`, `type: "remote"` |

Exposure guidance: `prism_status`, `prism_models`, `prism_image`, `prism_audio`
as `direct`; the rest `deferred`/`codemode` so the tool list stays small.

The skill (`agents/skills/prism/SKILL.md`) carries the judgement the tools
cannot: which model for a scanned Slovak invoice, when RealESRGAN beats
Z-Image, when to refuse an upscale on CPU, and the rule to call `prism_models`
before naming a model.

## Phases

**Phase 0 — prove the wedge.** `pc/mcp_server.py` with `prism_status` and
`prism_models`, mounted at `/mcp`, lifespan composed, `mcp[cli]` added to
requirements, plus `agents/` config snippets and a `doctor` that reports each
harness's connection state.
*Verify:* server starts; `curl` initialize against `/mcp` succeeds; Cindy, Claude
Code, Codex and OpenCode each list `prism_status` and return real host stats.
*Risk:* lifespan composition (mitigation: `:8103` fallback above).

**Phase 1 — capability surface.** `prism_image`, `prism_video`, `prism_audio`,
`prism_document`, `prism_vision`, `prism_text`, `prism_artifact`; path-based I/O;
base64 stripped from all results; typed errors.
*Verify:* an agent turns a supplied `testdata/` file into an output path for each
of the nine image actions, three audio actions, OCR and extract; results are
byte-identical to the equivalent REST call on the same input.

**Phase 2 — jobs.** Extend `_job_start`/`_job_update` coverage to video upscale
and model install; `prism_job wait` with progress notifications; cancellation.
*Verify:* a 900-frame upscale completes through `prism_job wait` under a 60 s
client timeout; cancel actually stops it and frees the model.

**Phase 3 — security.** Bearer token (`PRISM_TOKEN`), `PRISM_MCP_ROOTS` path
allowlist, per-tool flags for `ffmpeg/run`, and a single-heavy-job gate.
*Verify:* wrong token is rejected; a path outside the roots is refused with a
clear reason; a second heavy job queues instead of evicting the resident model.

**Phase 4 — distribution.** The skill, installer across four harnesses,
`doctor`, docs, `docs/CHECKLIST.md` entry.
*Verify:* fresh-shell `doctor` reports all green; a new harness is added by one
command.

**Phase 5 — video generation (separate project).** ComfyUI backend plus
`prism_video generate`; out of scope until Phases 0–3 are green.

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Lifespan composition fails | Phase 0 blocked | serve MCP on `:8103` from the same process |
| Agent evicts the user's resident model | user-visible breakage | single-heavy-job gate, `PRISM_MCP_ALLOW_MODEL_SWITCH` |
| `ffmpeg/run` + path inputs = arbitrary file read | data exfiltration | path roots, flag allowlist, token — before LAN exposure |
| CPU-only inference is slow | agent gives up or retries | `prism_status` reports device; tools state expected duration |
| Tool surface drifts from REST | silent divergence | one implementation, tests asserting REST/MCP parity |
| Blocking calls on the event loop | WebUI stalls | `to_thread` dispatch |

## Open questions

1. Camel case or snake case for tool names? Default: `snake_case`, matching MCP
   conventions everywhere.
2. Should `prism_artifact` inline small images as image blocks (better for
   visual verification) or always return paths? Default: inline under 1 MB.
3. Token in Phase 3 or a minimal shared-secret check in Phase 0? Default: Phase
   3, loopback-only until then.
