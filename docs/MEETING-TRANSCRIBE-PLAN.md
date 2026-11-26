# Lenswright Feature Plan: Auto-Update & Meeting Transcription

## 1. Auto-Update from GitHub

### Current State
- Git remote: `https://github.com/DracieVajko/Prism.git`
- No update mechanism exists
- Models are downloaded separately via `download_models.py`

### Plan: `pc/unified/update_checker.py`

**Endpoint:** `GET /v1/update/check` and `POST /v1/update/apply`

**Check flow:**
1. Fetch latest commit SHA from `https://api.github.com/repos/DracieVajko/Prism/commits/main`
2. Compare with local `git rev-parse HEAD`
3. If different, return `{update_available: true, current: <sha>, latest: <sha>, commits_behind: N}`
4. Fetch commit messages between current and latest for changelog

**Apply flow (requires restart):**
1. `git stash` (preserve local changes)
2. `git pull origin main`
3. `git stash pop` (restore local changes, may conflict)
4. Re-run `pip install -r requirements.txt` if requirements changed
5. Restart server (via `os.execv` or signal to supervisor)

**WebUI integration:**
- Settings tab: "Check for updates" button
- Shows current version (git SHA short) and latest
- "Update now" button with progress
- Auto-check on startup (configurable, off by default)

**Android parity:**
- Check GitHub releases API for APK updates
- Download and install via Android package manager

### Safety
- Never auto-apply without user confirmation
- Backup `models.json` and `models/` before update
- Rollback on failure (git reset to previous SHA)
- Log all update actions

---

## 2. Meeting Transcription (call.md Integration)

### What call.md provides
- Dual-channel transcription (mic vs system audio)
- Real-time AI intelligence during calls
- Post-meeting summaries (overview, key points, action items)
- Markdown export with full transcript
- Webhook integration (n8n, Zapier, CRMs)
- MCP auto-triggering

### Lenswright Already Has
- `/v1/audio/transcriptions` - file-based transcription
- `/v1/audio/translations` - translation
- `/v1/realtime` - WebSocket realtime transcription
- `/v1/chat/completions` - LLM analysis
- `/v1/audio/speech` - TTS
- `/v1/ffmpeg/*` - audio processing

### Plan: Meeting Mode

#### Phase 1: Dual-Channel Capture
- Add `/v1/meetings/start` endpoint that accepts two audio streams (mic + system)
- Mix and transcribe both channels separately
- Label transcripts as "You" vs "Them"
- Store in meeting session

#### Phase 2: Live Intelligence
- During recording, periodically send transcript chunks to LLM
- Generate contextual suggestions (things to say, questions to ask)
- Detect information needs and trigger MCP tools
- Show conversation metrics (talk ratio, WPM, questions)

#### Phase 3: Post-Meeting Analysis
- After recording ends, generate:
  - Short narrative summary
  - Key points by topic (attributed to participants)
  - Action items (concrete next steps)
- Export as structured Markdown
- Optional webhook to n8n/Zapier/CRM

#### Phase 4: WebUI Meeting Tab
- New "Meeting" tab in WebUI
- Start/stop recording controls
- Live transcript display (dual-channel)
- Live assist panel
- Post-meeting summary display
- Export button (Markdown/PDF)

### Implementation Steps
1. Add meeting session management to `media_routes.py`
2. Add dual-channel audio buffering to `/v1/realtime` WebSocket
3. Add `/v1/meetings/{id}/summary` endpoint using LLM
4. Add `/v1/meetings/{id}/export` endpoint for Markdown
5. Add WebUI meeting tab with recording controls
6. Add optional webhook configuration in settings

### Privacy
- All processing local (Whisper + local LLM)
- No cloud dependency (unlike call.md which uses VideoDB)
- Meeting data stored in `output/meetings/`
