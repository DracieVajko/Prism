#!/usr/bin/env bash
# =============================================================================
# Prism Server Installer (Linux / macOS)
# ============================================================================
# Installs Prism as a system service that starts automatically at boot/login.
#
# On Linux:   creates a systemd user service (~/.config/systemd/user/prism.service)
# On macOS:   creates a launchd agent (~/Library/LaunchAgents/com.prism.server.plist)
#
# Usage:
#   ./pc/scripts/install-prism-service.sh              # install and start
#   ./pc/scripts/install-prism-service.sh --uninstall  # remove
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PORT="${PRISM_PORT:-8101}"
PYTHON="${PRISM_PYTHON:-python3}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[prism]${NC} $*"; }
warn() { echo -e "${YELLOW}[prism]${NC} $*"; }
err()  { echo -e "${RED}[prism]${NC} $*" >&2; }

# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--uninstall" ]; then
    OS="$(uname -s)"
    if [ "$OS" = "Darwin" ]; then
        PLIST="$HOME/Library/LaunchAgents/com.prism.server.plist"
        if [ -f "$PLIST" ]; then
            launchctl unload "$PLIST" 2>/dev/null || true
            rm -f "$PLIST"
            log "Removed launchd agent."
        else
            warn "No launchd agent found."
        fi
    else
        if systemctl --user list-unit-files 2>/dev/null | grep -q '^prism.service'; then
            systemctl --user disable --now prism.service 2>/dev/null || true
            rm -f "$HOME/.config/systemd/user/prism.service"
            systemctl --user daemon-reload 2>/dev/null || true
            log "Removed systemd user service."
        else
            warn "No systemd user service found."
        fi
    fi
    exit 0
fi

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
if ! command -v "$PYTHON" &> /dev/null; then
    err "Python '$PYTHON' not found. Install Python 3.8+ or set PRISM_PYTHON."
    exit 1
fi

if [ ! -f "$PC_DIR/unified_server.py" ]; then
    err "unified_server.py not found in $PC_DIR"
    exit 1
fi

mkdir -p "$PC_DIR/logs"

log "Prism installer"
log "  repo   : $PC_DIR"
log "  python : $PYTHON"
log "  port   : $PORT"
echo ""

# ---------------------------------------------------------------------------
# Linux - systemd user service
# ---------------------------------------------------------------------------
if [ "$(uname -s)" = "Linux" ]; then
    if ! command -v systemctl &> /dev/null; then
        err "systemctl not found; this Linux does not use systemd."
        exit 1
    fi

    UNIT_DIR="$HOME/.config/systemd/user"
    mkdir -p "$UNIT_DIR"

    cat > "$UNIT_DIR/prism.service" <<EOF
[Unit]
Description=Prism unified server (images, audio, OCR, VLM, ffmpeg)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$PC_DIR
ExecStart=$PYTHON -m uvicorn unified_server:app --host 0.0.0.0 --port $PORT
Restart=on-failure
RestartSec=5
Environment=PRISM_PORT=$PORT
StandardOutput=append:$PC_DIR/logs/service.out.log
StandardError=append:$PC_DIR/logs/service.err.log

[Install]
WantedBy=default.target
EOF

    systemctl --user daemon-reload
    systemctl --user enable prism.service
    systemctl --user restart prism.service

    log "systemd user service 'prism' installed and started."
    log "  status : systemctl --user status prism"
    log "  logs   : tail -f $PC_DIR/logs/service.out.log"
    log "  stop   : systemctl --user stop prism"

    # Enable lingering so the service starts at boot, not just at login.
    if command -v loginctl &> /dev/null; then
        if loginctl enable-linger "$USER" 2>/dev/null; then
            log "  boot start enabled (loginctl lingering)."
        else
            warn "  Could not enable lingering; service starts at login only."
            warn "  Run: sudo loginctl enable-linger $USER"
        fi
    fi

# ---------------------------------------------------------------------------
# macOS - launchd agent
# ---------------------------------------------------------------------------
elif [ "$(uname -s)" = "Darwin" ]; then
    PLIST_DIR="$HOME/Library/LaunchAgents"
    PLIST="$PLIST_DIR/com.prism.server.plist"
    mkdir -p "$PLIST_DIR"

    cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.prism.server</string>
    <key>WorkingDirectory</key>
    <string>$PC_DIR</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PYTHON</string>
        <string>-m</string>
        <string>uvicorn</string>
        <string>unified_server:app</string>
        <string>--host</string>
        <string>0.0.0.0</string>
        <string>--port</string>
        <string>$PORT</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>StandardOutPath</key>
    <string>$PC_DIR/logs/service.out.log</string>
    <key>StandardErrorPath</key>
    <string>$PC_DIR/logs/service.err.log</string>
</dict>
</plist>
EOF

    launchctl unload "$PLIST" 2>/dev/null || true
    launchctl load "$PLIST"

    log "launchd agent 'com.prism.server' installed and started."
    log "  status : launchctl list | grep prism"
    log "  logs   : tail -f $PC_DIR/logs/service.out.log"
    log "  stop   : launchctl unload $PLIST"

else
    err "Unsupported OS: $(uname -s)"
    exit 1
fi

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
echo ""
log "Waiting for the server to answer on port $PORT..."
for i in $(seq 1 30); do
    if command -v curl &> /dev/null; then
        if curl -fsS "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
            log "Health check OK: http://127.0.0.1:$PORT/health"
            log "WebUI:            http://127.0.0.1:$PORT/webui"
            exit 0
        fi
    else
        if "$PYTHON" -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=2)" 2>/dev/null; then
            log "Health check OK: http://127.0.0.1:$PORT/health"
            exit 0
        fi
    fi
    sleep 1
done

warn "Server did not answer within 30 s. Check the logs:"
warn "  $PC_DIR/logs/service.err.log"
