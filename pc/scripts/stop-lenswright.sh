#!/usr/bin/env bash
# =============================================================================
# Lenswright Server Stop Script (Linux / macOS)
# ============================================================================
# Stops the Lenswright server running on port 8101 (or LENSWRIGHT_PORT).
# ============================================================================

set -euo pipefail

PORT="${LENSWRIGHT_PORT:-8101}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Stopping Lenswright server on port $PORT..."

if command -v lsof &> /dev/null; then
    PIDS=$(lsof -ti:$PORT 2>/dev/null || true)
    if [ -z "$PIDS" ]; then
        echo -e "${YELLOW}No process found listening on port $PORT${NC}"
        exit 0
    fi
    echo "Found PID(s): $PIDS"
    for pid in $PIDS; do
        kill "$pid" 2>/dev/null || true
        echo -e "  ${GREEN}Sent SIGTERM to PID $pid${NC}"
    done
    sleep 2
    # Force kill if still running
    for pid in $PIDS; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
            echo -e "  ${RED}Force-killed PID $pid${NC}"
        fi
    done
    echo -e "${GREEN}Lenswright server stopped.${NC}"
else
    echo -e "${RED}lsof not found. Install lsof or stop the server manually.${NC}"
    exit 1
fi
