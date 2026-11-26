#!/usr/bin/env bash
# =============================================================================
# Lenswright Server Start Script (Linux / macOS)
# ============================================================================
# Starts the Lenswright unified server on port 8101.
# Models are downloaded lazily on first use.
#
# Usage:
#   ./pc/scripts/start-lenswright.sh           # start in foreground
#   LENSWRIGHT_PORT=8102 ./pc/scripts/start-lenswright.sh  # custom port
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PORT="${LENSWRIGHT_PORT:-8101}"
HOST="${LENSWRIGHT_HOST:-127.0.0.1}"
PYTHON="${LENSWRIGHT_PYTHON:-python3}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================================="
echo "  Lenswright Server"
echo "=================================================="
echo "Working directory: $PC_DIR"
echo "Port: $PORT"
echo "Host: $HOST"
echo "Python: $PYTHON"
echo ""
if [ "$HOST" != "127.0.0.1" ] && [ "$HOST" != "localhost" ]; then
    echo -e "${YELLOW}WARNING: Binding to $HOST exposes an UNAUTHENTICATED API on the network.${NC}"
    echo -e "${YELLOW}         Only use this on a trusted LAN. Do NOT expose port $PORT to the internet.${NC}"
    echo ""
fi

# Check Python exists
if ! command -v "$PYTHON" &> /dev/null; then
    echo -e "${RED}Error: Python executable '$PYTHON' not found.${NC}"
    echo "Install Python 3.8+ or set LENSWRIGHT_PYTHON environment variable."
    exit 1
fi

# Check Python version
PY_VERSION=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 8 ]; }; then
    echo -e "${RED}Error: Python 3.8+ required, found $PY_VERSION${NC}"
    exit 1
fi

# Check if unified_server.py exists
if [ ! -f "$PC_DIR/unified_server.py" ]; then
    echo -e "${RED}Error: unified_server.py not found in $PC_DIR${NC}"
    echo "Make sure you are running this from the pc/ directory or set LENSWRIGHT_PYTHON correctly."
    exit 1
fi

# Check if models directory exists (create if not)
MODELS_DIR="$PC_DIR/models"
if [ ! -d "$MODELS_DIR" ]; then
    echo -e "${YELLOW}Warning: models directory not found, creating $MODELS_DIR${NC}"
    mkdir -p "$MODELS_DIR"
fi

# Check if output directory exists
OUTPUT_DIR="$PC_DIR/output"
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Kill any existing Lenswright server on this port
if command -v lsof &> /dev/null; then
    EXISTING_PID=$(lsof -ti:$PORT 2>/dev/null || true)
    if [ -n "$EXISTING_PID" ]; then
        echo -e "${YELLOW}Warning: Port $PORT is already in use (PID $EXISTING_PID). Killing existing process...${NC}"
        kill "$EXISTING_PID" 2>/dev/null || true
        sleep 2
    fi
fi

echo ""
echo "Starting Lenswright server..."
echo "----------------------------------------"

# Run the server
exec "$PYTHON" -m uvicorn unified_server:app --host "$HOST" --port "$PORT"
