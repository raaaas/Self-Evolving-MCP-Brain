#!/usr/bin/env bash
# Run both servers in separate terminals with the project venv activated.
# Frontend on port 3000, backend on port 8000.
#
# Usage: bash run_servers.sh
# Requirements: gnome-terminal or xterm

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_ACTIVATE="$SCRIPT_DIR/.venv/bin/activate"
PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "ERROR: .venv not found at $SCRIPT_DIR/.venv" >&2
    exit 1
fi

# Build the command that will run inside each terminal.
# source activate, then run the server with the venv python.
backend_cmd="cd '$SCRIPT_DIR' && source '$VENV_ACTIVATE' && $PYTHON '$SCRIPT_DIR/ui_server.py'; exec \$SHELL"
frontend_cmd="cd '$SCRIPT_DIR' && source '$VENV_ACTIVATE' && $PYTHON '$SCRIPT_DIR/frontend_server.py'; exec \$SHELL"

# Try to open two separate terminals.
launch_term() {
    local title="$1" cmd="$2"
    if command -v gnome-terminal &>/dev/null; then
        gnome-terminal --title="$title" -- bash -c "$cmd" &
    elif command -v xterm &>/dev/null; then
        xterm -title "$title" -e bash -c "$cmd" &
    elif command -v konsole &>/dev/null; then
        konsole --new-tab -p tabtitle="$title" -e bash -c "$cmd" &
    else
        echo "No supported terminal found (gnome-terminal/xterm/konsole)."
        echo "Run manually in two terminals:"
        echo "  source $VENV_ACTIVATE && $PYTHON $SCRIPT_DIR/ui_server.py"
        echo "  source $VENV_ACTIVATE && $PYTHON $SCRIPT_DIR/frontend_server.py"
        exit 1
    fi
}

launch_term "MCP-Brain Backend :8000" "$backend_cmd"
echo "Backend terminal spawning (port 8000)..."

launch_term "MCP-Brain Frontend :3000" "$frontend_cmd"
echo "Frontend terminal spawning (port 3000)..."

sleep 1
echo "Done. Two terminals should be open."
