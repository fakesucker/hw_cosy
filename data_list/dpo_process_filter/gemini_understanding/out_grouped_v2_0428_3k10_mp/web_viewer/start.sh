#!/bin/bash
# DPO Audio Viewer - Start Script
# Usage: bash start.sh [port]

PORT=${1:-8765}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo "  DPO Audio Comparison Viewer"
echo "============================================================"

# Auto-detect public IPv6
IPV6=$(ip -6 addr show ppp0 2>/dev/null | grep 'scope global' | grep -v deprecated | head -1 | awk '{print $2}' | cut -d/ -f1)
if [ -z "$IPV6" ]; then
    IPV6=$(ip -6 addr show tun0 2>/dev/null | grep 'scope global' | head -1 | awk '{print $2}' | cut -d/ -f1)
fi

# Auto-detect LAN IPv4
IPV4=$(hostname -I 2>/dev/null | awk '{print $1}')

echo ""
if [ -n "$IPV6" ]; then
    echo "  Public IPv6:  http://[${IPV6}]:${PORT}"
fi
if [ -n "$IPV4" ] && [ "$IPV4" != "127.0.0.1" ]; then
    echo "  LAN   IPv4:   http://${IPV4}:${PORT}"
fi
echo "  Local:        http://127.0.0.1:${PORT}"
echo ""
echo "============================================================"
echo ""

# Kill existing instance
pkill -f "python3.*server.py" 2>/dev/null
sleep 1

cd "$SCRIPT_DIR"
python3 server.py &
sleep 2
echo ""
echo "Server started. Press Ctrl+C or close terminal to stop."
wait
