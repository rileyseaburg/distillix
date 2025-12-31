#!/bin/bash
# =============================================================================
# Start OpenCode Server for Teacher Model Queries
# =============================================================================
#
# Starts the OpenCode server that provides access to teacher models:
#   - Azure AI Foundry (Claude)
#   - ZAI Coding Plan (GLM 4.7)
#   - MiniMax (M2.1)
#
# Usage:
#   ./scripts/start_server.sh [port]
#
# Copyright (c) 2025 Distillix. All Rights Reserved.
# =============================================================================

PORT=${1:-4096}
HOST="127.0.0.1"

echo "========================================"
echo "Starting OpenCode Server"
echo "========================================"
echo ""
echo "Server URL: http://${HOST}:${PORT}"
echo "API Docs: http://${HOST}:${PORT}/doc"
echo ""

# Check if opencode is installed
if ! command -v opencode &> /dev/null; then
    echo "Error: opencode CLI not found"
    echo "Install with: curl -fsSL https://opencode.ai/install | bash"
    exit 1
fi

# Check if port is already in use
if lsof -i:${PORT} > /dev/null 2>&1; then
    echo "Warning: Port ${PORT} is already in use"
    echo "Checking if it's an opencode server..."
    
    if curl -s "http://${HOST}:${PORT}/global/health" | grep -q "healthy"; then
        echo "OpenCode server already running on port ${PORT}"
        exit 0
    else
        echo "Another process is using port ${PORT}"
        exit 1
    fi
fi

# Start the server
echo "Starting server..."
opencode serve --port ${PORT} --hostname ${HOST} --print-logs &
SERVER_PID=$!

# Wait for server to start
echo "Waiting for server to start..."
for i in {1..30}; do
    if curl -s "http://${HOST}:${PORT}/global/health" | grep -q "healthy"; then
        echo ""
        echo "Server started successfully!"
        echo "PID: ${SERVER_PID}"
        echo ""
        echo "Available models can be listed with:"
        echo "  curl http://${HOST}:${PORT}/provider"
        echo ""
        echo "To stop the server:"
        echo "  kill ${SERVER_PID}"
        exit 0
    fi
    sleep 1
done

echo "Error: Server failed to start within 30 seconds"
kill ${SERVER_PID} 2>/dev/null
exit 1
