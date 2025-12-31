#!/bin/bash
# =============================================================================
# Distillix Training Script
# =============================================================================
#
# Full training pipeline:
#   1. Start OpenCode server (if not running)
#   2. Generate training data (if not exists)
#   3. Train BitNet student model
#
# Usage:
#   ./scripts/train.sh [--generate] [--config path/to/config.json]
#
# Copyright (c) 2025 Distillix. All Rights Reserved.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
GENERATE_DATA=false
CONFIG_PATH=""
OPENCODE_PORT=4096

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --generate)
            GENERATE_DATA=true
            shift
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --port)
            OPENCODE_PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Distillix Training Pipeline"
echo "========================================"
echo ""
echo "Project directory: $PROJECT_DIR"
echo ""

# Change to project directory
cd "$PROJECT_DIR"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Check Python dependencies
echo "Checking dependencies..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}')" || {
    echo "Error: PyTorch not installed. Run ./scripts/setup_cuda.sh first."
    exit 1
}

# Check GPU
echo "Checking GPU..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('Warning: No GPU detected. Training will be slow.')
"

# Check/Start OpenCode server
echo ""
echo "Checking OpenCode server..."
if curl -s "http://127.0.0.1:${OPENCODE_PORT}/global/health" | grep -q "healthy"; then
    echo "OpenCode server running on port ${OPENCODE_PORT}"
else
    echo "Starting OpenCode server..."
    ./scripts/start_server.sh ${OPENCODE_PORT} &
    sleep 5
    
    if ! curl -s "http://127.0.0.1:${OPENCODE_PORT}/global/health" | grep -q "healthy"; then
        echo "Error: Failed to start OpenCode server"
        exit 1
    fi
fi

# Generate training data if requested or if data doesn't exist
TRAIN_DATA="data/distillation/train.jsonl"
if [ "$GENERATE_DATA" = true ] || [ ! -f "$TRAIN_DATA" ]; then
    echo ""
    echo "========================================"
    echo "Generating Training Data"
    echo "========================================"
    echo ""
    
    # Create data directory
    mkdir -p data/distillation
    
    # Run data generation
    python3 -m foundry.generate \
        --builtin \
        --category all \
        --output "$TRAIN_DATA" \
        --port ${OPENCODE_PORT}
    
    echo ""
    echo "Training data generated: $TRAIN_DATA"
    
    # Show sample count
    SAMPLE_COUNT=$(wc -l < "$TRAIN_DATA")
    echo "Samples: $SAMPLE_COUNT"
fi

# Verify training data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: Training data not found at $TRAIN_DATA"
    echo "Run with --generate to create training data"
    exit 1
fi

echo ""
echo "========================================"
echo "Starting Training"
echo "========================================"
echo ""

# Run training
if [ -n "$CONFIG_PATH" ]; then
    python3 -m smelter.train --config "$CONFIG_PATH"
else
    python3 -m smelter.train
fi

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo ""
echo "Model saved to: artifacts/students/"
echo ""
