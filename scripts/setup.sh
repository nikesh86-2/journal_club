#!/bin/bash

# Journal Club Setup Script
# This script sets up the environment for the journal club pipeline

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to repo root
cd "$REPO_ROOT"

echo "=== Journal Club Setup ==="
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p cache/faiss_index
mkdir -p output/markdown
mkdir -p output/json
mkdir -p logs

# Check if .env exists, if not create from example
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "Please edit .env with your API keys and configuration"
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check if VLAB2 is available for shared components
if [ -d "../VLAB2" ]; then
    echo "VLAB2 found at ../VLAB2 - will use shared FAISS index and semantic search"
else
    echo "VLAB2 not found - will use standalone configuration"
fi

# Make scripts executable
chmod +x scripts/*.sh

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Edit .env with your API keys (S2_API_KEY, ENTREZ_EMAIL)"
echo "2. Configure topics in config/topics.yaml"
echo "3. Configure domains in config/domains.yaml"
echo "4. Run: ./scripts/run_journal_club.sh all"
