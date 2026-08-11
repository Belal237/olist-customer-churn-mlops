#!/usr/bin/env bash
set -euo pipefail

# Create directories required by tests or code that expect processed data paths
mkdir -p data/processed
mkdir -p data/raw
echo "Created data directories: data/processed and data/raw"
