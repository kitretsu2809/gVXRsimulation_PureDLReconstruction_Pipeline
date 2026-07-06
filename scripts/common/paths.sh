#!/bin/bash
# Universal paths configuration for all scripts

# Get repo root (parent of scripts directory)
export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export DATA_DIR="${REPO_ROOT}/data"
export OUTPUTS_DIR="${REPO_ROOT}/outputs"
export SRC_DIR="${REPO_ROOT}/src"

# Sample paths
export SAMPLE_1_DIR="${DATA_DIR}/sample_1"
export SAMPLE_2_DIR="${DATA_DIR}/sample_2"

# Python path setup
export PYTHONPATH="${SRC_DIR}:${PYTHONPATH}"
