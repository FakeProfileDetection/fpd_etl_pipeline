#!/usr/bin/env bash
# Convenient activation script for the project environment

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment
if [[ -f "${SCRIPT_DIR}/.venv/bin/activate" ]]; then
  source "${SCRIPT_DIR}/.venv/bin/activate"

  # Load environment variables if .env exists
  if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    set -a
    source "${SCRIPT_DIR}/.env"
    set +a
    echo "Loaded environment variables from .env"
  fi

  echo "FPD ETL Pipeline environment activated (Python $(python --version 2>&1 | awk '{print $2}'))"
  echo "Run 'deactivate' to exit the environment"
else
  echo "Error: Virtual environment not found. Run ./setup.sh first."
  exit 1
fi
