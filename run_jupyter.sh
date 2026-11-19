#!/usr/bin/env bash
# Starts Jupyter Lab with ALL caches/config kept inside this folder.
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] || { echo "Run: bash setup_linux.sh first"; exit 1; }
export JUPYTER_CONFIG_DIR="$PWD/.jupyter/config" JUPYTER_DATA_DIR="$PWD/.jupyter/data" JUPYTER_RUNTIME_DIR="$PWD/.jupyter/runtime"
export IPYTHONDIR="$PWD/.ipython" TORCH_HOME="$PWD/.cache/torch" MPLCONFIGDIR="$PWD/.cache/matplotlib" XDG_CACHE_HOME="$PWD/.cache" HF_HOME="$PWD/.cache/huggingface"
source .venv/bin/activate
jupyter lab --notebook-dir="$PWD"
