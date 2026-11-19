#!/usr/bin/env bash
# One-time setup: creates .venv INSIDE this folder (nothing is installed system-wide).
set -e
cd "$(dirname "$0")"
CUDA_TAG="cu126"
python3 --version
echo "=== 1/4 Creating virtual environment in $(pwd)/.venv ==="
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --no-cache-dir --upgrade pip
echo "=== 2/4 Installing PyTorch (CUDA $CUDA_TAG) ==="
python -m pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
echo "=== 3/4 Installing the other libraries ==="
python -m pip install --no-cache-dir -c constraints-local.txt -r requirements.txt
echo "=== 4/4 Registering the notebook kernel inside .venv ==="
python -m ipykernel install --prefix "$(pwd)/.venv" --name edgeai-local --display-name "Python (EdgeAI local)"
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
echo "Setup finished. Now run: bash run_jupyter.sh"
