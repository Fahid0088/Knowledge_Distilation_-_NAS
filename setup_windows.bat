@echo off
REM ===== One-time setup: creates .venv INSIDE this folder (nothing is installed system-wide) =====
setlocal
cd /d "%~dp0"
set "CUDA_TAG=cu126"
echo Python used:
python --version || (echo Python was not found. Install Python 3.10-3.12 from python.org first. & pause & exit /b 1)

echo.
echo === 1/4 Creating virtual environment in "%cd%\.venv" ===
python -m venv .venv || (echo Could not create the environment. & pause & exit /b 1)
call ".venv\Scripts\activate.bat"
python -m pip install --no-cache-dir --upgrade pip

echo.
echo === 2/4 Installing PyTorch (CUDA %CUDA_TAG%) - large download, please wait ===
python -m pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/%CUDA_TAG% || (echo PyTorch install failed. & pause & exit /b 1)

echo.
echo === 3/4 Installing the other libraries ===
python -m pip install --no-cache-dir -c constraints-local.txt -r requirements.txt || (echo Library install failed. & pause & exit /b 1)

echo.
echo === 4/4 Registering the notebook kernel inside .venv ===
python -m ipykernel install --prefix "%cd%\.venv" --name edgeai-local --display-name "Python (EdgeAI local)"

echo.
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE - update the NVIDIA driver')"
echo.
echo Setup finished. Now double-click run_jupyter.bat
pause
