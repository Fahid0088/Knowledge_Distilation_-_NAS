@echo off
REM Starts Jupyter Lab with ALL caches/config kept inside this folder.
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (echo Run setup_windows.bat first. & pause & exit /b 1)
set "JUPYTER_CONFIG_DIR=%cd%\.jupyter\config"
set "JUPYTER_DATA_DIR=%cd%\.jupyter\data"
set "JUPYTER_RUNTIME_DIR=%cd%\.jupyter\runtime"
set "IPYTHONDIR=%cd%\.ipython"
set "TORCH_HOME=%cd%\.cache\torch"
set "MPLCONFIGDIR=%cd%\.cache\matplotlib"
set "XDG_CACHE_HOME=%cd%\.cache"
set "HF_HOME=%cd%\.cache\huggingface"
call ".venv\Scripts\activate.bat"
jupyter lab --notebook-dir="%cd%"
