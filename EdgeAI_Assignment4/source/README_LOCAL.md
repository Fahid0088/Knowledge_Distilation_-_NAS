# Assignment 4 - local version (RTX 4050)

## Folder contents
- `Assignment4_KD_NAS_EdgeAI_Local.ipynb` - the whole assignment (Q1 distillation + Q2 NAS)
- `setup_windows.bat` / `setup_linux.sh` - one-time setup (creates `.venv` in this folder)
- `run_jupyter.bat` / `run_jupyter.sh` - starts Jupyter with all caches kept in this folder
- `requirements.txt`

## Steps (Windows)
1. Install Python 3.10, 3.11 or 3.12 from python.org (tick "Add python.exe to PATH") and make sure your NVIDIA driver is up to date.
2. Double-click `setup_windows.bat` (about 10-20 min, ~3 GB download). At the end it must print `CUDA available: True`.
3. Double-click `run_jupyter.bat`, open the notebook, set the kernel to `Python (EdgeAI local)` if asked.
4. Run all cells. First do a test with `QUICK_TEST = True`, then set it to `False` for the real run.

## Linux
`bash setup_linux.sh`, then `bash run_jupyter.sh`.

## If the run stops
Run all cells again. Finished models are loaded, a half-trained model continues from its last epoch, the NAS search continues from its last trial.

## Where is everything stored? (all inside this folder)
| Folder | Content | Size |
|---|---|---|
| `.venv/` | Python environment + PyTorch | ~5-7 GB |
| `data/` | CIFAR-10 / CIFAR-100 | ~0.7 GB |
| `.cache/` | pretrained ResNet-50 weights, matplotlib cache | ~0.1 GB |
| `.jupyter/`, `.ipython/` | Jupyter settings | small |
| `EdgeAI_Assignment4/` | checkpoints, ONNX models, CSVs, figures, ZIP | ~0.5 GB |

## Remove everything
Delete this folder. (Python itself and the NVIDIA driver are installed separately and stay on your PC. If you open the notebook with VS Code instead of `run_jupyter`, VS Code keeps a few small settings files of its own outside this folder.)

## Tips for a 6 GB laptop GPU
- Plug in the charger and set Windows power mode to *Best performance*.
- Close games/browsers using the GPU. If you get "CUDA out of memory", lower `TEACHER_BS` (e.g. 32) in the config cell.
- Expect roughly 1.5-3 hours for the full run.
