"""Execute the notebook, saving outputs after each cell, then build the report/package."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
os.chdir(BASE)
for name, folder in {'TORCH_HOME':'.cache/torch', 'MPLCONFIGDIR':'.cache/matplotlib',
                     'XDG_CACHE_HOME':'.cache', 'HF_HOME':'.cache/huggingface',
                     'JUPYTER_CONFIG_DIR':'.jupyter', 'JUPYTER_DATA_DIR':'.jupyter/data',
                     'JUPYTER_RUNTIME_DIR':'.jupyter/runtime', 'IPYTHONDIR':'.ipython',
                     'TEMP':'.tmp', 'TMP':'.tmp'}.items():
    target = BASE / folder
    target.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(target)
os.environ['MPLBACKEND'] = 'module://matplotlib_inline.backend_inline'
if os.name == 'nt':
    import asyncio
    import ctypes
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # Keep training awake; Windows releases this request when the process exits.
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    os.environ['EDGEAI_QUICK'] = '1' if args.quick else '0'
    import nbformat
    from nbclient import NotebookClient
    source = BASE / 'Assignment4_KD_NAS_EdgeAI_Local.ipynb'
    output = BASE / ('Assignment4_QUICK_executed.ipynb' if args.quick else 'Assignment4_executed.ipynb')
    nb = nbformat.read(source, as_version=4)
    # Explicit interpreter avoids using a global notebook kernel.
    from jupyter_client.kernelspec import KernelSpecManager
    kernel_dir = BASE / '.jupyter/kernels/edgeai-run'
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / 'kernel.json').write_text(json.dumps(dict(argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'], display_name='EdgeAI local', language='python')))
    manager = KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)])
    def save(**kwargs):
        tmp = output.with_suffix('.ipynb.tmp')
        nbformat.write(nb, tmp)
        os.replace(tmp, output)
        index = kwargs.get('cell_index', -1)
        print(f'[{time.strftime("%H:%M:%S")}] Completed cell {index}', flush=True)
        cell = kwargs.get('cell', {})
        for item in cell.get('outputs', []):
            if item.get('output_type') == 'stream':
                print(item.get('text', '')[-4000:], flush=True)
    client = NotebookClient(nb, timeout=None, kernel_name='edgeai-run',
                            resources={'metadata':{'path':str(BASE)}},
                            on_cell_executed=save)
    client.km = __import__('jupyter_client').AsyncKernelManager(kernel_name='edgeai-run', kernel_spec_manager=manager)
    try:
        client.execute()
    finally:
        save()
    if not args.quick:
        # Also place the completed outputs in the user's original notebook.
        nbformat.write(nb, source)
        subprocess.run([sys.executable, 'finalize_assignment.py'], check=True)
    print('ALL REQUESTED EXPERIMENTS COMPLETED', flush=True)

if __name__ == '__main__':
    main()
