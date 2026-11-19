"""Read saved training progress without touching the running GPU kernel."""
import json
from pathlib import Path

base=Path(__file__).resolve().parent
status=base/'bootstrap_status.txt'
if status.exists():print(status.read_text(encoding='utf-8').strip())
root=base/'EdgeAI_Assignment4'
history=root/'results/histories.json'
if history.exists():
    histories=json.loads(history.read_text())
    for name,h in histories.items():
        if h.get('epoch'):
            print(f'{name:30s} epoch {h["epoch"][-1]:3d}  accuracy {100*h["val_acc"][-1]:6.2f}%  loss {h["train_loss"][-1]:7.3f}  last epoch {h["time"][-1]:6.1f}s')
    print('Retained epoch checkpoints:',sum(1 for _ in (root/'checkpoints/epochs').rglob('*.pt')))
search=root/'results/nas_search_history.json'
if search.exists():
    trials=json.loads(search.read_text())
    print('NAS trials:',len(trials),'accepted:',sum(t['status']=='accepted' for t in trials),'rejected:',sum(t['status']=='rejected' for t in trials))
