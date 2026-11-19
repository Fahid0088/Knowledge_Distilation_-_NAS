"""Check the standalone NumPy preprocessing against the notebook's PyTorch operations."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from inference import load_input

torch.set_num_threads(2)
image=np.random.default_rng(42).integers(0,256,(32,32,3),dtype=np.uint8)
path=Path('.cache/preprocessing_check.png');Image.fromarray(image).save(path)
for res in (16,24,32):
    meta=dict(res=res,mean=[0.5,0.4,0.3],std=[0.2,0.25,0.3])
    x=torch.tensor(image).permute(2,0,1)[None].float()/255
    if res!=32:x=torch.nn.functional.interpolate(x,size=(res,res),mode='bilinear',align_corners=False)
    reference=(x-torch.tensor(meta['mean']).view(1,3,1,1))/torch.tensor(meta['std']).view(1,3,1,1)
    difference=float(np.abs(reference.numpy()-load_input(path,meta)).max())
    assert difference<3e-6,(res,difference)
    print(f'{res}x{res}: maximum preprocessing difference {difference:.3g}')
