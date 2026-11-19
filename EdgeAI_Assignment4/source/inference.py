"""Unified image inference for student and searched models, on CPU."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

def load_input(image, meta):
    image = Image.open(image).convert('RGB').resize((32, 32), Image.Resampling.BILINEAR)
    x = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / 255
    # Match PyTorch bilinear (align_corners=False) without requiring Torch on edge CPUs.
    if meta['res'] != 32:
        res = meta['res']
        position = np.clip((np.arange(res, dtype=np.float32)+0.5)*np.float32(32/res)-0.5, 0, 31)
        lower = np.floor(position).astype(np.int64)
        upper = np.minimum(lower+1, 31)
        wy = (position-lower.astype(np.float32))[None,None,:,None]
        wx = (position-lower.astype(np.float32))[None,None,None,:]
        top = x[:,:,lower[:,None],lower[None,:]]*(1-wx) + x[:,:,lower[:,None],upper[None,:]]*wx
        bottom = x[:,:,upper[:,None],lower[None,:]]*(1-wx) + x[:,:,upper[:,None],upper[None,:]]*wx
        x = top*(1-wy) + bottom*wy
    mean = np.asarray(meta['mean'], dtype=np.float32).reshape(1,3,1,1)
    std = np.asarray(meta['std'], dtype=np.float32).reshape(1,3,1,1)
    return np.ascontiguousarray((x-mean)/std, dtype=np.float32)

def logits(root, key, x, backend='onnx'):
    root = Path(root)
    meta = json.loads((root / 'onnx/meta.json').read_text())[key]
    if backend == 'onnx':
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = 1
        session = ort.InferenceSession(str(root / 'onnx' / meta['onnx']), so, providers=['CPUExecutionProvider'])
        return session.run(None, {session.get_inputs()[0].name:x})[0]
    import torch
    from models import make_student, build_model
    torch.set_num_threads(4)
    if key == 'nas_best':
        cfg = json.loads((root / 'results/best_architecture.json').read_text())['config']
        model = build_model(cfg, 10)
    else:
        model = make_student(key.rsplit('_', 1)[0], 100)
    model.load_state_dict(torch.load(root / 'checkpoints' / (key + '.pt'), map_location='cpu', weights_only=True))
    with torch.no_grad():
        return model.eval()(torch.from_numpy(x)).numpy()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='EdgeAI_Assignment4')
    ap.add_argument('--key', default='mobilenet_v2_kd')
    ap.add_argument('--image', required=True)
    ap.add_argument('--backend', choices=['torch','onnx'], default='onnx')
    ap.add_argument('--topk', type=int, default=5)
    args = ap.parse_args()
    meta = json.loads((Path(args.root) / 'onnx/meta.json').read_text())[args.key]
    out = logits(args.root, args.key, load_input(args.image, meta), args.backend)[0]
    p = np.exp(out - out.max()); p /= p.sum()
    for i in np.argsort(-p)[:args.topk]:
        print(f'{meta["classes"][i]:20s} {100*p[i]:6.2f}%')

if __name__ == '__main__':
    main()
