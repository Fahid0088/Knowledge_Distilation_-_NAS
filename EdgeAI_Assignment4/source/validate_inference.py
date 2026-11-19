"""Validate both standalone inference backends on real CIFAR images."""
import argparse
import json
from pathlib import Path
import pickle
import numpy as np
from PIL import Image
from inference import load_input, logits

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='EdgeAI_Assignment4')
    args=ap.parse_args()
    root=Path(args.root)
    meta=json.loads((root/'onnx/meta.json').read_text())
    samples={}
    for name,source in [('cifar100','data/cifar-100-python/test'),('cifar10','data/cifar-10-batches-py/test_batch')]:
        with open(source,'rb') as handle: data=pickle.load(handle,encoding='bytes')
        image=data[b'data'][0].reshape(3,32,32).transpose(1,2,0)
        path=root/'results'/f'demo_{name}.png'
        Image.fromarray(image).save(path)
        samples[name]=path
    checks=[]
    for key,info in meta.items():
        x=load_input(samples['cifar10' if key=='nas_best' else 'cifar100'],info)
        reference=logits(root,key,x,'torch')
        output=logits(root,key,x,'onnx')
        np.testing.assert_allclose(output,reference,rtol=1e-4,atol=1e-4)
        row=dict(model=key,max_abs_difference=float(np.abs(output-reference).max()),
                 torch_class=info['classes'][int(reference.argmax())],
                 onnx_class=info['classes'][int(output.argmax())])
        checks.append(row)
        print(row)
    (root/'results/standalone_inference_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print('Standalone inference validated for all models.')

if __name__=='__main__':main()
