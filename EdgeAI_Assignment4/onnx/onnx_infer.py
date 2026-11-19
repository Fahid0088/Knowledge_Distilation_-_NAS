# Standalone CPU inference with ONNX Runtime.
# Usage: python onnx_infer.py --key mobilenet_v2_kd --image my_image.png --topk 5
import argparse, json, os
import numpy as np
import onnxruntime as ort
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--key", required=True)
ap.add_argument("--image", required=True)
ap.add_argument("--topk", type=int, default=5)
ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
a = ap.parse_args()

meta = json.load(open(os.path.join(a.dir, "meta.json")))[a.key]
res = meta["res"]
img = Image.open(a.image).convert("RGB").resize((res, res), Image.BILINEAR)
x = np.asarray(img, dtype=np.float32) / 255.0
x = (x - np.array(meta["mean"], dtype=np.float32)) / np.array(meta["std"], dtype=np.float32)
x = x.transpose(2, 0, 1)[None].astype(np.float32)

so = ort.SessionOptions(); so.intra_op_num_threads = 1
sess = ort.InferenceSession(os.path.join(a.dir, meta["onnx"]), so, providers=["CPUExecutionProvider"])
logits = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
p = np.exp(logits - logits.max()); p /= p.sum()
for j in np.argsort(-p)[:a.topk]:
    print(meta["classes"][j].ljust(20), "%6.2f%%" % (p[j] * 100))
