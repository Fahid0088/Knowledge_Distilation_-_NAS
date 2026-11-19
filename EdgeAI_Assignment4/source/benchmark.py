"""Repeat ONNX CPU timing without retraining any network."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import onnxruntime as ort

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='EdgeAI_Assignment4')
    ap.add_argument('--runs', type=int, default=100)
    args = ap.parse_args()
    root = Path(args.root)
    meta = json.loads((root / 'onnx/meta.json').read_text())
    rows = []
    for key, info in meta.items():
        path = root / 'onnx' / info['onnx']
        so = ort.SessionOptions(); so.intra_op_num_threads = so.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path), so, providers=['CPUExecutionProvider'])
        x = np.zeros((1,3,info['res'],info['res']), np.float32)
        feed = {session.get_inputs()[0].name:x}
        for _ in range(20): session.run(None, feed)
        times = []
        for _ in range(args.runs):
            start = time.perf_counter(); session.run(None, feed)
            times.append((time.perf_counter()-start)*1000)
        rows.append(dict(model=key, size_MB=path.stat().st_size/1e6,
                         latency_ms_median=float(np.median(times)), latency_ms_p95=float(np.percentile(times,95))))
    frame = pd.DataFrame(rows)
    frame.to_csv(root / 'results/benchmark_repeated.csv', index=False)
    print(frame.to_string(index=False))

if __name__ == '__main__': main()
