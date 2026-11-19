"""Validate completed artifacts, generate the PDF, and package deliverables only."""
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

BASE = Path(__file__).resolve().parent
ROOT = BASE / 'EdgeAI_Assignment4'

def export_model_source():
    source = (BASE / 'notebook_source.py').read_text(encoding='utf-8')
    wanted = {'make_teacher','make_student','conv_block','dwsep_block','SearchedNet','build_model','sample_cfg'}
    tree = ast.parse(source)
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted:
            selected.append(ast.get_source_segment(source, node))
        if isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'SPACE','ACTS'} for t in node.targets):
            selected.append(ast.get_source_segment(source, node))
    header = '"""Architectures used in the executed assignment notebook."""\nimport torch\nfrom torch import nn\nimport torchvision\nimport timm\n\n'
    (BASE / 'models.py').write_text(header + '\n\n'.join(selected) + '\n', encoding='utf-8')

def report():
    import pandas as pd
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from xml.sax.saxutils import escape
    styles = getSampleStyleSheet()
    config = json.loads((ROOT/'results/run_config.json').read_text())
    quick = config['quick_test']
    ntrain, ntest = (5000,1000) if quick else (50000,10000)
    story = []
    def text(s, style='BodyText'):
        story.append(Paragraph(escape(s), styles[style])); story.append(Spacer(1,8))
    def table(rows):
        item = Table([[Paragraph(escape(str(v)), styles['BodyText']) for v in row] for row in rows], repeatRows=1, hAlign='LEFT')
        item.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAD5BE')),('GRID',(0,0),(-1,-1),0.3,colors.grey),('VALIGN',(0,0),(-1,-1),'TOP'),('FONTSIZE',(0,0),(-1,-1),8)]))
        story.append(item); story.append(Spacer(1,12))
    text('Assignment 4: Knowledge Distillation and Hardware-Aware NAS','Title')
    if quick:text('SMOKE TEST: small-data validation run; final experiments are saved separately.','Heading2')
    text('23i-0061 | AI-7B | Edge Computing & Intelligent Systems')
    text('Repository: https://github.com/Fahid0088/Knowledge_Distilation_-_NAS')
    text('Experimental setup','Heading1')
    import torch, torchvision, timm, onnxruntime
    text(f'Python {platform.python_version()}, PyTorch {torch.__version__}, torchvision {torchvision.__version__}, timm {timm.__version__}, ONNX Runtime {onnxruntime.__version__}. Training device: {torch.cuda.get_device_name(0)}. CPU: {platform.processor()}.')
    text(f'This run uses seed {config["seed"]}, {ntrain:,} CIFAR training images and {ntest:,} test images per dataset. Teacher: {config["teacher_epochs"]} epochs; each baseline and distilled student: {config["student_epochs"]} epochs. NAS: {config["nas_trials"]} candidate trials, {config["nas_candidate_epochs"]} training epochs per accepted candidate on {config["nas_train_samples"]:,} training images and {min(5000,ntrain//5):,} held-out training images for validation. The winning architecture is initialized afresh and trained for {config["nas_final_epochs"]} epochs on all {ntrain:,} available CIFAR-10 training images.')
    text('Question 1: knowledge distillation','Heading1')
    text('The ResNet-50 teacher starts from ImageNet V2 weights and is fine-tuned on CIFAR-100. Its stem stride is one and max-pooling is removed for 32x32 inputs. The three student implementations use torchvision MobileNetV2, torchvision SqueezeNet 1.1, and timm EfficientNet-Lite0. Student stem strides are adapted to CIFAR. SqueezeNet produces unrestricted logits by removing its classifier ReLU.')
    text('Teacher training uses SGD with momentum 0.9 and weight decay 0.0005. Students use AdamW, weight decay 0.05, learning rate 0.002 (0.001 for SqueezeNet), a OneCycle schedule, batch size 128, mixed precision, and gradient clipping at 5. Random crop and horizontal flip are applied identically to teacher and student inputs during distillation. Baseline and KD student weights start from the same seeded initialization.')
    text('KD loss = 0.7 * T^2 * KL(softmax(teacher/T) || softmax(student/T)) + 0.3 * CE(student, label) + 1.0 * feature_loss, with T=4. The final student feature map is projected to 2,048 teacher channels by a learned 1x1 convolution and batch normalization. Adaptive pooling matches spatial sizes; feature loss is the mean of one minus cosine similarity. Teacher parameters are frozen. Feature adapters are training-only and retained in recovery snapshots.')
    hist = json.loads((ROOT/'results/histories.json').read_text())
    text(f'Teacher final CIFAR-100 test accuracy: {100*hist["teacher_resnet50"]["val_acc"][-1]:.2f}%.')
    kd = pd.read_csv(ROOT/'results/kd_vs_baseline.csv')
    table([['Student','Baseline (%)','KD (%)','Gain (pp)']] + [[r.student,f'{r.baseline_acc:.2f}',f'{r.distilled_acc:.2f}',f'{r.kd_gain:+.2f}'] for r in kd.itertuples()])
    for r in kd.itertuples():
        text(f'{r.student}: distillation {"improved" if r.kd_gain>0 else "did not improve"} accuracy by {abs(r.kd_gain):.2f} percentage points in this run. This is an observed result, not a guaranteed property of distillation.')
    text('Question 2: hardware-aware architecture search','Heading1')
    text('Optuna TPE explores depth 2-6, stem widths 16/24/32, block widths 16/24/32/48/64/96/128, 3x3 or 5x5 kernels, standard or depthwise-separable convolution, ReLU/ReLU6/SiLU/HardSwish, stride one or two, and 16/24/32 input resolution. At most three downsampling blocks are allowed. One small architecture seeds the study; later proposals are automatic.')
    text('Before candidate training, each architecture is exported to ONNX and measured with CPUExecutionProvider, one thread, batch one. Hard limits are median latency <=10 ms, ONNX storage <=1 MB, and incremental process peak memory <=200 MiB where measurable. Latency includes ten warmup and 50 timed passes during search. Memory is measured in a fresh subprocess using Windows peak working set (including ONNX session initialization). Unsupported memory measurement is explicitly NaN.')
    text('Candidate score = validation accuracy fraction - 0.05*(latency/10 ms) - 0.02*(size/1 MB). Rejected trials never enter accuracy-based ranking. The CIFAR-10 test set is excluded from candidate selection. The final winning model is trained from scratch on the complete training set.')
    search = pd.read_csv(ROOT/'results/nas_search_history.csv')
    best = json.loads((ROOT/'results/best_architecture.json').read_text())
    text(f'Trials: {len(search)}; accepted: {sum(search.status=="accepted")}; rejected: {sum(search.status=="rejected")}. Best search validation accuracy: {100*best["search_val_acc"]:.2f}%. Selected resolution: {best["config"]["resolution"]}; parameters: {best["params"]:,}.')
    table([['Block','Type','Channels','Kernel','Stride','Activation']] + [[i,b['type'],b['out'],b['k'],b['stride'],b['act']] for i,b in enumerate(best['config']['blocks'])])
    text('ONNX deployment and measured results','Heading1')
    text('Student accuracies are measured on CIFAR-100 (100 classes); the NAS architecture is measured on CIFAR-10 (10 classes). Accuracy values across these two datasets are not directly comparable.')
    bench = pd.read_csv(ROOT/'results/benchmark_results.csv')
    table([['Model','Torch test %','ONNX %','MB','CPU ms','Peak MiB']] + [[r.model,f'{r.torch_test_acc:.2f}',f'{r.onnx_acc:.2f}',f'{r.onnx_size_MB:.3f}',f'{r.latency_ms_median:.3f}',f'{r.peak_mem_MB:.1f}'] for r in bench.itertuples()])
    text(f'Torch accuracy uses all {ntest:,} test images; ONNX accuracy and Torch/ONNX agreement use the first {config["ort_eval_samples"]:,} test images. Those accuracy columns therefore have different denominators. CPU deployment benchmarks use 20 warmup and 100 timed passes; CSV includes median and p95. Parameter count, disk size, accuracy, and incremental peak memory are recorded automatically.')
    text('Full-test PyTorch evaluation uses GPU mixed precision. PyTorch/ONNX agreement is measured between CPU float32 implementations on identical preprocessed inputs, separating export correctness from the full-test evaluation precision and sample-count differences.')
    checks_path=ROOT/'results/standalone_inference_checks.json'
    if checks_path.exists():
        checks=json.loads(checks_path.read_text())
        text(f'The standalone CLI was also checked on real CIFAR images for all {len(checks)} deployed models. Maximum observed absolute PyTorch/ONNX logit difference on those examples: {max(c["max_abs_difference"] for c in checks):.3g}. NumPy bilinear preprocessing was separately compared with the notebook at 16, 24, and 32 pixels; the maximum observed preprocessing difference was below 0.000001.')
    text('Limitations and reproducibility','Heading1')
    text('This is one seeded run on one laptop, with a limited epoch budget. Accuracy may improve with longer training or hyperparameter tuning. CIFAR-100 test accuracy is monitored each epoch; delivered checkpoints are the final epoch, with separate best-observed snapshots retained locally for recovery. No claim of repeated-seed statistical significance is made. CPU timing can vary with thermals, power settings, and other applications. Search uses untrained architecture latency; final trained model latency is separately benchmarked. The datasets are downloaded from the official CIFAR source; standard architecture implementations come from torchvision and timm.')
    text('Every epoch is saved atomically with model, optimizer, scheduler, AMP scaler, feature projection where relevant, training history, and RNG state. Finished model checkpoints and NAS histories are reloaded on rerun. The submission archive excludes the virtual environment, datasets, caches, executables, and intermediate candidate checkpoints; all recovery material remains in the project folder.')
    text('Execution evidence and result screenshots','Heading1')
    for filename in sorted((ROOT/'figures').glob('*.png')):
        story.append(PageBreak()); text(filename.stem.replace('_',' '),'Heading2')
        from PIL import Image as PILImage
        with PILImage.open(filename) as im: width,height=im.size
        scale = min(6.8*inch/width, 8.3*inch/height)
        story.append(Image(str(filename), width*scale, height*scale))
    doc = SimpleDocTemplate(str(ROOT/'Technical_Report.pdf'), rightMargin=36,leftMargin=36,topMargin=36,bottomMargin=36)
    doc.build(story)

def main():
    export_model_source()
    import pandas as pd
    import onnx
    import torch
    keys = [f'{s}_{v}' for s in ['mobilenet_v2','squeezenet','efficientnet_lite0'] for v in ['baseline','kd']] + ['nas_best']
    for key in ['teacher_resnet50'] + keys:
        state = torch.load(ROOT/'checkpoints'/f'{key}.pt',map_location='cpu',weights_only=True)
        assert state and all(torch.isfinite(t).all() for t in state.values() if t.is_floating_point()), key
    for key in keys: onnx.checker.check_model(str(ROOT/'onnx'/f'{key}.onnx'))
    frame = pd.read_csv(ROOT/'results/benchmark_results.csv')
    assert set(frame.model) == set(keys)
    assert frame.torch_onnx_agreement.min() > 98, 'Inspect ONNX numerical agreement'
    subprocess.run([sys.executable,'validate_inference.py'],check=True)
    report()
    completion = ('## Completed experiment\n\n'
                  'All training, NAS, ONNX inference and benchmark outputs are saved in this notebook. '
                  'Final checkpoints, CSV/JSON results, figures and `Technical_Report.pdf` are in '
                  '`EdgeAI_Assignment4/` in the project (or the submission root after extracting the ZIP). '
                  'Every completed epoch is retained locally in `checkpoints/epochs/`.\n\n'
                  'The final submission package is `23i-0061_AI-7B.zip`. '
                  'Source and results repository: https://github.com/Fahid0088/Knowledge_Distilation_-_NAS\n')
    for name in ['Assignment4_KD_NAS_EdgeAI_Local.ipynb','Assignment4_executed.ipynb']:
        path=BASE/name
        notebook=json.loads(path.read_text(encoding='utf-8'))
        assert notebook['cells'][-1]['cell_type']=='markdown'
        notebook['cells'][-1]['source']=completion.splitlines(keepends=True)
        path.write_text(json.dumps(notebook,indent=1,ensure_ascii=False),encoding='utf-8')
    note = ('# Packaged submission\n\n'
            'Inside the submission ZIP, the completed notebooks and scripts are in `source/`. '
            'Open `source/Assignment4_KD_NAS_EdgeAI_Local.ipynb` to view all saved outputs. '
            'Run setup and training commands below from `source/` when reproducing experiments. '
            'To use the existing packaged models from this directory, run '
            '`python source/inference.py --root . --key nas_best --image example.png` or '
            '`python source/benchmark.py --root .`.\n\n')
    (ROOT/'README.md').write_text(note+(BASE/'README.md').read_text(encoding='utf-8'),encoding='utf-8')
    source_dir = ROOT/'source'; source_dir.mkdir(exist_ok=True)
    names = ['Assignment4_KD_NAS_EdgeAI_Local.ipynb','Assignment4_executed.ipynb','notebook_source.py','run_assignment.py','finalize_assignment.py','models.py','inference.py','benchmark.py','validate_inference.py','show_progress.py','requirements.txt','constraints-local.txt','dependencies-local.txt','README.md','README_LOCAL.md','setup_windows.bat','run_jupyter.bat']
    for name in names: shutil.copy2(BASE/name,source_dir/name)
    freeze = subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (ROOT/'requirements-lock.txt').write_text(freeze,encoding='utf-8')
    manifest = {}
    def eligible(path):
        intermediate_best = path.name.endswith('_best.pt') and path.name != 'nas_best.pt'
        return path.is_file() and 'epochs' not in path.relative_to(ROOT).parts and not path.name.endswith(('_resume.pt','.tmp')) and not intermediate_best and not path.name.startswith('nas_t')
    for path in sorted(ROOT.rglob('*')):
        if eligible(path) and path.name != 'sha256.json':
            manifest[path.relative_to(ROOT).as_posix()] = hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
    (ROOT/'sha256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    target = BASE/'23i-0061_AI-7B.zip'
    temporary = target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob('*')):
            if eligible(path): archive.write(path,Path('23i-0061_AI-7B')/path.relative_to(ROOT))
    with zipfile.ZipFile(temporary) as archive: assert archive.testzip() is None
    os.replace(temporary,target)
    print(f'Validated submission: {target} ({target.stat().st_size/1e6:.1f} MB)')

if __name__ == '__main__': main()
