"""Architectures used in the executed assignment notebook."""
import torch
from torch import nn
import torchvision
import timm

def make_teacher(num_classes=100, pretrained=True):
    try:
        w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = torchvision.models.resnet50(weights=w)
    except Exception as e:
        raise RuntimeError("ImageNet teacher initialization failed; check the local weight cache or network.") from e
    m.conv1.stride = (1, 1)           # keep resolution for 32x32 inputs
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(2048, num_classes)
    return m

def make_student(name, num_classes=100):
    if name == "mobilenet_v2":
        m = torchvision.models.mobilenet_v2(weights=None, num_classes=num_classes)
        m.features[0][0].stride = (1, 1)                       # stem stride 2 -> 1
    elif name == "squeezenet":
        m = torchvision.models.squeezenet1_1(weights=None, num_classes=num_classes)
        m.features[0].stride = (1, 1); m.features[0].padding = (1, 1)
        m.features[2] = nn.Identity()                          # drop the first max-pool
        m.classifier[2] = nn.Identity()                        # remove final ReLU so outputs are real logits
    elif name == "efficientnet_lite0":
        m = timm.create_model("efficientnet_lite0", pretrained=False, num_classes=num_classes)
        m.conv_stem.stride = (1, 1)
    else:
        raise ValueError(name)
    return m

SPACE = dict(resolution=[16, 24, 32], min_depth=2, max_depth=6, stem_channels=[16, 24, 32],
             channels=[16, 24, 32, 48, 64, 96, 128], kernels=[3, 5],
             block_types=["conv", "dwsep"], activations=["relu", "relu6", "silu", "hardswish"],
             strides=[1, 2], max_downsamples=3)

ACTS = {"relu": nn.ReLU, "relu6": nn.ReLU6, "silu": nn.SiLU, "hardswish": nn.Hardswish}

def conv_block(cin, cout, k, stride, act):
    """Standard conv -> BN -> activation."""
    return nn.Sequential(nn.Conv2d(cin, cout, k, stride, k // 2, bias=False), nn.BatchNorm2d(cout), ACTS[act]())

def dwsep_block(cin, cout, k, stride, act):
    """Depthwise conv -> BN -> act -> pointwise 1x1 conv -> BN -> act."""
    return nn.Sequential(
        nn.Conv2d(cin, cin, k, stride, k // 2, groups=cin, bias=False), nn.BatchNorm2d(cin), ACTS[act](),
        nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), ACTS[act]())

class SearchedNet(nn.Module):
    """Network generated from an architecture configuration dict."""
    def __init__(self, cfg, num_classes=10):
        super().__init__()
        sc = cfg["stem_channels"]
        layers = [conv_block(3, sc, 3, 1, cfg["stem_act"])]
        c = sc
        for b in cfg["blocks"]:
            fn = dwsep_block if b["type"] == "dwsep" else conv_block
            layers.append(fn(c, b["out"], b["k"], b["stride"], b["act"]))
            c = b["out"]
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, num_classes))
    def forward(self, x):
        return self.head(self.features(x))

def build_model(cfg, num_classes=10):
    return SearchedNet(cfg, num_classes)

def sample_cfg(trial):
    """Turn an Optuna trial into an architecture config.
    Parameters for all 6 possible blocks are always suggested; only the first `depth` blocks are used."""
    res   = trial.suggest_categorical("resolution", SPACE["resolution"])
    depth = trial.suggest_int("depth", SPACE["min_depth"], SPACE["max_depth"])
    stem  = trial.suggest_categorical("stem_channels", SPACE["stem_channels"])
    stem_act = trial.suggest_categorical("stem_act", SPACE["activations"])
    blocks, downs = [], 0
    for i in range(SPACE["max_depth"]):
        t = trial.suggest_categorical(f"b{i}_type", SPACE["block_types"])
        ch = trial.suggest_categorical(f"b{i}_ch", SPACE["channels"])
        k = trial.suggest_categorical(f"b{i}_k", SPACE["kernels"])
        a = trial.suggest_categorical(f"b{i}_act", SPACE["activations"])
        s = trial.suggest_categorical(f"b{i}_stride", SPACE["strides"])
        if i < depth:
            if s == 2 and downs >= SPACE["max_downsamples"]:
                s = 1
            downs += int(s == 2)
            blocks.append(dict(type=t, out=ch, k=k, act=a, stride=s))
    return dict(resolution=res, stem_channels=stem, stem_act=stem_act, blocks=blocks)
