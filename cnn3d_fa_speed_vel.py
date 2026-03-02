# -*- coding: utf-8 -*-
"""
cnn3d_fa_speed_vel.py
3D CNN training 

Experiment types:
  - fa          : FA only (single channel)
  - speed       : speed only (single channel)
  - vel         : vel only (single channel)
  - fa_speed    : FA + speed parallel two branches -> flatten merge
  - fa_vel      : FA + vel parallel two branches -> flatten merge
  - fa_velspeed : FA + speed + vel 2 channel concat (current)
  - fa_speed_vel: FA + speed + vel parallel three branches -> flatten merge

NUM_RUNS=10, run_1, run_2, ... saved in same experiment folder.

Author: Miccai_5340 
Code written with assistance from Cursor AI.
"""

import os, glob, json, time, re
from dataclasses import asdict, dataclass

import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False


# =========================
# 0) CONFIG - GPU
# =========================
ROOT = "/content/drive/MyDrive/MICCAI"
DATASET_CV_DIR = "Dataset_CrossValidation"
# vel/speed export path (if EXPORT_TO_LOCAL: "/content/flow_matching_velocity_speed_t0.80")
FM_EXPORT_ROOT = os.path.join(ROOT, "flow_matching_velocity_speed_t0.80")

# CV: RUN_ALL_CVS=False -> only CV_FOLD. RUN_ALL_CVS=True -> those in CV_LIST
CV_FOLD = "CV1"
RUN_ALL_CVS = True
CV_LIST = ["CV3", "CV4", "CV5", "CV10"]  #"CV1", "CV2", "CV3", "CV4",
MODALITY = "rd"   # fa, md, rd, ad, wm, gm, mri

OUT_PARENT = os.path.join(ROOT, "cnn3d_runs")

# Experiment: fa | speed | vel | fa_speed | fa_vel | fa_velspeed | fa_speed_vel
# fa_speed, fa_vel, fa_velspeed: 2 channel concat - ResNet3D_CBAM(in_ch=2)
# fa_speed_vel: 3 parallel branches - ParallelBranchesModel
EXPERIMENT = "rd_vel"   # Raw: fa, wm, md, rd, ad, gm, mri

MODEL_TYPE = "cbam_resnet"

EPOCHS = 80
LR = 2e-4
WEIGHT_DECAY = 1e-4

BATCH_SIZE = 8
NUM_WORKERS = 8
PIN_MEMORY = True

USE_ZSCORE = False

USE_EARLY_STOP = True
EARLY_STOP_PATIENCE = 50
EARLY_STOP_MIN_DELTA = 1e-6

LR_PLATEAU_PATIENCE = 10
LR_FACTOR = 0.5
LR_MIN = 1e-6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_RUNS = 10
# Same folder run_1, run_2, ... (True) or each run separate timestamp (False)
SAME_FOLDER_RUNS = True


# =========================
# 1) Helpers
# =========================
def set_random_mode():
    import random
    seed = int(time.time() * 1000) % 2_000_000_000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    print("Random seed:", seed)
    return seed


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def now_tag():
    return time.strftime("%Y%m%d_%H%M%S")


def robust_zscore(x, eps=1e-6):
    m, s = float(np.mean(x)), float(np.std(x))
    return (x - m) / (s + eps)


def extract_id_from_filename(fname: str) -> str:
    base = os.path.basename(fname)
    parts = base.split("_")
    if len(parts) < 2:
        return base.replace(".nii.gz", "").replace(".nii", "")
    return parts[1].split(".")[0]


def collate_fn_parallel(batch):
    """Correctly stacks tuple (x_a, x_b) or (x_a, x_b, x_c) for fa_speed, fa_vel, fa_speed_vel"""
    from torch.utils.data.dataloader import default_collate
    elem = batch[0]
    if isinstance(elem.get("x"), (tuple, list)):
        # if x is tuple: [(t1,t2), (t1,t2), ...] -> (stack(t1s), stack(t2s))
        xs = [b["x"] for b in batch]
        xs = tuple(torch.stack([xs[i][j] for i in range(len(batch))]) for j in range(len(xs[0])))
        return {"x": xs, "y": default_collate([b["y"] for b in batch]), "id": [b["id"] for b in batch]}
    return default_collate(batch)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_csv_row(path, header, row_dict):
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if not exists:
            f.write(",".join(header) + "\n")
        f.write(",".join(str(row_dict[h]) for h in header) + "\n")


def plot_curves(csv_path, out_dir):
    if not HAS_PLT:
        return
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    ep, tr_loss, te_loss = data["epoch"], data["train_loss"], data["test_loss"]
    tr_acc, te_acc = data["train_acc"], data["test_acc"]
    plt.figure()
    plt.plot(ep, tr_loss, label="train_loss")
    plt.plot(ep, te_loss, label="test_loss")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150)
    plt.close()
    plt.figure()
    plt.plot(ep, tr_acc, label="train_acc")
    plt.plot(ep, te_acc, label="test_acc")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "acc_curve.png"), dpi=150)
    plt.close()


def write_best_last_csv(out_dir, header, best_row, last_row):
    path = os.path.join(out_dir, "metrics_best_last.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        f.write(",".join(str(best_row[h]) for h in header) + "\n")
        f.write(",".join(str(last_row[h]) for h in header) + "\n")
    return path


# =========================
# 2) Paths
# =========================
_current_cv = None  # set in loop when RUN_ALL_CVS

def get_cv_folds():
    """CV1, CV2, ... CV10 under Dataset_CrossValidation"""
    cv_base = os.path.join(ROOT, DATASET_CV_DIR)
    if not os.path.exists(cv_base):
        return []
    folds = [d for d in os.listdir(cv_base)
             if os.path.isdir(os.path.join(cv_base, d)) and d.upper().startswith("CV")]
    def _key(n):
        m = re.match(r"CV(\d+)", n, re.I)
        return (int(m.group(1)),) if m else (999,)
    return sorted(folds, key=_key)

# Raw data (normal AD/FA): ROOT/Dataset_CrossValidation/CV1/Train/0|1/AD/
#   e.g.: .../Dataset_CrossValidation/CV1/Train/1/AD/n1_S05_1_dti2_AD_reg.nii.gz
# FM export (speed/vel): FM_EXPORT_ROOT/CV1/AD/speed_t0.80|vel_t0.80/Train/0|1/AD/
#   e.g.: .../flow_matching_velocity_speed_t0.80/CV1/AD/speed_t0.80/Train/1/AD/n1_S05_1_dti2_AD_reg_spd_t0.80.nii.gz
def _cv():
    return _current_cv if _current_cv is not None else CV_FOLD

def _base_dir():
    cv = _cv()
    if cv:
        return os.path.join(ROOT, DATASET_CV_DIR, cv)
    return ROOT


def _fm_mod_dir(mod, vel_or_speed):
    """FM export: vel_t0.80 or speed_t0.80"""
    cv = _cv()
    if cv:
        return os.path.join(FM_EXPORT_ROOT, cv, mod.upper(), f"{vel_or_speed}_t0.80")
    return os.path.join(FM_EXPORT_ROOT, f"{vel_or_speed}_t0.80")


def _find_mod_dir(base_path):
    """Find modality folder (case-insensitive)"""
    if not os.path.exists(base_path):
        return None
    mod = MODALITY.upper()
    cand = os.path.join(base_path, mod)
    if os.path.isdir(cand):
        return cand
    cand = os.path.join(base_path, mod.lower())
    if os.path.isdir(cand):
        return cand
    for name in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, name)) and name.upper() == mod:
            return os.path.join(base_path, name)
    return None


_RAW_MOD_EXPS = ("fa", "md", "rd", "ad", "wm", "gm", "mri")  # raw modality - must match MODALITY

# parse wm_speed, md_vel, ad_velspeed etc. -> (mod, base_exp)
_MOD_SUFFIX_MAP = {"speed": "fa_speed", "vel": "fa_vel", "velspeed": "fa_velspeed", "speed_vel": "fa_speed_vel"}
def _parse_exp_modality(exp: str):
    """wm_speed -> (wm, fa_speed), md_vel -> (md, fa_vel). Otherwise (None, exp)."""
    for mod in _RAW_MOD_EXPS:
        prefix = mod + "_"
        if exp.startswith(prefix):
            suffix = exp[len(prefix):]
            if suffix in _MOD_SUFFIX_MAP:
                return mod, _MOD_SUFFIX_MAP[suffix]
    return None, exp

def get_split_dir(exp: str, split: str, label: str) -> str:
    """Single channel: fa, wm, md, ... speed, vel"""
    mod = MODALITY
    base = os.path.join(_base_dir(), split, label)
    if exp in _RAW_MOD_EXPS:
        d = _find_mod_dir(base)
        return d if d else os.path.join(base, mod)
    if exp in ("fa_vel", "vel"):
        return os.path.join(_fm_mod_dir(mod, "vel"), split, label, mod.upper())
    if exp in ("fa_speed", "speed"):
        return os.path.join(_fm_mod_dir(mod, "speed"), split, label, mod.upper())
    raise ValueError(f"get_split_dir: {exp}")


def get_split_dirs_2branch(exp: str, split: str, label: str):
    """fa_speed: (fa_dir, speed_dir), fa_vel: (fa_dir, vel_dir)"""
    mod = MODALITY
    base_label = os.path.join(_base_dir(), split, label)
    cv = _cv()
    fm = FM_EXPORT_ROOT
    d_fa = _find_mod_dir(base_label) or os.path.join(base_label, mod)
    if cv:
        fm_cv = os.path.join(fm, cv, mod.upper())
        d_speed = os.path.join(fm_cv, "speed_t0.80", split, label, mod.upper())
        d_vel = os.path.join(fm_cv, "vel_t0.80", split, label, mod.upper())
    else:
        d_speed = os.path.join(fm, "speed_t0.80", split, label, mod.upper())
        d_vel = os.path.join(fm, "vel_t0.80", split, label, mod.upper())
    if exp == "fa_speed":
        return (d_fa, d_speed)
    if exp == "fa_vel":
        return (d_fa, d_vel)
    raise ValueError(f"get_split_dirs_2branch: {exp}")


def get_split_dirs_multichannel(exp: str, split: str, label: str):
    """fa_velspeed: (speed_dir, vel_dir). fa_speed_vel: (fa_dir, speed_dir, vel_dir)"""
    mod = MODALITY
    base = _base_dir()
    fm = FM_EXPORT_ROOT
    base_label = os.path.join(base, split, label)
    cv = _cv()
    if cv:
        fm_cv = os.path.join(fm, cv, mod.upper())
        d_fa = _find_mod_dir(base_label) or os.path.join(base_label, mod)
        d_speed = os.path.join(fm_cv, "speed_t0.80", split, label, mod.upper())
        d_vel = os.path.join(fm_cv, "vel_t0.80", split, label, mod.upper())
    else:
        d_fa = _find_mod_dir(base_label) or os.path.join(base_label, mod)
        d_speed = os.path.join(fm, "speed_t0.80", split, label, mod.upper())
        d_vel = os.path.join(fm, "vel_t0.80", split, label, mod.upper())

    if exp == "fa_velspeed":
        return (d_speed, d_vel)
    if exp == "fa_speed_vel":
        return (d_fa, d_speed, d_vel)
    raise ValueError(f"get_split_dirs_multichannel: {exp}")


def is_parallel_exp(exp: str) -> bool:
    return exp == "fa_speed_vel"  # only 3 parallel branches

def is_2channel_concat_exp(exp: str) -> bool:
    """fa_speed, fa_vel, fa_velspeed: 2 channel single tensor (like compare script)"""
    return exp in ("fa_speed", "fa_vel", "fa_velspeed")


def is_multichannel_concat_exp(exp: str) -> bool:
    return exp == "fa_velspeed"


# =========================
# 3) Dataset
# =========================
class Nifti3DClassificationDataset(Dataset):
    def __init__(self, exp: str, split="Train", use_zscore=True):
        self.exp = exp
        self.split = split
        self.use_zscore = use_zscore
        self.items = []

        if exp in _RAW_MOD_EXPS + ("speed", "vel"):
            for label_str in ["0", "1"]:
                in_dir = get_split_dir(exp, split, label_str)
                if in_dir is None or not os.path.isdir(in_dir):
                    continue
                for f in sorted(glob.glob(os.path.join(in_dir, "*.nii*"))):
                    self.items.append({"y": int(label_str), "id": extract_id_from_filename(f), "path": f})
        elif exp in ("fa_speed", "fa_vel"):
            df, ds = get_split_dirs_2branch(exp, split, "0")
            for label_str in ["0", "1"]:
                df, ds = get_split_dirs_2branch(exp, split, label_str)
                files_fa = sorted(glob.glob(os.path.join(df, "*.nii*")))
                other_map = {extract_id_from_filename(f): f for f in glob.glob(os.path.join(ds, "*.nii*"))}
                for ff in files_fa:
                    pid = extract_id_from_filename(ff)
                    if pid in other_map:
                        self.items.append({"y": int(label_str), "id": pid, "path_a": ff, "path_b": other_map[pid]})
        elif exp == "fa_velspeed":
            d_speed, d_vel = get_split_dirs_multichannel(exp, split, "0")
            for label_str in ["0", "1"]:
                ds, dv = get_split_dirs_multichannel(exp, split, label_str)
                files_s = sorted(glob.glob(os.path.join(ds, "*.nii*")))
                vel_map = {extract_id_from_filename(f): f for f in glob.glob(os.path.join(dv, "*.nii*"))}
                for fs in files_s:
                    pid = extract_id_from_filename(fs)
                    if pid in vel_map:
                        self.items.append({"y": int(label_str), "id": pid, "path_speed": fs, "path_vel": vel_map[pid]})
        elif exp == "fa_speed_vel":
            d_fa, d_speed, d_vel = get_split_dirs_multichannel(exp, split, "0")
            for label_str in ["0", "1"]:
                df, ds, dv = get_split_dirs_multichannel(exp, split, label_str)
                files_fa = sorted(glob.glob(os.path.join(df, "*.nii*")))
                speed_map = {extract_id_from_filename(f): f for f in glob.glob(os.path.join(ds, "*.nii*"))}
                vel_map = {extract_id_from_filename(f): f for f in glob.glob(os.path.join(dv, "*.nii*"))}
                for ff in files_fa:
                    pid = extract_id_from_filename(ff)
                    if pid in speed_map and pid in vel_map:
                        self.items.append({
                            "y": int(label_str), "id": pid,
                            "path_fa": ff, "path_speed": speed_map[pid], "path_vel": vel_map[pid]
                        })

        if not self.items:
            if exp in _RAW_MOD_EXPS + ("speed", "vel"):
                sample_dir = get_split_dir(exp, split, "0")
            elif exp in ("fa_speed", "fa_vel"):
                sample_dir = get_split_dirs_2branch(exp, split, "0")
            else:
                sample_dir = get_split_dirs_multichannel(exp, split, "0")
            raise ValueError(
                f"0 samples: {exp} {split}\n"
                f"Searched path: {sample_dir}\n"
                f"MODALITY={MODALITY} - Was FM export done with this modality? (If AD export, MODALITY='ad')"
            )
        print(f"[{split}] exp={exp} N={len(self.items)}")

    def __len__(self):
        return len(self.items)

    def _load(self, path):
        x = nib.load(path).get_fdata().astype(np.float32)
        if self.use_zscore:
            x = robust_zscore(x)
        return x[None, ...]

    def __getitem__(self, idx):
        it = self.items[idx]
        y = int(it["y"])
        pid = it["id"]

        if self.exp in _RAW_MOD_EXPS + ("speed", "vel"):
            x = self._load(it["path"])
        elif self.exp in ("fa_speed", "fa_vel"):
            xa = self._load(it["path_a"])
            xb = self._load(it["path_b"])
            x = np.concatenate([xa, xb], axis=0)  # (2,D,H,W) - 2 channel like compare script
        elif self.exp == "fa_velspeed":
            xs = self._load(it["path_speed"])
            xv = self._load(it["path_vel"])
            x = np.concatenate([xs, xv], axis=0)
        else:
            xf = self._load(it["path_fa"])
            xs = self._load(it["path_speed"])
            xv = self._load(it["path_vel"])
            x = (xf, xs, xv)

        x = torch.from_numpy(x) if isinstance(x, np.ndarray) else tuple(torch.from_numpy(t) for t in x)
        return {"x": x, "y": torch.tensor(y, dtype=torch.long), "id": pid}


# =========================
# 4) Model
# =========================
class ChannelAttention3D(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        mid = max(channels // ratio, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid), nn.ReLU(inplace=True), nn.Linear(mid, channels)
        )

    def forward(self, x):
        b, c = x.size(0), x.size(1)
        avg = x.mean(dim=(2, 3, 4)).view(b, c)
        max_ = x.amax(dim=(2, 3, 4)).view(b, c)
        w = torch.sigmoid(self.fc(avg) + self.fc(max_))
        return x * w.view(b, c, 1, 1, 1)


class SpatialAttention3D(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, k, padding=k // 2)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        max_ = x.amax(dim=1, keepdim=True)
        w = torch.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))
        return x * w


class CBAM3D(nn.Module):
    def __init__(self, ch, ratio=8):
        super().__init__()
        self.ca = ChannelAttention3D(ch, ratio)
        self.sa = SpatialAttention3D()

    def forward(self, x):
        return self.sa(self.ca(x))


class ResidualBlock3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv3d(ch, ch, 3, padding=1), nn.BatchNorm3d(ch), nn.ReLU(inplace=True))
        self.c2 = nn.Sequential(nn.Conv3d(ch, ch, 3, padding=1), nn.BatchNorm3d(ch))

    def forward(self, x):
        return F.relu(self.c2(self.c1(x)) + x)


class ResidualBlock3D_CBAM(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        s = 2 if downsample else 1
        self.c1 = nn.Sequential(nn.Conv3d(in_ch, out_ch, 3, padding=1), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True))
        self.c2 = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, 3, padding=1), nn.BatchNorm3d(out_ch),
            nn.AvgPool3d(3, stride=2, padding=1) if downsample else nn.Identity()
        )
        self.shortcut = nn.Sequential(nn.Conv3d(in_ch, out_ch, 1, stride=s), nn.BatchNorm3d(out_ch))
        self.cbam = CBAM3D(out_ch)

    def forward(self, x):
        out = self.cbam(self.c2(self.c1(x)))
        return F.relu(out + self.shortcut(x))


class Block4_CBAM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1), nn.BatchNorm3d(out_ch)
        )
        self.cbam = CBAM3D(out_ch)
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        return self.pool(self.cbam(F.relu(self.conv(x))))


L = [32, 16, 16, 16, 16, 16, 16, 64, 128, 256, 512, 256, 256, 256]  # 14 elem (head: L[11]->L[12]->L[13]->num_classes)


class ResNet3D_CBAM(nn.Module):
    def __init__(self, in_ch=1, num_classes=2):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_ch, L[0], 3, padding=1), nn.BatchNorm3d(L[0]), nn.ReLU(inplace=True))
        self.pool1 = nn.AvgPool3d(3, stride=2, padding=1)
        self.block1 = nn.Sequential(nn.Conv3d(L[0], L[1], 3, padding=1), nn.BatchNorm3d(L[1]), nn.ReLU(inplace=True))
        self.b2_1 = ResidualBlock3D(L[2])
        self.b2_2 = ResidualBlock3D(L[3])
        self.b2_3 = ResidualBlock3D(L[4])
        self.b2_4 = ResidualBlock3D(L[5])
        self.b2_5 = ResidualBlock3D(L[6])
        self.b3_1 = ResidualBlock3D_CBAM(L[6], L[7], downsample=True)
        self.b3_2 = ResidualBlock3D_CBAM(L[7], L[8], downsample=True)
        self.b3_3 = ResidualBlock3D_CBAM(L[8], L[9], downsample=True)
        self.b3_4 = ResidualBlock3D_CBAM(L[9], L[10], downsample=True)
        self.b4 = Block4_CBAM(L[10], L[11])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(L[11], L[12]), nn.BatchNorm1d(L[12]), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(L[12], L[13]), nn.BatchNorm1d(L[13]), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(L[13], num_classes),
        )

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.block1(x)
        for b in [self.b2_1, self.b2_2, self.b2_3, self.b2_4, self.b2_5, self.b3_1, self.b3_2, self.b3_3, self.b3_4]:
            x = b(x)
        x = self.b4(x)
        return self.head(x)

    def forward_features(self, x):
        """Feature vector (L[11]=256) - for parallel branch"""
        x = self.pool1(self.conv1(x))
        x = self.block1(x)
        for b in [self.b2_1, self.b2_2, self.b2_3, self.b2_4, self.b2_5, self.b3_1, self.b3_2, self.b3_3, self.b3_4]:
            x = b(x)
        x = self.b4(x)
        return x.view(x.size(0), -1)


class ParallelBranchesModel(nn.Module):
    """N single-channel branches, forward_features (L[11]=256 dim) concat, FC"""
    def __init__(self, num_branches=2, feat_dim=256, num_classes=2, drop=0.2):
        super().__init__()
        self.branches = nn.ModuleList([ResNet3D_CBAM(in_ch=1, num_classes=2) for _ in range(num_branches)])
        self.fc = nn.Sequential(
            nn.Linear(num_branches * feat_dim, 256),
            nn.ReLU(inplace=True), nn.Dropout(drop),
            nn.Linear(256, num_classes),
        )
        self.feat_dim = feat_dim
        self.num_branches = num_branches

    def forward(self, x_tuple):
        feats = []
        for i, br in enumerate(self.branches):
            xi = x_tuple[i] if isinstance(x_tuple, (list, tuple)) else x_tuple
            feats.append(br.forward_features(xi))
        out = torch.cat(feats, dim=1)
        return self.fc(out)


# =========================
# 5) Metrics
# =========================
@dataclass
class EvalResult:
    loss: float
    acc: float
    tp: int
    tn: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    balanced_acc: float


def compute_confusion(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def metrics_from_confusion(tp, tn, fp, fn):
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    tnr = tn / (tn + fp + eps)
    bal = 0.5 * (recall + tnr)
    return float(precision), float(recall), float(f1), float(bal)


def _to_device(x, device):
    if isinstance(x, (list, tuple)):
        return tuple(t.to(device) for t in x)
    return x.to(device)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_n = 0.0, 0
    ys, ps = [], []
    for batch in loader:
        x = _to_device(batch["x"], device)
        y = batch["y"].to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        pred = torch.argmax(logits, dim=1)
        total_loss += float(loss.item()) * y.size(0)
        total_n += y.size(0)
        ys.append(y.cpu())
        ps.append(pred.cpu())
    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(ps).numpy()
    tp, tn, fp, fn = compute_confusion(y_true, y_pred)
    prec, rec, f1, bal = metrics_from_confusion(tp, tn, fp, fn)
    return EvalResult(loss=total_loss / max(1, total_n), acc=float((y_true == y_pred).mean()),
                     tp=tp, tn=tn, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1, balanced_acc=bal)


def is_better_by_acc(te_acc, te_loss, best_acc, best_loss, delta=1e-12):
    if te_acc > best_acc + delta:
        return True
    if abs(te_acc - best_acc) <= delta and te_loss < best_loss - delta:
        return True
    return False


# =========================
# 6) Train
# =========================
def train_one_run(exp: str, run_idx: int, cv_fold: str = None):
    global _current_cv, MODALITY
    _current_cv = cv_fold
    # wm_speed, md_vel etc. -> update MODALITY, use base_exp for logic
    mod_from_exp, base_exp = _parse_exp_modality(exp)
    if mod_from_exp is not None:
        MODALITY = mod_from_exp
    exp_logic = base_exp  # for Dataset, model
    run_seed = set_random_mode()
    ensure_dir(OUT_PARENT)

    if SAME_FOLDER_RUNS:
        exp_dir = ensure_dir(os.path.join(OUT_PARENT, exp))
        if cv_fold:
            exp_dir = ensure_dir(os.path.join(exp_dir, cv_fold))
        out_dir = ensure_dir(os.path.join(exp_dir, f"run_{run_idx}"))
    else:
        out_dir = ensure_dir(os.path.join(OUT_PARENT, f"{exp}_{now_tag()}"))

    print("\n" + "=" * 50)
    print(f"RUN {run_idx}/{NUM_RUNS} | {exp} | {cv_fold or 'flat'} | {out_dir}")
    print("=" * 50)
    # Print data paths (for single channel exp)
    if exp_logic in _RAW_MOD_EXPS + ("speed", "vel", "fa_speed", "fa_vel"):
        for split in ["Train", "Test"]:
            for label in ["0", "1"]:
                p = get_split_dir(exp_logic, split, label)
                exists = "✓" if os.path.isdir(p) else "✗"
                print(f"  [{exists}] {split}/{label}: {p}")

    train_ds = Nifti3DClassificationDataset(exp=exp_logic, split="Train", use_zscore=USE_ZSCORE)
    test_ds = Nifti3DClassificationDataset(exp=exp_logic, split="Test", use_zscore=USE_ZSCORE)

    collate_fn = collate_fn_parallel if is_parallel_exp(exp_logic) else None  # only for fa_speed_vel
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=(NUM_WORKERS > 0),
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=collate_fn)

    if is_parallel_exp(exp_logic):
        model = ParallelBranchesModel(num_branches=3, feat_dim=256, num_classes=2).to(DEVICE)
    else:
        in_ch = 2 if is_2channel_concat_exp(exp_logic) else 1  # fa_speed, fa_vel, fa_velspeed -> 2 channel
        model = ResNet3D_CBAM(in_ch=in_ch, num_classes=2).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=LR_FACTOR, patience=LR_PLATEAU_PATIENCE, min_lr=LR_MIN
    )

    cfg = {"exp": exp, "run_idx": run_idx, "run_seed": run_seed, "root": ROOT, "cv_fold": cv_fold or CV_FOLD}
    save_json(os.path.join(out_dir, "config.json"), cfg)

    header = ["epoch", "train_loss", "train_acc", "test_loss", "test_acc",
              "test_precision", "test_recall", "test_f1", "test_balanced_acc", "TP", "TN", "FP", "FN", "lr"]
    metrics_csv = os.path.join(out_dir, "metrics_epoch.csv")
    best_path = os.path.join(out_dir, "best.pt")
    last_path = os.path.join(out_dir, "last.pt")

    best_acc, best_loss, best_epoch, best_row, last_row = -1.0, float("inf"), 0, None, None
    early_left = EARLY_STOP_PATIENCE

    for ep in range(1, EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        t0 = time.time()

        for batch in train_loader:
            x = _to_device(batch["x"], DEVICE)
            y = batch["y"].to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running_loss += float(loss.item()) * y.size(0)
            correct += int((torch.argmax(logits, 1) == y).sum().item())
            total += y.size(0)

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        test_res = evaluate(model, test_loader, DEVICE)
        scheduler.step(test_res.acc)
        cur_lr = float(opt.param_groups[0]["lr"])

        print(f"Ep {ep}/{EPOCHS} | tr_loss={train_loss:.4f} tr_acc={train_acc:.4f} | "
              f"te_acc={test_res.acc:.4f} | lr={cur_lr:.2e} | {time.time()-t0:.1f}s")

        torch.save({"epoch": ep, "model_state_dict": model.state_dict(), "cfg": cfg}, last_path)

        row = {"epoch": ep, "train_loss": f"{train_loss:.6f}", "train_acc": f"{train_acc:.6f}",
               "test_loss": f"{test_res.loss:.6f}", "test_acc": f"{test_res.acc:.6f}",
               "test_precision": f"{test_res.precision:.6f}", "test_recall": f"{test_res.recall:.6f}",
               "test_f1": f"{test_res.f1:.6f}", "test_balanced_acc": f"{test_res.balanced_acc:.6f}",
               "TP": test_res.tp, "TN": test_res.tn, "FP": test_res.fp, "FN": test_res.fn, "lr": f"{cur_lr:.8e}"}
        append_csv_row(metrics_csv, header, row)
        last_row = row

        if is_better_by_acc(test_res.acc, test_res.loss, best_acc, best_loss):
            best_acc, best_loss, best_epoch, best_row = test_res.acc, test_res.loss, ep, row
            torch.save({"epoch": ep, "best_test_acc": best_acc, "model_state_dict": model.state_dict(), "cfg": cfg}, best_path)
            print(f"  ✅ BEST ep={ep} acc={best_acc:.4f}")
            early_left = EARLY_STOP_PATIENCE
        else:
            early_left -= 1

        if USE_EARLY_STOP and early_left <= 0:
            print(f"🛑 Early stop")
            break

    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
    final = evaluate(model, test_loader, DEVICE)
    save_json(os.path.join(out_dir, "summary.json"), {
        "best_epoch": best_epoch, "best_test_acc": float(best_acc),
        "final_test": asdict(final), "run_seed": run_seed
    })
    if best_row is None:
        best_row = last_row
    write_best_last_csv(out_dir, header, best_row, last_row)
    try:
        plot_curves(metrics_csv, out_dir)
    except Exception as e:
        print("[WARN] plot:", e)
    print("DONE:", out_dir)
    return out_dir


# =========================
# 7) MAIN
# =========================
if __name__ == "__main__":
    import gc
    if RUN_ALL_CVS:
        cv_list = CV_LIST if CV_LIST else get_cv_folds()
        print(f"CV folds: {cv_list} | Each {NUM_RUNS} runs")
        for cv_fold in cv_list:
            for run_idx in range(1, NUM_RUNS + 1):
                train_one_run(EXPERIMENT, run_idx, cv_fold=cv_fold)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    else:
        for run_idx in range(1, NUM_RUNS + 1):
            train_one_run(EXPERIMENT, run_idx, cv_fold=CV_FOLD)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()