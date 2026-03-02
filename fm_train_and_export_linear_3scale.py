# -*- coding: utf-8 -*-
"""
fm_train_and_export_linear_3scale.py
Flow Matching training + velocity/speed export for GPU.
For each CV under Dataset_CrossValidation (CV1, CV2, ...).

INPUT:  /content/drive/MyDrive/MICCAI/Dataset_CrossValidation/CV1, CV2, ...
        Each CV: Train/0|1/{AD,MD,RD,FA,WM,GM,MRI}/, Test/0|1/{...}/

OUTPUT: /content/drive/MyDrive/MICCAI/flow_matching_velocity_speed_t0.80/
        CV1/AD/checkpoints/, vel_t0.80/, speed_t0.80/
        CV1/MD/, RD/, FA/, WM/, GM/, MRI/
        CV2/...
        ...

Author: Miccai_5340
Code written with assistance from Cursor AI.
Code written with assistance from Cursor AI.
"""

import os, glob, time, hashlib, gc, re, shutil
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================
# 0) CONFIG - GPU
# =========================
ROOT = "/content/drive/MyDrive/MICCAI"
DATASET_CV_DIR = "Dataset_CrossValidation"   # under ROOT

# Output folder: velocity and speed maps from Flow Matching
OUT_ROOT_NAME = "flow_matching_velocity_speed_t0.80"
# Checkpoints always on Drive (persist after session ends)
OUT_ROOT = os.path.join(ROOT, OUT_ROOT_NAME)
# EXPORT_TO_LOCAL=True -> vel/speed under /content, no Drive I/O. Download zip when done.
EXPORT_TO_LOCAL = True
EXPORT_OUT_ROOT = os.path.join("/content", OUT_ROOT_NAME) if EXPORT_TO_LOCAL else OUT_ROOT

# Only CV1 and AD (leave empty to run all)
ONLY_CV = ["CV1"]
ONLY_MODALITIES = ["AD"]
ALL_MODALITIES = ["AD", "MD", "RD", "FA", "WM", "GM", "MRI"]
MODALITIES = ONLY_MODALITIES if ONLY_MODALITIES else ALL_MODALITIES

# TVAL: Flow Matching "time" param (0=noise, 1=real image).
# t=0.80: take velocity field at 80% along path from noise to image.
# Velocity vector and magnitude (speed) at this point are used as CNN input.
TVAL = 0.80

# Increased if loss still decreasing at 150
EPOCHS = 300
LR = 2e-4
PATIENCE = 80
BATCH_SIZE = 16
NUM_WORKERS = 8
PIN_MEMORY = True

SEED = 0
USE_ZSCORE = True   # False produces noise

USE_CPU = False
FORCE_GPU = True

def _get_device():
    if USE_CPU:
        return "cpu"
    if not torch.cuda.is_available():
        print("CUDA not available -> CPU")
        return "cpu"
    if FORCE_GPU:
        print("FORCE_GPU=True -> using cuda")
        return "cuda"
    try:
        x = torch.randn(2, 1, 16, 16, 16).cuda()
        _ = torch.randn_like(x)
        return "cuda"
    except RuntimeError as e:
        if "no kernel image" in str(e) or "sm_" in str(e).lower():
            print("GPU not PyTorch compatible -> CPU")
            return "cpu"
        raise
DEVICE = _get_device()
# Export: GPU much faster.
USE_CPU_FOR_EXPORT = False   # False = GPU (fast), True = CPU
EXPORT_BATCH_SIZE = 8        # Batch inference - 4/8/16, depends on GPU RAM

# Training only, stop when done. Run export separately later.
RUN_TRAINING_ONLY = True   # True = Phase 1 only, stop. False = training + export

# Delete specific CV/modality checkpoint (e.g. if CV9/WM corrupted)
CLEAN_INCOMPLETE = False
CLEAN_INCOMPLETE_SPECIFIC = []  

# If ONLY_CV/ONLY_MODALITIES set, only those run. Else SKIP_CV used.
SKIP_CV = []
FORCE_RETRAIN = [("CV1", "AD")]   # overwrite checkpoint if exists


# =========================
# 1) UTILS
# =========================
def set_deterministic(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_id_from_filename(fname: str) -> str:
    base = os.path.basename(fname)
    parts = base.split("_")
    if len(parts) < 2:
        raise ValueError(f"ID could not be parsed: {base}")
    pid = parts[1].split(".")[0]
    return pid


def robust_zscore(x, eps=1e-6):
    m = float(np.mean(x))
    s = float(np.std(x))
    return (x - m) / (s + eps)


def deterministic_noise_like(shape, pid: str):
    h = hashlib.md5(pid.encode("utf-8")).hexdigest()
    seed = int(h[:8], 16)
    rng = np.random.RandomState(seed)
    x0 = rng.randn(*shape).astype(np.float32)
    return x0


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def _save_nii_with_retry(img, path, max_retries=5):
    """Retry on Drive I/O error"""
    for attempt in range(max_retries):
        try:
            nib.save(img, path)
            return
        except OSError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s, 8s
            else:
                raise


def _cv_sort_key(name):
    """CV1, CV2, CV10 -> (1,), (2,), (10,) for natural sort order"""
    m = re.match(r"CV(\d+)", name, re.IGNORECASE)
    return (int(m.group(1)),) if m else (999,)


def get_cv_folds():
    """Find CV1, CV2, CV3, CV10 ... folders under Dataset_CrossValidation (natural order)."""
    cv_base = os.path.join(ROOT, DATASET_CV_DIR)
    if not os.path.exists(cv_base):
        raise FileNotFoundError(f"Dataset_CrossValidation not found: {cv_base}")
    folds = [d for d in os.listdir(cv_base)
             if os.path.isdir(os.path.join(cv_base, d)) and d.upper().startswith("CV")]
    folds = sorted(folds, key=_cv_sort_key)
    return [os.path.join(cv_base, f) for f in folds]


# =========================
# 2) DATASET
# =========================
def _find_modality_dir(root, split, label_str, modality):
    """Find modality folder - AD/ad, FA/fa etc. case may differ."""
    base = os.path.join(root, split, label_str)
    if not os.path.exists(base):
        return None
    # First exact match
    cand = os.path.join(base, modality)
    if os.path.isdir(cand):
        return cand
    # Then match folder names (case-insensitive)
    for name in os.listdir(base):
        if os.path.isdir(os.path.join(base, name)) and name.upper() == modality.upper():
            return os.path.join(base, name)
    return None


class StrokeMapsDataset(Dataset):
    """root/split/{0,1}/{AD,FA,...}/*.nii* - modality name case-insensitive"""

    def __init__(self, root, split="Train", modality="FA"):
        self.root = root
        self.split = split
        self.modality = modality
        self.items = []

        for label_str in ["0", "1"]:
            label = int(label_str)
            in_dir = _find_modality_dir(root, split, label_str, modality)
            if in_dir is None:
                base = os.path.join(root, split, label_str)
                if os.path.exists(base):
                    subdirs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
                    raise FileNotFoundError(
                        f"Modality '{modality}' not found: {base}\n"
                        f"Existing folders: {subdirs}"
                    )
                raise FileNotFoundError(f"Folder does not exist: {base}")
            files = sorted(glob.glob(os.path.join(in_dir, "*.nii*")))
            for f in files:
                pid = extract_id_from_filename(f)
                self.items.append((label, pid, f))

        if len(self.items) == 0:
            raise ValueError(
                f"0 samples found: {root}/{split}/0|1/{modality}\n"
                f"Folder exists but no .nii files. Is path correct?"
            )
        print(f"[{split}] sample count: {len(self.items)} | modality: {modality}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        label, pid, path = self.items[idx]
        img = nib.load(path)
        x = img.get_fdata().astype(np.float32)
        if USE_ZSCORE:
            x = robust_zscore(x)
        x = x[None, ...]
        x = torch.from_numpy(x)
        return {"x": x, "y": torch.tensor(label, dtype=torch.long), "id": pid, "path": path}


# =========================
# 3) 3-SCALE 3D U-NET
# =========================
def match_size(src, ref):
    _, _, Ds, Hs, Ws = src.shape
    _, _, Dr, Hr, Wr = ref.shape
    if Ds >= Dr and Hs >= Hr and Ws >= Wr:
        d0 = (Ds - Dr) // 2
        h0 = (Hs - Hr) // 2
        w0 = (Ws - Wr) // 2
        return src[:, :, d0:d0+Dr, h0:h0+Hr, w0:w0+Wr]
    pd = max(Dr - Ds, 0)
    ph = max(Hr - Hs, 0)
    pw = max(Wr - Ws, 0)
    pad = (pw//2, pw - pw//2, ph//2, ph - ph//2, pd//2, pd - pd//2)
    src = F.pad(src, pad, mode="constant", value=0.0)
    _, _, Ds2, Hs2, Ws2 = src.shape
    d0 = (Ds2 - Dr) // 2
    h0 = (Hs2 - Hr) // 2
    w0 = (Ws2 - Wr) // 2
    return src[:, :, d0:d0+Dr, h0:h0+Hr, w0:w0+Wr]


class TimeMLP(nn.Module):
    def __init__(self, tdim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, tdim), nn.SiLU(),
            nn.Linear(tdim, tdim), nn.SiLU()
        )
    def forward(self, t):
        return self.net(t.view(-1, 1))


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.c2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
    def forward(self, x):
        x = F.silu(self.c1(x))
        x = F.silu(self.c2(x))
        return x


class UNet3Scale(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=16, tdim=64):
        super().__init__()
        self.tmlp = TimeMLP(tdim)
        self.t1 = nn.Linear(tdim, base)
        self.t2 = nn.Linear(tdim, base*2)
        self.t3 = nn.Linear(tdim, base*4)
        self.t4 = nn.Linear(tdim, base*2)
        self.t5 = nn.Linear(tdim, base)
        self.enc1 = Block(in_ch, base)
        self.down1 = nn.Conv3d(base, base*2, 3, stride=2, padding=1)
        self.enc2 = Block(base*2, base*2)
        self.down2 = nn.Conv3d(base*2, base*4, 3, stride=2, padding=1)
        self.mid  = Block(base*4, base*4)
        self.up2 = nn.ConvTranspose3d(base*4, base*2, 4, stride=2, padding=1)
        self.dec2 = Block(base*4, base*2)
        self.up1 = nn.ConvTranspose3d(base*2, base, 4, stride=2, padding=1)
        self.dec1 = Block(base*2, base)
        self.out = nn.Conv3d(base, out_ch, 3, padding=1)

    def forward(self, x, t):
        B = x.shape[0]
        te = self.tmlp(t)
        h1 = self.enc1(x) + self.t1(te).view(B, -1, 1, 1, 1)
        d1 = F.silu(self.down1(h1))
        h2 = self.enc2(d1) + self.t2(te).view(B, -1, 1, 1, 1)
        d2 = F.silu(self.down2(h2))
        h3 = self.mid(d2) + self.t3(te).view(B, -1, 1, 1, 1)
        u2 = F.silu(self.up2(h3) + self.t4(te).view(B, -1, 1, 1, 1))
        u2 = match_size(u2, h2)
        cat2 = torch.cat([u2, h2], dim=1)
        y2 = self.dec2(cat2)
        u1 = F.silu(self.up1(y2) + self.t5(te).view(B, -1, 1, 1, 1))
        u1 = match_size(u1, h1)
        cat1 = torch.cat([u1, h1], dim=1)
        y1 = self.dec1(cat1)
        return self.out(y1)


# =========================
# 4) LINEAR FLOW LOSS
# =========================
def linear_flow_loss(model, x1, device):
    x1 = x1.to(device)
    x0 = torch.randn_like(x1)
    B = x1.shape[0]
    t = torch.rand(B, device=device)
    tb = t.view(B, 1, 1, 1, 1)
    xt = (1 - tb) * x0 + tb * x1
    target = x1 - x0
    v = model(xt, t)
    return F.mse_loss(v, target)


# =========================
# 5) TRAIN FM (per CV, per modality)
# =========================
def train_fm_linear_3scale(cv_root, out_mod_root, modality):
    print("Device:", DEVICE, "| Modality:", modality)
    set_deterministic(SEED)

    train_ds = StrokeMapsDataset(cv_root, split="Train", modality=modality)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0)
    )

    model = UNet3Scale(in_ch=1, out_ch=1, base=16, tdim=64).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    ckpt_dir = ensure_dir(os.path.join(out_mod_root, "checkpoints"))
    best_path = os.path.join(ckpt_dir, "fm_best.pt")
    last_path = os.path.join(ckpt_dir, "fm_last.pt")

    best_loss = float("inf")
    epochs_without_improvement = 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        running = 0.0

        for batch in train_loader:
            x = batch["x"]
            loss = linear_flow_loss(model, x, DEVICE)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item())

        epoch_loss = running / max(1, len(train_loader))
        dt = time.time() - t0
        print(f"Epoch {ep}/{EPOCHS} | loss={epoch_loss:.6f} | time={dt:.1f}s")

        payload = {
            "epoch": ep,
            "best_loss": float(best_loss),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "cfg": {"modality": modality, "base": 16, "tdim": 64, "lr": LR, "epochs": EPOCHS}
        }
        torch.save(payload, last_path)

        if epoch_loss < best_loss - 1e-12:
            best_loss = epoch_loss
            epochs_without_improvement = 0
            payload["best_loss"] = float(best_loss)
            torch.save(payload, best_path)
            print(f"  ✅ New BEST saved (best_loss={best_loss:.6f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"  Early stop: no improvement for {PATIENCE} epochs. (ep={ep})")
                break

    print("Training done. Best:", best_path)
    return best_path


def load_fm_checkpoint(ckpt_path, device):
    model = UNet3Scale(in_ch=1, out_ch=1, base=16, tdim=64).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# =========================
# 6) EXPORT (per CV, per modality)
# =========================
@torch.no_grad()
def export_all_vel_and_speed(model, cv_root, cv_name, modality):
    """Load checkpoint from Drive, write vel/speed to EXPORT_OUT_ROOT."""
    export_device = "cpu" if USE_CPU_FOR_EXPORT else DEVICE
    model = model.to(export_device)
    print(f"  Export device: {export_device} | batch={EXPORT_BATCH_SIZE}")

    export_mod_root = _export_root(cv_name, modality)
    vel_root = ensure_dir(os.path.join(export_mod_root, f"vel_t{TVAL:.2f}"))
    spd_root = ensure_dir(os.path.join(export_mod_root, f"speed_t{TVAL:.2f}"))

    items = []
    for split in ["Train", "Test"]:
        ds = StrokeMapsDataset(cv_root, split=split, modality=modality)
        for (label, pid, path) in ds.items:
            items.append((split, label, pid, path))

    total = 0
    for b in range(0, len(items), EXPORT_BATCH_SIZE):
        batch_items = items[b : b + EXPORT_BATCH_SIZE]
        xt_list, meta = [], []
        for (split, label, pid, path) in batch_items:
            img = nib.load(path)
            affine, header = img.affine, img.header
            x = img.get_fdata().astype(np.float32)
            if USE_ZSCORE:
                x = robust_zscore(x)
            x1_np = x[None, ...]
            x0_np = deterministic_noise_like(x1_np.shape, pid)
            x1 = torch.from_numpy(x1_np).unsqueeze(0).to(export_device)
            x0 = torch.from_numpy(x0_np).unsqueeze(0).to(export_device)
            t = torch.tensor([TVAL], device=export_device, dtype=torch.float32)
            tb = t.view(1, 1, 1, 1, 1)
            xt = (1 - tb) * x0 + tb * x1
            xt_list.append(xt)
            meta.append((split, label, path, affine, header))

        xt_batch = torch.cat(xt_list, dim=0)
        t_batch = torch.full((len(batch_items),), TVAL, device=export_device, dtype=torch.float32)
        v_batch = model(xt_batch, t_batch)

        for i, (split, label, path, affine, header) in enumerate(meta):
            v_np = v_batch[i].squeeze(0).cpu().numpy().astype(np.float32)
            spd_np = np.abs(v_np).astype(np.float32)
            out_vel_dir = ensure_dir(os.path.join(vel_root, split, str(label), modality))
            out_spd_dir = ensure_dir(os.path.join(spd_root, split, str(label), modality))
            base_noext = os.path.basename(path).replace(".nii.gz", "").replace(".nii", "")
            out_v = os.path.join(out_vel_dir, f"{base_noext}_vel_t{TVAL:.2f}.nii.gz")
            out_s = os.path.join(out_spd_dir, f"{base_noext}_spd_t{TVAL:.2f}.nii.gz")
            _save_nii_with_retry(nib.Nifti1Image(v_np, affine, header=header), out_v)
            _save_nii_with_retry(nib.Nifti1Image(spd_np, affine, header=header), out_s)
            total += 1

        if total % 25 == 0:
            print(f"[EXPORT] {total} done...")

    print("Export done. Velocity:", vel_root, "| Speed:", spd_root, "| Total:", total)


# =========================
# 7) MAIN - First all trainings, then all exports
# =========================
def _ckpt_root(cv_name, mod):
    """Checkpoint path (always Drive)"""
    return os.path.join(OUT_ROOT, cv_name, mod)


def _export_root(cv_name, mod):
    """Export output path (EXPORT_TO_LOCAL -> /content, else Drive)"""
    return os.path.join(EXPORT_OUT_ROOT, cv_name, mod)


def is_modality_done(cv_name, mod):
    """Training + export completed?"""
    best_pt = os.path.join(_ckpt_root(cv_name, mod), "checkpoints", "fm_best.pt")
    vel_dir = os.path.join(_export_root(cv_name, mod), f"vel_t{TVAL:.2f}")
    if not os.path.exists(best_pt):
        return False
    if not os.path.isdir(vel_dir):
        return False
    nii_files = glob.glob(os.path.join(vel_dir, "**", "*.nii*"))
    return len(nii_files) > 0


def needs_training(cv_name, mod):
    """Training needed? (no checkpoint)"""
    best_pt = os.path.join(_ckpt_root(cv_name, mod), "checkpoints", "fm_best.pt")
    return not os.path.exists(best_pt)


def needs_export(cv_name, mod):
    """Export needed? (checkpoint exists, vel missing or empty)"""
    best_pt = os.path.join(_ckpt_root(cv_name, mod), "checkpoints", "fm_best.pt")
    vel_dir = os.path.join(_export_root(cv_name, mod), f"vel_t{TVAL:.2f}")
    return os.path.exists(best_pt) and (not os.path.isdir(vel_dir) or len(glob.glob(os.path.join(vel_dir, "**", "*.nii*"))) == 0)


def clean_incomplete_checkpoints():
    """Delete ONLY those in CLEAN_INCOMPLETE_SPECIFIC. If empty, do nothing."""
    if not CLEAN_INCOMPLETE_SPECIFIC:
        return
    for cv_mod in CLEAN_INCOMPLETE_SPECIFIC:  # e.g. ["CV9/WM"]
        parts = cv_mod.split("/")
        if len(parts) != 2:
            continue
        cv_name, mod = parts[0], parts[1]
        out_mod = os.path.join(OUT_ROOT, cv_name, mod)
        ckpt_dir = os.path.join(out_mod, "checkpoints")
        if os.path.isdir(ckpt_dir):
            import shutil
            shutil.rmtree(ckpt_dir)
            print(f"Deleted (only {cv_mod}): {ckpt_dir}")


def main():
    if DEVICE == "cuda":
        try:
            _ = torch.randn(2, 1, 8, 8, 8).cuda()
        except RuntimeError as e:
            if "no kernel image" in str(e):
                print("\n" + "=" * 60)
                print("GPU - PyTorch incompatible. Fix:")
                print("1) Run in new cell: !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --force-reinstall")
                print("2) Runtime > Restart runtime")
                print("3) Run script again")
                print("=" * 60)
                raise SystemExit(1)
            raise

    print("ROOT:", ROOT)
    print("ROOT exists:", os.path.exists(ROOT))
    cv_base = os.path.join(ROOT, DATASET_CV_DIR)
    print("Dataset_CrossValidation exists:", os.path.exists(cv_base))
    print("OUT_ROOT (checkpoint):", OUT_ROOT)
    if EXPORT_TO_LOCAL:
        print("EXPORT_OUT_ROOT (vel/speed):", EXPORT_OUT_ROOT)
    print("MODALITIES:", MODALITIES)
    print("DEVICE:", DEVICE)
    print("BATCH_SIZE:", BATCH_SIZE, "| NUM_WORKERS:", NUM_WORKERS)

    cv_roots = get_cv_folds()
    if ONLY_CV:
        cv_roots = [p for p in cv_roots if os.path.basename(p) in ONLY_CV]
        print(f"ONLY_CV: {ONLY_CV} → {[os.path.basename(p) for p in cv_roots]}")
    else:
        cv_roots = [p for p in cv_roots if os.path.basename(p) not in SKIP_CV]
        print(f"Found CV folds: {[os.path.basename(p) for p in cv_roots]}")
    if SKIP_CV:
        print(f"Skipped CVs: {SKIP_CV}")
    if cv_roots:
        sample_path = os.path.join(cv_roots[0], "Train", "0", MODALITIES[0])
        print(f"Sample data path (CV1/Train/0/AD): {sample_path}")
        print(f"  Does this folder exist: {os.path.exists(sample_path)}")
        if os.path.exists(os.path.dirname(sample_path)):
            print(f"  Contents under Train/0: {os.listdir(os.path.dirname(sample_path))}")

    ensure_dir(OUT_ROOT)
    if CLEAN_INCOMPLETE:
        clean_incomplete_checkpoints()

    # Status summary (checkpoint=Drive, export=Drive or local)
    print("\n--- Current status ---")
    for cv_root in cv_roots:
        cv_name = os.path.basename(cv_root)
        for mod in MODALITIES:
            if is_modality_done(cv_name, mod):
                print(f"  Done: {cv_name}/{mod}")
            elif needs_export(cv_name, mod):
                print(f"  Training done, export pending: {cv_name}/{mod}")
            elif needs_training(cv_name, mod):
                print(f"  Training pending: {cv_name}/{mod}")
            else:
                print(f"  Unknown: {cv_name}/{mod}")
    print("---\n")

    # ========== PHASE 1: All trainings ==========
    print("\n" + "#" * 60)
    print("#  PHASE 1: ALL TRAININGS")
    print("#" * 60)
    for cv_root in cv_roots:
        cv_name = os.path.basename(cv_root)
        out_cv_root = os.path.join(OUT_ROOT, cv_name)
        ensure_dir(out_cv_root)
        set_deterministic(SEED)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        for modality in MODALITIES:
            out_mod_root = _ckpt_root(cv_name, modality)
            ensure_dir(out_mod_root)
            if (cv_name, modality) not in FORCE_RETRAIN and not needs_training(cv_name, modality):
                print(f"  Skip: {cv_name}/{modality} training already exists")
                continue
            print("\n" + "=" * 60)
            print(f"  [TRAINING] {cv_name} / {modality}")
            print("=" * 60)
            train_fm_linear_3scale(cv_root, out_mod_root, modality)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if RUN_TRAINING_ONLY:
        print("\nTrainings done (RUN_TRAINING_ONLY=True). Set RUN_TRAINING_ONLY=False and run again for export.")
        return

    # ========== PHASE 2: All exports ==========
    print("\n" + "#" * 60)
    print("#  PHASE 2: ALL EXPORTS")
    print("#" * 60)
    for cv_root in cv_roots:
        cv_name = os.path.basename(cv_root)
        for modality in MODALITIES:
            if not needs_export(cv_name, modality):
                print(f"  Skip: {cv_name}/{modality} export already exists")
                continue
            print("\n" + "=" * 60)
            print(f"  [EXPORT] {cv_name} / {modality}")
            print("=" * 60)
            best_ckpt = os.path.join(_ckpt_root(cv_name, modality), "checkpoints", "fm_best.pt")
            load_dev = "cpu" if USE_CPU_FOR_EXPORT else DEVICE
            model = load_fm_checkpoint(best_ckpt, load_dev)
            export_all_vel_and_speed(model, cv_root, cv_name, modality)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nAll CV x Modality completed. Checkpoint:", OUT_ROOT, "| Export:", EXPORT_OUT_ROOT)

    # Option A: Local export -> create zip and download
    if EXPORT_TO_LOCAL and os.path.exists(EXPORT_OUT_ROOT):
        try:
            from google.colab import files
            zip_path = "/content/flow_matching_export.zip"
            print(f"\nCreating zip: {zip_path}")
            shutil.make_archive(zip_path.replace(".zip", ""), "zip", "/content", OUT_ROOT_NAME)
            print("Download starting...")
            files.download(zip_path)
            print("Download complete. Save to your machine.")
        except ImportError:
            print("Zip download unavailable (not in notebook). Manual path:", EXPORT_OUT_ROOT)


if __name__ == "__main__":
    main()
