# Flow Matching–Derived Velocity and Speed Representations for Post-Stroke Motor Outcome Prediction

This repository contains the official implementation of the study:
**"Flow Matching–Derived Velocity and Speed Representations for Post-Stroke Motor Outcome Prediction"**.

## Overview

This study introduces a flow matching (FM)–enhanced learning framework that incorporates velocity and speed maps derived from learned transport dynamics into a 3D CNN-based classifier. Rather than generating synthetic images, the approach extracts flow-informed representations from DTI metrics (FA, MD, RD, AD) and structural WM/GM maps, using them jointly with original volumetric maps to enrich representation learning without altering anatomical structure. Integrating FM-derived velocity and speed improves classification performance and cross-validation stability, with notable gains for FA, RD, WM, and GM.

---

## Study Overview

This study consists of two main components:

1. **Flow Matching** – A generative model that learns velocity and speed fields along the path from noise to image
2. **3D CNN** – A classifier that uses raw modalities and/or flow-matching-derived velocity/speed maps

---

## File Structure

| File | Description |
|------|-------------|
| `fm_train_and_export_linear_3scale_colab.py` | Flow Matching training + velocity/speed map export |
| `cnn3d_fa_speed_vel_colab.py` | 3D CNN classification (FA, speed, vel combinations) |

---

## Requirements

- Python 3.x with GPU support (CUDA recommended)
- Python packages: `torch`, `nibabel`, `numpy`, `matplotlib`

```bash
pip install torch nibabel numpy matplotlib
```

---

## Data Structure

Under your data root (e.g. `ROOT`), the following structure is expected:

```
ROOT/
└── Dataset_CrossValidation/
    ├── CV1/
    │   ├── Train/
    │   │   ├── 0/          # Class 0 (e.g. healthy)
    │   │   │   ├── AD/     # .nii or .nii.gz files
    │   │   │   ├── MD/
    │   │   │   ├── RD/
    │   │   │   ├── FA/
    │   │   │   ├── WM/
    │   │   │   ├── GM/
    │   │   │   └── MRI/
    │   │   └── 1/          # Class 1 (e.g. motor impairment)
    │   │       └── ...
    │   └── Test/
    │       ├── 0/
    │       └── 1/
    ├── CV2/
    ├── CV3/
    └── ...
```

---

## Usage

### Step 1: Flow Matching (Stage 1)

1. Run `fm_train_and_export_linear_3scale_colab.py`
2. Adjust **CONFIG** parameters as needed:
   - `ROOT` – Data root directory (default: `/content/drive/MyDrive/MICCAI`)
   - `ONLY_CV` – Which CV folds to run (e.g. `["CV1"]`)
   - `ONLY_MODALITIES` – Which modalities (e.g. `["AD"]`, `["FA","RD"]`, or empty = all)
   - `RUN_TRAINING_ONLY` – `True` = training only, `False` = training + export
3. Run the script

**Output:** `flow_matching_velocity_speed_t0.80/CV1/AD/vel_t0.80/`, `speed_t0.80/`, etc.

### Step 2: 3D CNN Classification (Stage 2)

1. Run `cnn3d_fa_speed_vel_colab.py` after flow-matching export
2. Adjust **CONFIG** parameters:
   - `ROOT` – Data root directory
   - `FM_EXPORT_ROOT` – Flow matching output directory (typically `ROOT + "flow_matching_velocity_speed_t0.80"`)
   - `EXPERIMENT` – Experiment type: `fa`, `speed`, `vel`, `fa_speed`, `fa_vel`, `fa_velspeed`, `fa_speed_vel`, or `rd_vel`, `ad_vel`, etc.
   - `MODALITY` – Modality: `fa`, `md`, `rd`, `ad`, `wm`, `gm`, `mri`
   - `CV_LIST` – Which CV folds to run
   - `NUM_RUNS` – Number of runs per fold (e.g. 10)
3. Run the script

---

## Configuration Parameters

### Flow Matching

| Parameter | Description | Default |
|-----------|-------------|---------|
| `TVAL` | Time point for velocity/speed (0–1) | 0.80 |
| `EPOCHS` | Number of training epochs | 300 |
| `EXPORT_TO_LOCAL` | Write output to local disk instead of remote storage | True |

### 3D CNN

| Parameter | Description | Default |
|-----------|-------------|---------|
| `EXPERIMENT` | Experiment type (fa, speed, vel, fa_speed, fa_vel, etc.) | rd_vel |
| `EPOCHS` | Number of training epochs | 80 |
| `NUM_RUNS` | Number of runs per fold | 10 |

---

## Experiment Types (3D CNN)

- `fa` – FA only
- `speed` – Speed map only
- `vel` – Velocity map only
- `fa_speed` – FA + speed (2 parallel branches)
- `fa_vel` – FA + velocity (2 parallel branches)
- `fa_velspeed` – FA + speed + vel (2-channel concat)
- `fa_speed_vel` – FA + speed + vel (3 parallel branches)
- `rd_vel`, `ad_vel`, etc. – Raw modality + velocity

---

## Local Setup

For local or custom environments:

1. Set `ROOT` to your data directory
2. Set `FM_EXPORT_ROOT` to the flow-matching output directory
3. A CUDA-capable GPU is recommended for training

---

## Notes

- Training can be time-consuming; GPU acceleration is recommended
- With `EXPORT_TO_LOCAL=True`, output is written to local disk and can be zipped for download
- Cross-validation folds (`CV1`, `CV2`, ...) should match your dataset structure

---

## Citation

If you use this repository, please cite our MICCAI 5340 paper:

> Flow Matching–Derived Velocity and Speed Representations for Post-Stroke Motor Outcome Prediction, MICCAI 2026

---

## License

This project is licensed under the MIT License.
