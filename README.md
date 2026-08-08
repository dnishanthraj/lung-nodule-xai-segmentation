# Lung Nodule XAI Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#prerequisites)

An end-to-end pipeline for detecting and segmenting pulmonary nodules — including
subsolid nodules — in CT scans, built around U-Net++, with explainability and
false-positive reduction layered on top so the output is usable by a clinician,
not just a benchmark score.

Originally developed as a University of Warwick CS310 third-year dissertation,
*"Enhancing Lung Nodule Segmentation with Explainability and False Positive
Reduction"* (First Class, 83%).

## What it does

- **Segmentation**: U-Net and U-Net++ models trained on the LIDC-IDRI dataset.
  Hyperparameter tuning, loss function changes, and augmentation strategy
  produced an **8.5% relative improvement in Dice score** and a **9.3%
  improvement in IoU** over the baseline pipeline.
- **False positive reduction**: an XGBoost classifier trained on morphological
  features of predicted masks filters out spurious detections, cutting false
  positives per scan by **~24.5%** while preserving recall.
- **Explainability**: Grad-CAM heatmaps expose which regions of a scan drove
  each prediction, so a radiologist can sanity-check the model instead of
  trusting it blindly.
- **Dashboard**: a Streamlit app for reviewing segmentation outputs, comparing
  raw vs. FPR-filtered results, and inspecting Grad-CAM overlays per slice.

## Repository structure

```
lung-nodule-xai-segmentation/
├── preprocessing/       # LIDC-IDRI DICOM -> .npy image/mask extraction (pylidc-based)
│   ├── config_file_create.py
│   ├── prepare_dataset.py
│   ├── utils.py
│   ├── csv/              # Precomputed metadata + train/val/test splits
│   └── requirements.txt
├── segmentation/        # Model training, validation, Grad-CAM, FPR classifier
│   ├── unet/              # U-Net
│   ├── unet_nested/        # U-Net++
│   ├── train.py / train.sbatch
│   ├── validate.py / validate.sbatch
│   ├── dataset.py, losses.py, metrics.py, grad_cam.py, utils.py
│   └── requirements.txt
├── dashboard/            # Streamlit app for reviewing model outputs
│   ├── 01_Welcome.py
│   ├── pages/
│   ├── components/
│   └── requirements.txt
└── LICENSE
```

## Prerequisites

- **Dataset**: [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/) (CT scans)
- **Python**: 3.10+
- **PyTorch**: 2.5 with CUDA support (GPU strongly recommended for training)
- **Shell utilities**: `git`; `sbatch`/`ssh` only if you're running on a SLURM cluster

Each of the three stages below has its own `requirements.txt`, since they're
typically run in different environments (a preprocessing/training environment
on a GPU cluster, a lightweight local environment for the dashboard).

## 1. Preprocessing

Extracts nodule-containing slices from the raw DICOM dataset into `.npy`
images/masks, using [`pylidc`](https://pylidc.github.io/), building on the
preprocessing approach from
[Jaeho Kim's LIDC-IDRI-Preprocessing pipeline](https://github.com/jaeho3690/LIDC-IDRI-Preprocessing).

```bash
cd preprocessing
pip install -r requirements.txt

# Generates lung.conf — edit it to point at your downloaded LIDC-IDRI/ folder
python config_file_create.py

# Extracts .npy images/masks and writes metadata CSVs under data/ and csv/
python prepare_dataset.py
```

`csv/meta.csv` and `csv/clean_meta.csv` (train/val/test splits, kept
consistent per-nodule) are already committed in this repo so you can go
straight to training/validation against the same splits used in the
dissertation without re-running preprocessing yourself.

## 2. Segmentation (training & validation)

```bash
cd segmentation
pip install -r requirements.txt

python train.py --name NestedUNET --augmentation True
python validate.py --name NestedUNET --augmentation True --folder <run-folder-from-train>
```

Both scripts accept further arguments (`--epochs`, `--batch-size`, `--lr`,
etc.) — see `parse_args()` in each file.

On a SLURM cluster:

```bash
sbatch train.sbatch
sbatch validate.sbatch
```

`validate.py` also trains the XGBoost false-positive-reduction classifier
(on first run, if no saved classifier is found) and generates Grad-CAM
heatmaps for every prediction.

## 3. Dashboard

```bash
cd dashboard
pip install -r requirements.txt
streamlit run 01_Welcome.py
```

Reads model outputs from `segmentation/model_outputs/`, so run validation at
least once first. The contact form on the Support page is optional and reads
`SUPPORT_EMAIL_ADDRESS` / `SUPPORT_EMAIL_PASSWORD` from the environment if
you want to enable it.

## Acknowledgements

- Preprocessing approach adapted from [Jaeho Kim's LIDC-IDRI-Preprocessing pipeline](https://github.com/jaeho3690/LIDC-IDRI-Preprocessing).
- Reference implementation: [Mike Huang's LungNoduleDetectionClassification](https://github.com/mikejhuang/LungNoduleDetectionClassification).
- Supervised by Professor Nathan Griffiths, Department of Computer Science, University of Warwick.

## License

MIT — see [LICENSE](LICENSE).
