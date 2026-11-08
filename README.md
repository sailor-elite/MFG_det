# MFG_det — automated quality control of milk fat globule micrographs

Automated detection of defective bright-field micrographs of milk fat globules (MFG).
The project compares three method families on the same dataset:

- classical focus and contrast measures (variance of Laplacian, Tenengrad, Brenner, colour statistics),
- unsupervised anomaly detection trained on good images only (PatchCore, PaDiM via Anomalib),
- supervised CNNs (ResNet, Xception via timm) with Grad-CAM explanations.

Defect classes include defocus (light/heavy), colour filters, air bubbles, coverslip corners,
over-staining, and undiluted or over-diluted preparations. Frozen and UHT milk images are held out
to measure false alarms on good images of a different milk type.

> **Status:** work in progress (pilot / feasibility study). Code, results and the dataset will be
> published together with the accompanying article.

## Requirements

- Linux, Python 3.11
- [uv](https://docs.astral.sh/uv/) for environment management
- NVIDIA GPU with a CUDA 12.6-compatible driver (PyTorch wheels come from the `cu126` index
  configured in `pyproject.toml`); developed on a 4 GB GTX 1650

## Setup

```bash
git clone https://github.com/sailor-elite/MFG_det.git
cd MFG_det
uv sync                                                  # creates .venv with all dependencies
uv run python -c "import torch; print(torch.cuda.is_available())"
uv run nbstripout --install --attributes .gitattributes  # strip notebook outputs on commit
```

Notebooks are run in VS Code with the project's `.venv` as the Jupyter kernel.
Long training runs are started from `scripts/` with `uv run python scripts/<script>.py`.

## Repository layout

```
configs/    YAML configs per experiment
data/       manifest.csv (tracked); crop and feature caches (git-ignored)
notebooks/  exploration, figures and results review (outputs stripped on commit)
results/    metric tables and per-fold predictions; checkpoints and caches git-ignored
scripts/    CLI entry points for long runs (training, full evaluation)
src/        reusable code imported by notebooks and scripts
```

## Data

The raw micrographs (about 4,000 PNG files, ~38 GB, 2048 × 1536 px) are **not** part of this
repository. Images, crops and model checkpoints are excluded via `.gitignore`.
The dataset will be released publicly together with the article.
