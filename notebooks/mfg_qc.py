# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: mfg-qc (3.11.18)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Config
#

# %%
import os

# Pretrained network weights are downloaded into the project (git-ignored), not into ~/.cache
os.environ["HF_HOME"] = "/home/qq/mfg_qc/data/weights/huggingface"
os.environ["TORCH_HOME"] = "/home/qq/mfg_qc/data/weights/torch"

import hashlib
import pickle
import re
import shutil
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from anomalib.models.components.sampling import KCenterGreedy
from anomalib.models.image.dfm.torch_model import DFMModel
from anomalib.models.image.padim.torch_model import PadimModel
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from lightgbm import LGBMClassifier
from matplotlib.patches import Rectangle
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from skimage.feature import local_binary_pattern
from skimage.filters import threshold_otsu
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# %%
RAW_DIR = Path("/home/qq/Pictures/MFG_det")  # raw micrographs, READ ONLY
# 3.2 % fresh milk, taken 11 Nov 2026: sub-folders M01-M03 (undiluted), M04_diluted, M05_diluted;
# JPG files named only by number (e.g. 0244.jpg), the preparation comes from the folder name
EXTRA_DIR = Path("/home/qq/mfg_qc/data/MFG/3-2_fresh_milk")
MFG_DIR = Path("/home/qq/mfg_qc/data/MFG")    # copies of the raw images, one folder per label
EXCLUDED_CSV = Path("/home/qq/mfg_qc/data/excluded_images.csv")  # images removed for bad quality
MANIFEST_CSV = Path("/home/qq/mfg_qc/data/manifest.csv")  # one row per raw image (labels, preps, folds)
RESULTS_DIR = Path("/home/qq/mfg_qc/results")  # figures and tables (also for the paper)
OVERWRITE = False  # False = never overwrite files that already exist in RESULTS_DIR
USE_SAVED_RESULTS = True  # True: slow cells load their saved results (compute only what is not saved yet);
                          # False: recompute all slow cells and replace their saved results
                          # (set False after changing the data or settings, e.g. new exclusions)
CACHE_DIR = Path("/home/qq/mfg_qc/data/cache")  # saved results of slow cells (git-ignored)
SEED = 0           # random seed for everything

# ROI (CLAUDE.md §9 Step 1) and tiles (Step 2)
SHARP_TILE = 64            # px, cell size of the sharpness map (2048 x 1536 frame -> 32 x 24 cells)
ROI_THRESHOLD = 0.8        # sharp area = at least 80 % of the sharpness in the frame centre
CROP_SIZE = 384            # px, side of one tile
ROI = (256, 192, 1792, 1344)            # x0, y0, x1, y1 in px: 4 x 3 tiles of 384 px, centred
ROI_FIRST_GUESS = (250, 200, 1750, 1400)  # x0, y0, x1, y1: first box chosen by eye (for comparison)
WHOLE_SIZE = (768, 576)    # (width, height) in px of the whole-picture input: the ROI scaled to half
EXAMPLE_IMAGE = "A01_0141.png"  # corner image used to show what the models get

# Augmentation (training only): exact flips and rotations by multiples of 90 deg, no interpolation.
# One version = (number of 90 deg counter-clockwise rotations, horizontal flip after rotating).
AUG_TILE = [(k, flip) for flip in (False, True) for k in range(4)]  # all 8 versions of a square tile
AUG_WHOLE = [(0, False), (0, True), (2, True), (2, False)]  # original, h-flip, v-flip, 180 deg:
                                                             # the 4 versions that keep 768 x 576
AUG_EXAMPLE_TILE = 10  # tile of EXAMPLE_IMAGE shown in the augmentation figure (edge of the coverslip)

# Speed test of loading the model inputs
SPEED_TEST_IMAGES = 40  # random images (not excluded) that are timed
DATA_WORKERS = 4        # parallel loading processes assumed for training (CPU: 4 cores, 8 threads)

# Classical features (definitions in "# Classical methods")
GLOBULE_BLUR_SIGMA = 15      # px, Gaussian blur that estimates the local background around globules
GLOBULE_BLUR_SIGMA_WHOLE = GLOBULE_BLUR_SIGMA * WHOLE_SIZE[0] / (ROI[2] - ROI[0])  # 7.5 px: same in um
FEATURE_EXAMPLE_TILE = 5     # tile shown for every label in the feature figures (centre left)
SERIES_COLOUR = "#2a78d6"    # colour of the plotted label in comparison figures
REFERENCE_COLOUR = "#52514e" # colour of the "good" reference in comparison figures
FEATURES_DIR = Path("/home/qq/mfg_qc/data/features")      # feature tables (git-ignored)
TILE_FEATURES_CSV = FEATURES_DIR / "tile_features.csv"    # 95 features of every tile (12 per image)
WHOLE_FEATURES_CSV = FEATURES_DIR / "whole_features.csv"  # 95 features of every 768 x 576 whole picture
HIST_TYPES = [  # histogram features: (column prefix, title, x-axis label, bin edges)
    ("hist_bright", "brightness", "grey level", np.linspace(0, 256, 17)),
    ("hist_hue", "hue (weighted by saturation)", "hue [deg]", np.linspace(0, 360, 19)),
    ("hist_sat", "saturation", "saturation", np.linspace(0, 256, 17)),
    ("hist_grad", "gradient magnitude", "log(1 + |grad I|)", np.linspace(0, 7.3, 17)),
    ("hist_lbp", "LBP texture codes", "uniform LBP code", np.arange(11) - 0.5),
]

# Image number -> label for A preparations (CLAUDE.md §7.2): (last image number, label)
LABELS_A = [
    (40, "good"),
    (60, "blur_light"),
    (80, "blur_heavy"),
    (100, "filter_yellow"),
    (120, "filter_blue"),
    (140, "bubbles"),
    (160, "corner"),
]
# B, C, D, F, U preparations: one label for all images (1-80).
# F and U (frozen and UHT milk) look clearly different from fresh milk: they count as defective
# ("to be flagged") and are TEST ONLY, never used for training.
LABELS_BY_TYPE = {
    "B": "overstained",
    "C": "undiluted",
    "D": "overdiluted",
    "F": "frozen",
    "U": "uht",
}
# 3.2 % fresh milk (M): the label depends on the preparation. Like F and U it is another milk:
# counted as defective ("to be flagged") and TEST ONLY.
LABELS_BY_PREP = {
    "M01": "milk32_undiluted", "M02": "milk32_undiluted", "M03": "milk32_undiluted",  # 0 ml water
    "M04": "milk32", "M05": "milk32",                                                 # 3.5 ml water
}

# Label -> label group (CLAUDE.md §7.2); frozen, uht and 3.2 % milk: other milk type (counted as defective)
LABEL_GROUPS = {
    "good": "good",
    "blur_light": "acquisition",
    "blur_heavy": "acquisition",
    "filter_yellow": "acquisition",
    "filter_blue": "acquisition",
    "bubbles": "preparation",
    "overstained": "preparation",
    "undiluted": "preparation",
    "overdiluted": "preparation",
    "corner": "field_position",
    "frozen": "other_milk",
    "uht": "other_milk",
    "milk32_undiluted": "other_milk",
    "milk32": "other_milk",
}

# Expected number of raw images per label (CLAUDE.md §7.3; 3.2 % milk: the number of files taken)
EXPECTED_COUNTS = {
    "good": 640, "blur_light": 320, "blur_heavy": 320, "filter_yellow": 320, "filter_blue": 320,
    "bubbles": 320, "corner": 320, "overstained": 320, "undiluted": 320, "overdiluted": 320,
    "frozen": 240, "uht": 240, "milk32_undiluted": 366, "milk32": 239,
}

# Preparation log (CLAUDE.md §7.4, same as Dziennik_pomiary.xlsx):
# prep -> (distilled water [ml], milk type, day, preparation time, start of imaging)
# U01-U03: the log says "any day"; the file times show 2026-11-01.
PREP_LOG = {
    "A01": (1.5, "fresh", "2026-10-31", "22:10", "22:32"),
    "A02": (2.5, "fresh", "2026-11-01", "08:48", "09:08"),
    "A03": (3.5, "fresh", "2026-10-31", "19:42", "19:47"),
    "A04": (1.5, "fresh", "2026-10-31", "21:10", "21:34"),
    "A05": (2.5, "fresh", "2026-11-01", "09:09", "09:16"),
    "A06": (3.5, "fresh", "2026-10-31", "19:49", "20:06"),
    "A07": (1.5, "fresh", "2026-10-31", "21:35", "21:50"),
    "A08": (2.5, "fresh", "2026-11-01", "09:17", "09:24"),
    "A09": (3.5, "fresh", "2026-10-31", "20:07", "20:20"),
    "A10": (1.5, "fresh", "2026-10-31", "21:50", "22:03"),
    "A11": (2.5, "fresh", "2026-11-01", "09:40", "09:46"),
    "A12": (3.5, "fresh", "2026-10-31", "20:22", "20:31"),
    "A13": (1.5, "fresh", "2026-11-01", "08:08", "08:22"),
    "A14": (2.5, "fresh", "2026-11-01", "09:48", "10:07"),
    "A15": (3.5, "fresh", "2026-10-31", "20:34", "20:45"),
    "A16": (1.5, "fresh", "2026-11-01", "08:23", "08:34"),
    "B01": (1.5, "fresh", "2026-11-01", "12:49", "12:52"),
    "B02": (1.5, "fresh", "2026-11-01", "12:53", "12:58"),
    "B03": (1.5, "fresh", "2026-11-01", "12:58", "13:01"),
    "B04": (1.5, "fresh", "2026-11-01", "13:01", "13:04"),
    "C01": (0.0, "fresh", "2026-11-01", "10:29", "10:37"),
    "C02": (0.0, "fresh", "2026-11-01", "10:39", "10:46"),
    "C03": (0.0, "fresh", "2026-11-01", "10:50", "10:56"),
    "C04": (0.0, "fresh", "2026-11-01", "10:57", "11:05"),
    "D01": (20.0, "fresh", "2026-11-01", "11:50", "11:59"),
    "D02": (20.0, "fresh", "2026-11-01", "12:00", "12:14"),
    "D03": (20.0, "fresh", "2026-11-01", "12:30", "12:35"),
    "D04": (20.0, "fresh", "2026-11-01", "12:36", "12:37"),
    "F01": (3.5, "frozen", "2026-11-01", "13:08", "13:13"),
    "F02": (3.5, "frozen", "2026-11-01", "13:15", "13:17"),
    "F03": (3.5, "frozen", "2026-11-01", "13:19", "13:22"),
    "U01": (3.5, "UHT", "2026-11-01", "14:22", "14:31"),
    "U02": (3.5, "UHT", "2026-11-01", "14:32", "14:36"),
    "U03": (3.5, "UHT", "2026-11-01", "14:37", "14:39"),
    # 3.2 % fresh milk: no log; times of the first and the last file
    "M01": (0.0, "fresh 3.2%", "2026-11-11", "12:37", "12:42"),
    "M02": (0.0, "fresh 3.2%", "2026-11-11", "12:57", "13:03"),
    "M03": (0.0, "fresh 3.2%", "2026-11-11", "13:07", "13:10"),
    "M04": (3.5, "fresh 3.2%", "2026-11-11", "13:15", "13:18"),
    "M05": (3.5, "fresh 3.2%", "2026-11-11", "13:20", "13:24"),
}

# Fixed folds (CLAUDE.md §9 Step 6). F, U and M are not in any fold: they are test only.
FOLDS = {
    1: ["A01", "A02", "A03", "A13", "B01", "C01", "D01"],
    2: ["A04", "A05", "A06", "A16", "B02", "C02", "D02"],
    3: ["A07", "A08", "A09", "A14", "B03", "C03", "D03"],
    4: ["A10", "A11", "A12", "A15", "B04", "C04", "D04"],
}
# Evaluation runs: test on fold k, choose thresholds and settings on the next fold (cyclic),
# train on the other two. F, U and M (no fold) are tested in every run.
VALIDATION_FOLD = {k: k % len(FOLDS) + 1 for k in FOLDS}  # {1: 2, 2: 3, 3: 4, 4: 1}
TEST_ONLY_LABELS = ["frozen", "uht", "milk32_undiluted", "milk32"]  # other milk types, never trained on
FLAG_PERCENTILE = 95  # flag an image when it is more unusual than 95 % of good validation images
RUN_COLOURS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}  # one colour per run in figures

# One-class models on all 95 features (CLAUDE.md §9 Step 3c): trained on good images only,
# default settings (no tuning). Each entry makes a new, untrained detector.
ONE_CLASS_DETECTORS = {
    "Mahalanobis": lambda: LedoitWolf(),  # distance from the average good image (stabilised covariance)
    "One-Class SVM": lambda: OneClassSVM(kernel="rbf", gamma="scale", nu=0.05),  # boundary around good
    "Isolation Forest": lambda: IsolationForest(n_estimators=200, random_state=SEED),  # how easy to isolate
    "LOF": lambda: LocalOutlierFactor(n_neighbors=20, novelty=True),  # density vs the 20 nearest good
}

# Supervised models on all 95 features (CLAUDE.md §9 Step 3b): learn from good AND defective training
# images; class weights balance good vs defective; default settings (no tuning).
SUPERVISED_MODELS = {
    "Logistic regression": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
    "Random forest": lambda: RandomForestClassifier(
        n_estimators=300, class_weight="balanced", n_jobs=DATA_WORKERS, random_state=SEED),
    "HistGradientBoosting": lambda: HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
    "LightGBM": lambda: LGBMClassifier(class_weight="balanced", n_jobs=DATA_WORKERS, random_state=SEED, verbose=-1),
}

# Deep features: pretrained networks (no training) turn every tile and whole picture into numbers
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEEP_BACKBONES = {  # short name -> timm model name
    "resnet18": "resnet18.a1_in1k",               # ImageNet CNN; last two stages, averaged: 256 + 512 values
    # about 2 h on the GTX 1650 the first time (resumable in chunks of 200 images):
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",  # self-supervised ViT; summary + mean patch token: 384 + 384
}
DEEP_FEATURES_DIR = Path("/home/qq/mfg_qc/data/features/deep")  # saved chunks per backbone (git-ignored)
DEEP_BATCH_IMAGES = 4  # images per GPU batch (each = 12 tiles + 1 whole picture)

# Anomaly detection (CLAUDE.md §9 Step 4): models that learn only from good tiles.
# PatchCore (anomalib) on the 12 native 384-px tiles; about 45 min for all 4 runs on the GTX 1650
PATCHCORE_BACKBONE = "resnet18.a1_in1k"  # fastest choice; the paper's wide_resnet50_2 would take ~2.3 h
PATCHCORE_LAYERS = ["layer2", "layer3"]  # mid-level feature maps: 48 x 48 patches per tile, 128 + 256 numbers each
PATCHCORE_KEEP_SHARE = 0.025  # random share of the patches of every good training image kept (~220,000 per run)
PATCHCORE_BANK_SIZE = 10_000  # memory bank: patches that anomalib's coreset selection picks from the kept ones
PATCHCORE_NEIGHBORS = 9       # anomalib default: the tile score is reweighted using 9 nearest bank patches
# DFM (anomalib defaults, same network as PatchCore): every tile = its layer3 feature map averaged down to
# 6 x 6 cells (256 x 36 = 9,216 numbers); a PCA learns the main patterns of the good tiles
DFM_BACKBONE = PATCHCORE_BACKBONE
DFM_LAYER = "layer3"
DFM_POOLING = 4       # average 4 x 4 cells of the 24 x 24 feature map -> 6 x 6
DFM_VARIANCE = 0.97   # PCA keeps the patterns that explain 97 % of the variation of the good tiles
# PaDiM (paper defaults for ResNet-18, same network as PatchCore): one Gaussian per patch position
# (96 x 96 positions per tile), from a random subset of the feature-map channels
PADIM_BACKBONE = PATCHCORE_BACKBONE
PADIM_LAYERS = ["layer1", "layer2", "layer3"]
PADIM_FEATURES = 100  # random subset of the 64 + 128 + 256 = 448 channels (the same subset in every run)
PADIM_BATCH_IMAGES = 1  # images per GPU batch: anomalib first builds all 448 channels (~0.2 GB per image)

# Combining methods (saved scores, no training); names as in the "method" column of the summary tables
MAX_RULE_PAIR = [  # max rule: the higher unusualness of the two, then the usual 95th-percentile cut
    "dinov2_vits14 Logistic regression | whole",  # strong on everything except frozen milk
    "DFM resnet18 | tiles mean",                  # catches frozen milk, misses sparse fields
]
VOTERS = [  # voting: the best method of each family, each decides on its own (unusualness > 95)
    "Mahalanobis | whole",                        # handcrafted one-class
    "HistGradientBoosting | tiles max",           # handcrafted supervised
    "dinov2_vits14 LOF | whole",                  # deep one-class
    "dinov2_vits14 Logistic regression | whole",  # deep supervised
    "DFM resnet18 | tiles mean",                  # anomaly detection
]
VOTES_NEEDED = [1, 2, 3, 4, 5]  # flag an image if at least this many voters flag it (3 = majority)

# Benchmark: the best method of each of the 6 families by balanced accuracy; the n best of them vote, and
# an image is flagged if at least k of them flag it
BENCHMARK_VOTES = [(6, 1), (3, 2)]  # (n voters, k votes needed): any of the 6; 2 of the 3 best

# CNNs (CLAUDE.md §9 Step 5): ImageNet networks fine-tuned with two outputs, good vs defective (T1) and
# which label (T2). Design A: the whole picture (WHOLE_SIZE). Trained in fp32: fp16 overflows in these
# networks (NaN) and was 4x slower on the GTX 1650 (speed test: 22.5 training images/s, 1.75 GB GPU).
CNN_BACKBONES = {"resnet18": "resnet18.a1_in1k",  # short name -> timm name (same weights as above)
                 "resnet50": "resnet50.a1_in1k"}
CNN_GRAD_CHECKPOINTING = ["resnet50"]  # recompute activations in the backward pass instead of storing them:
                     # ResNet-50 needs > 4 GB GPU at batch 8 without it, 2.8 GB with it (6.0 images/s)
CNN_EPOCHS = 20      # at most this many passes over the training images (the learning rate is planned for 20)
CNN_PATIENCE = 5     # early stopping: stop when the validation loss has not improved for 5 epochs;
                     # the epoch with the lowest validation loss is kept
CNN_BATCH = 8        # images per GPU batch
CNN_ACCUMULATE = 8   # batches per weight update: effective batch of 64 images
CNN_LR = 3e-4        # AdamW learning rate: rises during the first epoch, then falls to 0 (cosine)
CNN_WEIGHT_DECAY = 0.01
CNN_DIR = Path("/home/qq/mfg_qc/data/cnn")  # checkpoints (git-ignored); training resumes after every epoch
CNN_TILE_BACKBONES = ["resnet18"]  # design B (12 native tiles, "any tile defective"): networks from CNN_BACKBONES
CNN_SCORE_PICTURES = 4  # design B: pictures per GPU batch when all their tiles are scored (4 x 12 = 48 tiles)
FLAG_COLOUR = "#e34948"  # frame of a flagged tile in the design-B figures

data = {}

# %% [markdown]
# # Helper functions

# %%
def image_name(path: Path) -> str:
    """Return the unique name of an image: "<PREP>_<NNNN>.<ext>".

    The raw files are already named like that (e.g. "A01_0042.png"). The 3.2 % milk files are named
    only by number (e.g. "M04_diluted/0244.jpg"); for them the preparation comes from the folder name.

    Args:
        path (Path): image file, e.g. Path(".../A/A01/A01_0042.png") or Path(".../M04_diluted/0244.jpg").

    Returns:
        str: e.g. "A01_0042.png" or "M04_0244.jpg".
    """
    if re.match(r"^[A-Z]\d{2}_", path.name):
        return path.name
    return f"{path.parent.name[:3]}_{path.name}"  # "M04_diluted" -> "M04"


def label_of(path: Path) -> str:
    """Return the label of one raw image, read from its (unique) name (CLAUDE.md §7.2).

    The folder is ignored on purpose: some files are stored in the wrong folder
    (e.g. A11 and A14 are inside A08/). Only the 3.2 % milk files take their preparation from
    the folder, because their file names have none (see image_name).

    Args:
        path (Path): raw image file, e.g. Path("/home/qq/Pictures/MFG_det/A/A01/A01_0042.png").

    Returns:
        str: label of the image, e.g. "blur_light"
            (one of the values in LABELS_A, LABELS_BY_TYPE or LABELS_BY_PREP).

    Raises:
        ValueError: if the preparation type or image number is not in the mapping.
    """
    prep_id, image_idx = Path(image_name(path)).stem.split("_")  # "A01_0042" -> "A01", "0042"
    prep_type = prep_id[0]                                         # "A01" -> "A"
    number = int(image_idx)                                        # "0042" -> 42

    if prep_id in LABELS_BY_PREP:
        return LABELS_BY_PREP[prep_id]
    if prep_type == "A":
        for last, label in LABELS_A:
            if number <= last:
                return label
    elif prep_type in LABELS_BY_TYPE:
        return LABELS_BY_TYPE[prep_type]

    raise ValueError(f"No label defined for {image_name(path)}")


def data_key(names: list[str]) -> str:
    """Return a short fingerprint of the set of images in use, for the file names of saved results.

    Saved model results depend on which images are used. With the fingerprint in their file name,
    they are recomputed automatically when images are added or excluded.

    Args:
        names (list[str]): unique image names, e.g. ["A01_0001.png", ...] (order does not matter).

    Returns:
        str: number of images + 8 hex characters, e.g. "4571-1a2b3c4d".
    """
    digest = hashlib.md5("\n".join(sorted(names)).encode()).hexdigest()[:8]
    return f"{len(names)}-{digest}"


def build_manifest(paths: list[Path], excluded_names: set[str]) -> pd.DataFrame:
    """Build the manifest: one row per raw image with its label, preparation details and fold.

    Uses label_of() and the Config globals LABEL_GROUPS, PREP_LOG and FOLDS.

    Args:
        paths (list[Path]): all raw image files, e.g. data["RAW_PICTURES"] (4605 paths).
        excluded_names (set[str]): unique image names removed for bad quality,
            e.g. {"A02_0009.png", "A08_0001.png", ...}.

    Returns:
        pd.DataFrame: one row per image, sorted by file name, with columns
            path (str): raw image path,
            file (str): unique image name (image_name), e.g. "A01_0042.png" or "M04_0244.jpg",
            prep_id (str): e.g. "A01"; prep_type (str): e.g. "A",
            image_idx (int): image number, e.g. 42,
            label (str): e.g. "blur_light"; label_group (str): e.g. "acquisition",
            is_defect (bool): True for every label except "good" (other milk types included),
            dilution_ml (float): distilled water added, e.g. 1.5,
            day (str): e.g. "2026-10-31"; milk_type (str): "fresh", "frozen", "UHT" or "fresh 3.2%",
            prep_time, photo_time (str): from the log, e.g. "2026-10-31 22:10",
            file_time (str): modification time of the raw file, e.g. "2026-10-31 22:35:12",
            fold (Int64): 1-4, <NA> for test-only preparations,
            test_only (bool): True for F, U and M (never used for training),
            excluded (bool): True if removed for bad quality.

    Raises:
        KeyError: if a preparation is missing from PREP_LOG.
    """
    fold_of = {prep: fold for fold, preps in FOLDS.items() for prep in preps}  # "A01" -> 1
    rows = []
    for path in paths:
        name = image_name(path)                                 # e.g. "A01_0042.png", "M04_0244.jpg"
        prep_id, image_idx = Path(name).stem.split("_")         # "A01_0042" -> "A01", "0042"
        water_ml, milk_type, day, prep_time, photo_time = PREP_LOG[prep_id]
        label = label_of(path)
        rows.append({
            "path": str(path),
            "file": name,
            "prep_id": prep_id,
            "prep_type": prep_id[0],
            "image_idx": int(image_idx),
            "label": label,
            "label_group": LABEL_GROUPS[label],
            "is_defect": label != "good",
            "dilution_ml": water_ml,
            "day": day,
            "milk_type": milk_type,
            "prep_time": f"{day} {prep_time}",
            "photo_time": f"{day} {photo_time}",
            "file_time": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "fold": fold_of.get(prep_id),  # None for F, U and M
            "test_only": prep_id not in fold_of,
            "excluded": name in excluded_names,
        })
    manifest = pd.DataFrame(rows).sort_values("file", ignore_index=True)
    manifest["fold"] = manifest["fold"].astype("Int64")
    return manifest


def copy_pictures_by_label(
    pictures_by_label: dict[str, list[Path]],
    dst_dir: Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Copy raw images into one folder per label, under their unique names (image_name).

    Result: dst_dir/<label>/<unique name>, e.g. data/MFG/blur_light/A01_0042.png or
    data/MFG/milk32/M04_0244.jpg.
    The raw files are only read. Each copy keeps the original modification time
    (needed later for the timestamp check) but is writable (the raw files are read-only).
    A file is first copied as "<name>.part" and renamed when complete, so an
    interrupted run never leaves a half-copied .png behind.

    Args:
        pictures_by_label (dict[str, list[Path]]): label -> raw image paths,
            e.g. data["PICTURES_BY_LABEL"].
        dst_dir (Path): destination root folder, e.g. MFG_DIR.
        overwrite (bool): False (default) = skip files that already exist in the
            destination; True = copy them again.

    Returns:
        pd.DataFrame: one row per label (index "label") with columns
            folder_existed (bool): the label folder existed before this call,
            copied (int): files copied in this call,
            skipped (int): files skipped because they already existed,
            n_expected (int): number of images with this label,
            n_in_folder (int): image files now in the label folder,
            ok (bool): the folder contains exactly the expected file names.
    """
    rows = []
    for label, paths in pictures_by_label.items():
        label_dir = dst_dir / label
        folder_existed = label_dir.is_dir()
        label_dir.mkdir(parents=True, exist_ok=True)

        copied = skipped = 0
        for src in paths:
            dst = label_dir / image_name(src)
            if dst.exists() and not overwrite:
                skipped += 1
                continue
            tmp = dst.with_name(dst.name + ".part")
            shutil.copyfile(src, tmp)
            src_stat = src.stat()
            os.utime(tmp, (src_stat.st_atime, src_stat.st_mtime))  # keep original timestamps
            tmp.replace(dst)
            copied += 1

        names_in_folder = {p.name for p in label_dir.iterdir() if p.suffix.lower() in (".png", ".jpg")}
        rows.append({
            "label": label,
            "folder_existed": folder_existed,
            "copied": copied,
            "skipped": skipped,
            "n_expected": len(paths),
            "n_in_folder": len(names_in_folder),
            "ok": names_in_folder == {image_name(p) for p in paths},
        })
        print(f"{label}: copied {copied}, skipped {skipped}")

    return pd.DataFrame(rows).set_index("label")


def sharpness_map(path: Path, cell: int) -> np.ndarray:
    """Return the sharpness of every cell of an image: variance of the Laplacian per cell.

    The image is read in grayscale at full resolution (no downscaling, so fine blur stays visible).

    Args:
        path (Path): image file, e.g. Path(".../A01/A01_0001.png").
        cell (int): side of one square cell in px, e.g. 64. Must divide the image width and height.

    Returns:
        np.ndarray: float64 array of shape (height // cell, width // cell),
            e.g. (24, 32) for a 2048 x 1536 image and cell = 64. Higher value = sharper.

    Raises:
        ValueError: if the cell size does not divide the image size.
    """
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)  # (1536, 2048) uint8
    h, w = gray.shape
    if h % cell or w % cell:
        raise ValueError(f"Cell size {cell} does not divide the image size {w} x {h}")
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # (h, w) -> (rows, cell, cols, cell) -> variance inside every cell -> (rows, cols)
    return laplacian.reshape(h // cell, cell, w // cell, cell).var(axis=(1, 3))


def sharp_extent(profile: np.ndarray, threshold: float, cell: int) -> tuple[int, int]:
    """Return the start and end (in px) of the sharp part of a 1-D sharpness profile.

    Args:
        profile (np.ndarray): relative sharpness per cell along one axis, e.g. shape (32,) along x.
        threshold (float): minimum relative sharpness, e.g. 0.8.
        cell (int): cell size in px, e.g. 64.

    Returns:
        tuple[int, int]: (start_px, end_px) from the first to the last cell with
            profile >= threshold, e.g. (192, 1856).

    Raises:
        ValueError: if no cell reaches the threshold.
    """
    sharp = np.flatnonzero(profile >= threshold)
    if sharp.size == 0:
        raise ValueError(f"No cell reaches the threshold {threshold}")
    return int(sharp[0] * cell), int((sharp[-1] + 1) * cell)


def build_tile_table(manifest: pd.DataFrame, roi: tuple[int, int, int, int], tile: int) -> pd.DataFrame:
    """Return one row per tile: every image that is not excluded, cut into a grid of tiles in the ROI.

    Args:
        manifest (pd.DataFrame): the manifest (one row per raw image), e.g. read from MANIFEST_CSV.
            Needs the columns file, path, prep_id, label, label_group, is_defect, fold,
            test_only and excluded.
        roi (tuple[int, int, int, int]): (x0, y0, x1, y1) of the ROI in px, e.g. (256, 192, 1792, 1344).
        tile (int): side of one square tile in px, e.g. 384. Must divide the ROI width and height.

    Returns:
        pd.DataFrame: one row per tile (images x tiles, e.g. 3966 x 12 = 47592 rows) with the image
            columns file, path, prep_id, label, label_group, is_defect, fold, test_only, plus
            tile_idx (int): 0-11, row by row from the top left,
            tile_row, tile_col (int): position in the grid, e.g. tile 5 -> row 1, column 1,
            x0, y0, x1, y1 (int): tile corners in px of the full frame, e.g. 640, 576, 1024, 960.

    Raises:
        ValueError: if the tile size does not divide the ROI size.
    """
    rx0, ry0, rx1, ry1 = roi
    if (rx1 - rx0) % tile or (ry1 - ry0) % tile:
        raise ValueError(f"Tile size {tile} does not divide the ROI {rx1 - rx0} x {ry1 - ry0}")
    n_cols, n_rows = (rx1 - rx0) // tile, (ry1 - ry0) // tile  # 4, 3

    grid = pd.DataFrame([
        {
            "tile_idx": row * n_cols + col,
            "tile_row": row,
            "tile_col": col,
            "x0": rx0 + col * tile,
            "y0": ry0 + row * tile,
            "x1": rx0 + (col + 1) * tile,
            "y1": ry0 + (row + 1) * tile,
        }
        for row in range(n_rows)
        for col in range(n_cols)
    ])
    image_columns = ["file", "path", "prep_id", "label", "label_group", "is_defect", "fold", "test_only"]
    images = manifest.loc[~manifest["excluded"], image_columns]
    return images.merge(grid, how="cross")  # every image x every tile


def load_roi(path: Path, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Read one image and return its ROI in RGB at full resolution.

    Args:
        path (Path): image file, e.g. Path(".../A/A01/A01_0141.png").
        roi (tuple[int, int, int, int]): (x0, y0, x1, y1) in px, e.g. (256, 192, 1792, 1344).

    Returns:
        np.ndarray: uint8 array of shape (height, width, 3) in RGB, e.g. (1152, 1536, 3).

    Raises:
        FileNotFoundError: if the image cannot be read.
    """
    img = cv2.imread(str(path))  # BGR, (1536, 2048, 3)
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    x0, y0, x1, y1 = roi
    return cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)


def cut_tiles(roi_img: np.ndarray, tile: int) -> np.ndarray:
    """Cut a ROI into square tiles, row by row from the top left (same order as tile_idx).

    Args:
        roi_img (np.ndarray): ROI of shape (height, width, 3), e.g. (1152, 1536, 3).
        tile (int): side of one tile in px, e.g. 384. Must divide the ROI width and height.

    Returns:
        np.ndarray: array of shape (n_tiles, tile, tile, 3), e.g. (12, 384, 384, 3).

    Raises:
        ValueError: if the tile size does not divide the ROI size.
    """
    h, w, c = roi_img.shape
    if h % tile or w % tile:
        raise ValueError(f"Tile size {tile} does not divide the ROI {w} x {h}")
    rows, cols = h // tile, w // tile  # 3, 4
    # (h, w, c) -> (rows, tile, cols, tile, c) -> (rows, cols, tile, tile, c) -> (rows * cols, tile, tile, c)
    return roi_img.reshape(rows, tile, cols, tile, c).transpose(0, 2, 1, 3, 4).reshape(-1, tile, tile, c)


def shrink_roi(roi_img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Scale the whole ROI down to the whole-picture input size (area interpolation, best for shrinking).

    Args:
        roi_img (np.ndarray): ROI of shape (height, width, 3), e.g. (1152, 1536, 3).
        size (tuple[int, int]): output (width, height) in px, e.g. (768, 576).

    Returns:
        np.ndarray: uint8 array of shape (height, width, 3), e.g. (576, 768, 3).
    """
    return cv2.resize(roi_img, size, interpolation=cv2.INTER_AREA)


def augment(img: np.ndarray, k_rot: int, flip: bool) -> np.ndarray:
    """Return one augmented version of an image: rotation by k_rot x 90 deg, then an optional flip.

    Both operations only move pixels (no interpolation), so the pixel values stay exactly the same.

    Args:
        img (np.ndarray): image of shape (height, width, 3), e.g. a tile (384, 384, 3).
        k_rot (int): number of 90 deg counter-clockwise rotations, 0-3.
        flip (bool): True = flip left-right after rotating.
            (k_rot=0, flip=True) = horizontal flip; (k_rot=2, flip=True) = vertical flip.

    Returns:
        np.ndarray: the augmented image; shape (height, width, 3) for k_rot 0 or 2,
            (width, height, 3) for k_rot 1 or 3.

    Raises:
        ValueError: if k_rot is not 0, 1, 2 or 3.
    """
    if k_rot not in (0, 1, 2, 3):
        raise ValueError(f"k_rot must be 0-3, got {k_rot}")
    out = np.rot90(img, k=k_rot, axes=(0, 1))
    if flip:
        out = out[:, ::-1]
    return np.ascontiguousarray(out)


def augment_name(k_rot: int, flip: bool) -> str:
    """Return a readable name of an augmented version, e.g. (1, True) -> "rot 90 + h-flip".

    Args:
        k_rot (int): number of 90 deg counter-clockwise rotations, 0-3.
        flip (bool): True = flip left-right after rotating.

    Returns:
        str: e.g. "original", "rot 180", "h-flip", "v-flip", "rot 90 + h-flip".
    """
    if (k_rot, flip) == (0, False):
        return "original"
    if (k_rot, flip) == (2, True):
        return "v-flip"  # 180 deg rotation + h-flip = vertical flip
    rot = f"rot {90 * k_rot}" if k_rot else ""
    return " + ".join(part for part in (rot, "h-flip" if flip else "") if part)


def measure_load_times(
    paths: list[Path], roi: tuple[int, int, int, int], tile: int, size: tuple[int, int]
) -> pd.DataFrame:
    """Time, for each image, every step of making the model inputs (one CPU core).

    Reading a cache is simulated by encoding the small images as PNG in memory and timing only the
    decoding, so nothing is written to disk.

    Args:
        paths (list[Path]): image files to time, e.g. 40 random raw images.
        roi (tuple[int, int, int, int]): (x0, y0, x1, y1) of the ROI in px, e.g. ROI.
        tile (int): tile side in px, e.g. 384.
        size (tuple[int, int]): whole-picture (width, height) in px, e.g. (768, 576).

    Returns:
        pd.DataFrame: one row per image, times in ms, with columns
            read_roi: read the 2048 x 1536 PNG from disk and cut the ROI (load_roi),
            cut_tiles: cut the ROI into tiles (cut_tiles),
            shrink: scale the ROI to the whole-picture size (shrink_roi),
            decode_whole_png: decode a cached whole picture (768 x 576 PNG),
            decode_tile_pngs: decode all cached tiles of one image (e.g. 12 PNGs of 384 x 384).
    """
    rows = []
    for path in paths:
        t0 = time.perf_counter()
        roi_img = load_roi(path, roi)
        t1 = time.perf_counter()
        tiles = cut_tiles(roi_img, tile)
        t2 = time.perf_counter()
        whole = shrink_roi(roi_img, size)
        t3 = time.perf_counter()

        whole_png = cv2.imencode(".png", whole)[1]                   # cached file, in memory
        tile_pngs = [cv2.imencode(".png", t)[1] for t in tiles]
        t4 = time.perf_counter()
        cv2.imdecode(whole_png, cv2.IMREAD_COLOR)
        t5 = time.perf_counter()
        for png in tile_pngs:
            cv2.imdecode(png, cv2.IMREAD_COLOR)
        t6 = time.perf_counter()

        rows.append({
            "read_roi": (t1 - t0) * 1000,
            "cut_tiles": (t2 - t1) * 1000,
            "shrink": (t3 - t2) * 1000,
            "decode_whole_png": (t5 - t4) * 1000,
            "decode_tile_pngs": (t6 - t5) * 1000,
        })
    return pd.DataFrame(rows)


def load_or_compute(path: Path, compute, use_saved: bool):
    """Return the saved result of a slow cell, or compute it, save it and return it.

    Args:
        path (Path): file of the saved result, e.g. CACHE_DIR / "sharpness_map.pkl".
        compute (callable): function without arguments that computes the result,
            e.g. lambda: np.mean([...], axis=0).
        use_saved (bool): True = load the saved file if it exists (e.g. USE_SAVED_RESULTS);
            False = always compute and replace the saved file.

    Returns:
        object: the result (any Python object: array, table, list, ...).
    """
    if use_saved and path.exists():
        with open(path, "rb") as f:
            result = pickle.load(f)
        print(f"Loaded saved result: {path.name}")
        return result
    t0 = time.time()
    result = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    print(f"Computed in {time.time() - t0:.0f} s and saved: {path.name}")
    return result


def save_figure(fig: plt.Figure, path: Path, overwrite: bool, dpi: int = 100) -> None:
    """Save a figure, but never replace an existing file unless overwrite is True.

    Args:
        fig (plt.Figure): the figure to save.
        path (Path): output file, e.g. RESULTS_DIR / "histograms_example.png".
        overwrite (bool): True = replace an existing file (e.g. OVERWRITE).
        dpi (int): resolution of the saved file, e.g. 100.

    Returns:
        None. Prints whether the file was saved.
    """
    if path.exists() and not overwrite:
        print(f"Not saved, file exists (set OVERWRITE = True to replace): {path}")
    else:
        fig.savefig(path, dpi=dpi)
        print("Saved", path)


def focus_features(gray: np.ndarray) -> dict[str, float]:
    """Return the focus (sharpness) measures of a grey tile: VoL, Tenengrad, Brenner.

    Args:
        gray (np.ndarray): grey tile, uint8, shape (384, 384).

    Returns:
        dict[str, float]: {"vol": ..., "tenengrad": ..., "brenner": ...}; all drop when blurred.
    """
    g = gray.astype(np.float64)
    laplacian = cv2.Laplacian(g, cv2.CV_64F)       # kernel [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)   # I_x
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)   # I_y
    dx = g[:, 2:] - g[:, :-2]                      # I(x+2, y) - I(x, y)
    dy = g[2:, :] - g[:-2, :]                      # I(x, y+2) - I(x, y)
    return {
        "vol": float(laplacian.var()),
        "tenengrad": float(np.mean(gx**2 + gy**2)),
        "brenner": float((np.sum(dx**2) + np.sum(dy**2)) / g.size),
    }


def grey_features(gray: np.ndarray) -> dict[str, float]:
    """Return entropy, mean, standard deviation and RMS contrast of a grey tile.

    Args:
        gray (np.ndarray): grey tile, uint8, shape (384, 384).

    Returns:
        dict[str, float]: {"entropy": bits, "mean": mu, "std": sigma, "rms_contrast": sigma / mu}.
    """
    p = np.bincount(gray.ravel(), minlength=256) / gray.size  # share of each grey level
    p = p[p > 0]
    mu, sigma = float(gray.mean()), float(gray.std())
    return {
        "entropy": float(-np.sum(p * np.log2(p))),
        "mean": mu,
        "std": sigma,
        "rms_contrast": sigma / mu if mu > 0 else 0.0,
    }


def colour_features(rgb: np.ndarray) -> dict[str, float]:
    """Return mean channels, chromaticities, B/R ratio, mean S and V, and the circular mean hue.

    Args:
        rgb (np.ndarray): RGB tile, uint8, shape (384, 384, 3).

    Returns:
        dict[str, float]: mean_r, mean_g, mean_b, chroma_r, chroma_g, chroma_b, b_over_r,
            mean_s, mean_v (0-255), hue_cos, hue_sin (mean hue as a point on the unit circle).
    """
    r, g, b = rgb.reshape(-1, 3).mean(axis=0)
    total = r + g + b
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)       # OpenCV: H 0-179 (2 deg steps), S and V 0-255
    hue = hsv[..., 0].astype(np.float64) * np.pi / 90  # -> radians
    sat = hsv[..., 1].astype(np.float64)
    mean_hue = np.arctan2(np.sum(sat * np.sin(hue)), np.sum(sat * np.cos(hue)))  # weighted by S
    return {
        "mean_r": float(r), "mean_g": float(g), "mean_b": float(b),
        "chroma_r": float(r / total), "chroma_g": float(g / total), "chroma_b": float(b / total),
        "b_over_r": float(b / r) if r > 0 else 0.0,
        "mean_s": float(sat.mean()), "mean_v": float(hsv[..., 2].mean()),
        "hue_cos": float(np.cos(mean_hue)), "hue_sin": float(np.sin(mean_hue)),
    }


def globule_deviation(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Return D = |I - I * G_sigma|: how much each pixel differs from the local background.

    Args:
        gray (np.ndarray): grey tile, uint8, shape (384, 384).
        sigma (float): standard deviation of the Gaussian blur in px, e.g. 15.

    Returns:
        np.ndarray: float32 array of the same shape; high on globules, near 0 on background.
    """
    g = gray.astype(np.float32)
    background = cv2.GaussianBlur(g, (0, 0), sigma)
    return np.abs(g - background)


def histogram_features(rgb: np.ndarray, gray: np.ndarray) -> dict[str, float]:
    """Return the normalised histograms (each sums to 1) as separate features.

    Args:
        rgb (np.ndarray): RGB tile, uint8, shape (384, 384, 3).
        gray (np.ndarray): the same tile in grey, uint8, shape (384, 384).

    Returns:
        dict[str, float]: hist_bright_00-15 (grey level, 16 bins of 16), hist_hue_00-17 (18 bins of
            20 deg, weighted by saturation), hist_sat_00-15 (16 bins), hist_grad_00-15 (16 bins of
            log(1 + gradient magnitude) on 0-7.3), hist_lbp_0-9 (uniform LBP codes, P = 8, R = 1).
    """
    n = gray.size
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue_deg = hsv[..., 0].astype(np.float64) * 2
    sat = hsv[..., 1].astype(np.float64)
    g = gray.astype(np.float64)
    grad = np.sqrt(cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3) ** 2 + cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3) ** 2)

    bright = np.histogram(gray, bins=16, range=(0, 256))[0] / n
    hue = np.histogram(hue_deg, bins=18, range=(0, 360), weights=sat)[0]
    hue = hue / hue.sum() if hue.sum() > 0 else hue
    sat_hist = np.histogram(sat, bins=16, range=(0, 256))[0] / n
    grad_hist = np.histogram(np.log1p(grad), bins=16, range=(0, 7.3))[0] / n  # max log1p(1443) = 7.27
    lbp = np.bincount(local_binary_pattern(gray, P=8, R=1, method="uniform").astype(int).ravel(),
                      minlength=10) / n

    features = {}
    for name, hist in [("bright", bright), ("hue", hue), ("sat", sat_hist), ("grad", grad_hist)]:
        features.update({f"hist_{name}_{k:02d}": float(v) for k, v in enumerate(hist)})
    features.update({f"hist_lbp_{k}": float(v) for k, v in enumerate(lbp)})
    return features


def tile_features(rgb: np.ndarray, globule_threshold: float, sigma: float) -> dict[str, float]:
    """Return all features of one tile (definitions in "# Classical methods").

    Args:
        rgb (np.ndarray): RGB tile, uint8, shape (384, 384, 3).
        globule_threshold (float): fixed threshold t on D for the globule area, e.g. 6.5.
        sigma (float): Gaussian blur for the local background in px, e.g. 15.

    Returns:
        dict[str, float]: 19 single-value features (focus, grey-level, colour, globule_area)
            followed by 76 histogram bins (95 values in total).
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return {
        **focus_features(gray),
        **grey_features(gray),
        **colour_features(rgb),
        "globule_area": float(np.mean(globule_deviation(gray, sigma) > globule_threshold)),
        **histogram_features(rgb, gray),
    }


def deviation_counts(
    path: Path, roi: tuple[int, int, int, int], tile: int, whole_size: tuple[int, int],
    sigma_tile: float, sigma_whole: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return how often each value of D occurs in the tiles and in the whole picture of one image.

    Used to find the globule thresholds with Otsu's method on all good images.

    Args:
        path (Path): image file, e.g. Path(".../A01_0001.png").
        roi (tuple[int, int, int, int]): (x0, y0, x1, y1) of the ROI in px, e.g. ROI.
        tile (int): tile side in px, e.g. 384.
        whole_size (tuple[int, int]): whole-picture (width, height) in px, e.g. (768, 576).
        sigma_tile, sigma_whole (float): background blur in px for tiles (15) and whole pictures (7.5).

    Returns:
        tuple[np.ndarray, np.ndarray]: counts of D in 256 bins of 1 grey level, summed over the
            12 tiles, and for the whole picture; e.g. two arrays of shape (256,).
    """
    roi_img = load_roi(path, roi)
    bins = dict(bins=256, range=(0, 256))
    tile_counts = sum(
        np.histogram(globule_deviation(cv2.cvtColor(t, cv2.COLOR_RGB2GRAY), sigma_tile), **bins)[0]
        for t in cut_tiles(roi_img, tile)
    )
    whole_gray = cv2.cvtColor(shrink_roi(roi_img, whole_size), cv2.COLOR_RGB2GRAY)
    whole_counts = np.histogram(globule_deviation(whole_gray, sigma_whole), **bins)[0]
    return tile_counts, whole_counts


def image_features(
    path: Path, roi: tuple[int, int, int, int], tile: int, whole_size: tuple[int, int],
    thresholds: tuple[float, float], sigmas: tuple[float, float],
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Return the 95 features of every tile of one image and of its whole picture.

    Args:
        path (Path): image file, e.g. Path(".../A01_0001.png").
        roi (tuple[int, int, int, int]): (x0, y0, x1, y1) of the ROI in px, e.g. ROI.
        tile (int): tile side in px, e.g. 384.
        whole_size (tuple[int, int]): whole-picture (width, height) in px, e.g. (768, 576).
        thresholds (tuple[float, float]): globule thresholds t for (tiles, whole picture).
        sigmas (tuple[float, float]): background blur in px for (tiles, whole picture), e.g. (15, 7.5).

    Returns:
        tuple[list[dict[str, float]], dict[str, float]]: a list of 12 feature dicts (one per tile,
            in tile_idx order) and one feature dict for the whole picture.
    """
    roi_img = load_roi(path, roi)
    tiles = [tile_features(t, thresholds[0], sigmas[0]) for t in cut_tiles(roi_img, tile)]
    whole = tile_features(shrink_roi(roi_img, whole_size), thresholds[1], sigmas[1])
    return tiles, whole


def run_parallel(func, items: list, workers: int, desc: str = "") -> list:
    """Apply func to every item in parallel processes, keep the order, and show a tqdm progress bar.

    Args:
        func (callable): function of one argument, e.g. partial(image_features, roi=ROI, ...).
        items (list): the arguments, e.g. a list of image paths.
        workers (int): number of processes, e.g. DATA_WORKERS = 4.
        desc (str): label of the progress bar, e.g. "features".

    Returns:
        list: func(item) for every item, in the same order as items.
    """
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(tqdm(pool.map(func, items, chunksize=4), total=len(items), desc=desc, unit="image"))


def save_table(table: pd.DataFrame, path: Path, overwrite: bool) -> None:
    """Save a table as CSV (numbers with 6 significant digits); never replace an existing file
    unless overwrite is True.

    Args:
        table (pd.DataFrame): the table to save.
        path (Path): output file, e.g. TILE_FEATURES_CSV.
        overwrite (bool): True = replace an existing file (e.g. OVERWRITE).

    Returns:
        None. Prints whether the file was saved.
    """
    if path.exists() and not overwrite:
        print(f"Not saved, file exists (set OVERWRITE = True to replace): {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, float_format="%.6g")  # 6 significant digits
        print(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB)")


def plot_label_histograms(mean_hists: pd.DataFrame, title: str) -> plt.Figure:
    """Plot the histogram features of every label (blue bars) against "good" (grey line).

    Args:
        mean_hists (pd.DataFrame): rows = histogram feature names (e.g. "hist_bright_00"),
            columns = labels (must include "good"); e.g. the mean over all tiles of each label.
        title (str): figure title.

    Returns:
        plt.Figure: one row per label, one column per histogram type in HIST_TYPES.
    """
    labels = list(mean_hists.columns)
    fig, axes = plt.subplots(len(labels), len(HIST_TYPES), figsize=(20, 2.2 * len(labels)),
                             constrained_layout=True, sharex="col")
    for i, label in enumerate(labels):
        for j, (prefix, hist_title, xlabel, edges) in enumerate(HIST_TYPES):
            ax = axes[i, j]
            rows = mean_hists.index.str.startswith(prefix)
            values, reference = mean_hists.loc[rows, label].to_numpy(), mean_hists.loc[rows, "good"].to_numpy()
            ax.bar((edges[:-1] + edges[1:]) / 2, values, width=0.85 * np.diff(edges),
                   color=SERIES_COLOUR, label="this label")
            ax.step(edges, np.r_[reference, reference[-1]], where="post", color=REFERENCE_COLOUR,
                    lw=1.5, label="good (reference)")
            ax.tick_params(labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(hist_title, fontsize=11)
            if i == len(labels) - 1:
                ax.set_xlabel(xlabel)
        axes[i, 0].set_ylabel(f"{label}\nshare of pixels", fontsize=9)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper right", ncol=2, frameon=False)
    fig.suptitle(title, fontsize=13)
    return fig


def plot_feature_boxplots(
    table: pd.DataFrame, label_of_file: pd.Series, features: list[str], label_order: list[str],
    title: str, log_features: tuple[str, ...] = ("vol", "tenengrad", "brenner"),
) -> plt.Figure:
    """Plot one small box plot per feature: the spread of the feature for every label.

    Box = 25th-75th percentile, line = median, whiskers = 5th-95th percentile (single extreme
    values are not drawn). "good" is grey, with a dashed line at its median for comparison.

    Args:
        table (pd.DataFrame): one row per tile or whole picture, with a "file" column and the
            feature columns, e.g. data["TILE_FEATURES"].
        label_of_file (pd.Series): file name -> label, e.g. "A01_0001.png" -> "good".
        features (list[str]): the features to plot, e.g. the 19 single-value features.
        label_order (list[str]): labels from left to right, e.g. ["good", "blur_light", ...].
        title (str): figure title.
        log_features (tuple[str, ...]): features drawn on a log scale (they span orders of magnitude).

    Returns:
        plt.Figure: a grid of 4 columns with one panel per feature.
    """
    labels = table["file"].map(label_of_file)
    n_cols = 4
    n_rows = -(-len(features) // n_cols)  # round up, e.g. 19 features -> 5 rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4.4 * n_rows), constrained_layout=True)
    for ax, feature in zip(axes.flat, features):
        values = [table.loc[labels == label, feature].to_numpy() for label in label_order]
        boxes = ax.boxplot(values, whis=(5, 95), showfliers=False, patch_artist=True, widths=0.6,
                           medianprops=dict(color="white", lw=1.5))
        for box, label in zip(boxes["boxes"], label_order):
            colour = REFERENCE_COLOUR if label == "good" else SERIES_COLOUR
            box.set(facecolor=colour, edgecolor=colour)
        ax.axhline(np.median(values[label_order.index("good")]), color=REFERENCE_COLOUR, lw=1, ls="--")
        if feature in log_features:
            ax.set_yscale("log")
        ax.set_title(feature, fontsize=11)
        ax.set_xticks(range(1, len(label_order) + 1), label_order, rotation=90, fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(features):]:  # empty panels at the end
        ax.set_visible(False)
    fig.suptitle(f"{title}\nbox = 25-75 %, line = median, whiskers = 5-95 %; "
                 "grey = good, dashed line = median of good", fontsize=13)
    return fig


def run_roles(folds: pd.Series, test_fold: int) -> pd.Series:
    """Return the role of every row in one evaluation run, from its fold.

    Args:
        folds (pd.Series): fold of every row (1-4, <NA> for the test-only preparations F and U),
            e.g. data["MANIFEST"]["fold"].
        test_fold (int): the fold tested in this run, 1-4.

    Returns:
        pd.Series: same index, with "test" (test fold), "validation" (VALIDATION_FOLD[test_fold]),
            "train" (the other two folds) or "test_only" (F and U, tested in every run).
    """
    roles = pd.Series("train", index=folds.index)
    roles[folds == test_fold] = "test"
    roles[folds == VALIDATION_FOLD[test_fold]] = "validation"
    roles[folds.isna()] = "test_only"
    return roles


def unusualness(scores: np.ndarray, good_val_scores: np.ndarray, two_sided: bool) -> np.ndarray:
    """Return how unusual every score is compared with the good validation images, from 0 to 100.

    p = percentile of the score among the good validation scores (share of them below it, in %;
    ties count half). One-sided (higher score = more defective): u = p. Two-sided (a defect can push
    the score either way): u = 2 * |p - 50|, so u > 95 means p < 2.5 or p > 97.5.
    Either way, about 5 % of good images get u > 95.

    Args:
        scores (np.ndarray): the score of every image, e.g. VoL, shape (n,).
        good_val_scores (np.ndarray): the scores of the good validation images, e.g. shape (156,).
        two_sided (bool): True for single features, False for "higher = more defective" scores.

    Returns:
        np.ndarray: float array of shape (n,) with values 0-100; above FLAG_PERCENTILE = flagged.
    """
    ref = np.sort(np.asarray(good_val_scores, dtype=float))
    s = np.asarray(scores, dtype=float)
    below = np.searchsorted(ref, s, side="left")      # good scores strictly below
    below_or_equal = np.searchsorted(ref, s, side="right")
    p = 100 * (below + below_or_equal) / (2 * len(ref))
    return 2 * np.abs(p - 50) if two_sided else p


def two_sided_score(values: np.ndarray, good_ref: np.ndarray) -> np.ndarray:
    """Return the two-sided unusualness u (0-100) with ties broken, for ranking (AUROC) and tiles.

    Values beyond the range of the good reference all get u = 100 (ties). The ties are broken by the
    distance from the good median in interquartile ranges: u + 0.1 * d / (1 + d). The added amount
    is < 0.1, smaller than one step of u, so the order of different u values never changes.

    Args:
        values (np.ndarray): feature values, e.g. VoL of every image or tile, shape (n,).
        good_ref (np.ndarray): the same feature of the good validation images or tiles.

    Returns:
        np.ndarray: float array of shape (n,), about 0-100.1; higher = more unusual.
    """
    ref = np.asarray(good_ref, dtype=float)
    q25, median, q75 = np.percentile(ref, [25, 50, 75])
    spread = q75 - q25 if q75 > q25 else 1.0
    distance = np.abs(np.asarray(values, dtype=float) - median) / spread
    return unusualness(values, ref, two_sided=True) + 0.1 * distance / (1 + distance)


def tile_image_scores(tiles: pd.DataFrame, feature: str, info: pd.DataFrame, agg: str) -> pd.DataFrame:
    """Return, for every run, an image score made from the 12 tile scores of one feature.

    In every run, each tile gets its two-sided unusualness (two_sided_score) compared with the good
    tiles of the validation fold; the image score is the max ("its most unusual tile") or the mean
    of its 12 tile values. Higher = more unusual (so it is evaluated one-sided).

    Args:
        tiles (pd.DataFrame): one row per tile with columns file and the feature, e.g. data["TILE_FEATURES"].
        feature (str): feature name, e.g. "vol".
        info (pd.DataFrame): index = file, columns fold and label (images that are not excluded).
        agg (str): "max" or "mean".

    Returns:
        pd.DataFrame: index = file, one column per run (1-4) with the image score (about 0-100).
    """
    files = tiles["file"]
    tile_fold, tile_label = files.map(info["fold"]), files.map(info["label"])
    image_scores = {}
    for run in FOLDS:
        roles = run_roles(tile_fold, run)
        reference = tiles.loc[(roles == "validation") & (tile_label == "good"), feature]
        tile_score = pd.Series(two_sided_score(tiles[feature], reference), index=tiles.index)
        image_scores[run] = tile_score.groupby(files).agg(agg)
    return pd.DataFrame(image_scores)


def harmonic_mean(values: list[float]) -> float:
    """Return the harmonic mean, e.g. of recalls (HAR); 0 if any value is 0.

    Args:
        values (list[float]): e.g. [specificity, sensitivity] = [0.95, 0.80].

    Returns:
        float: n / sum(1 / v), e.g. 0.869 for [0.95, 0.80].
    """
    return 0.0 if min(values) <= 0 else len(values) / sum(1 / v for v in values)


def reference_roles(reference: str) -> list[str]:
    """Return the roles whose good pictures set the cut (the reference of the unusualness).

    Args:
        reference (str): "validation" = the good pictures of the validation fold (4 slides);
            "non-test" = also those of the two training folds (12 slides). With "non-test" the training
            pictures must be scored by models that did not learn from them (see out_of_fold_scores).

    Returns:
        list[str]: e.g. ["validation"] or ["validation", "train"].

    Raises:
        ValueError: if reference is not "validation" or "non-test".
    """
    if reference not in ("validation", "non-test"):
        raise ValueError(f"reference must be 'validation' or 'non-test', got {reference}")
    return ["validation"] if reference == "validation" else ["validation", "train"]


def evaluate_method(
    scores: pd.DataFrame, info: pd.DataFrame, two_sided: bool, cut: float | None = None,
    reference: str = "validation",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one method in all runs with the common rules (percentile of good validation, 95 % cut).

    Args:
        scores (pd.DataFrame): index = file, one column per run (1-4): the method's score of every
            image in that run (for scores that need no training, the same column 4 times). Only the
            images the run uses need a score: test fold, test-only images and good validation images
            (PatchCore scores only these; the other images may be NaN).
        info (pd.DataFrame): index = file (images that are not excluded), columns fold, label, is_defect.
        two_sided (bool): True for single features (a defect can move them either way).
        cut (float | None): None = the common rule (flag if u > FLAG_PERCENTILE); a number = flag
            the images whose score is above it instead, e.g. 2.5 for votes: "at least 3 of 5 models".
        reference (str): whose good pictures set the cut: "validation" (4 slides, the sections above) or
            "non-test" (12 slides, "Final evaluation"; then the good training pictures need scores too).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            metrics: one row per run, on the test fold (A-D images): auroc, balanced_accuracy,
                har (harmonic mean of the good and defect recall), false_alarms_good,
                detected_defects; plus detected_<label> for every label in TEST_ONLY_LABELS
                (e.g. detected_frozen, detected_milk32);
            predictions: one row per image and run (test and test-only images): run, file, label,
                is_defect, role ("test" or "test_only"), score, unusualness, flagged, and
                ranking_score (what AUROC uses: the raw score, or for two-sided scores u with ties
                at 0/100 broken by the distance from the good median).
    """
    metric_rows, prediction_parts = [], []
    defect = info["is_defect"].astype(bool)
    ref_roles = reference_roles(reference)
    for run in scores.columns:
        roles = run_roles(info["fold"], run)
        s = scores[run].reindex(info.index)
        good_ref = roles.isin(ref_roles) & (info["label"] == "good")
        needed = roles.isin(["test", "test_only"]) | good_ref
        missing = s.index[needed & s.isna()]
        if len(missing):  # e.g. saved results from before images were added
            raise ValueError(f"No score for {len(missing)} images (e.g. {missing[0]}); "
                             "the saved results are out of date: set USE_SAVED_RESULTS = False")
        good_val = s[good_ref]
        u = pd.Series(unusualness(s, good_val, two_sided), index=info.index)
        flagged = u > FLAG_PERCENTILE if cut is None else s > cut
        if two_sided:  # for AUROC / ROC only: u with its ties at 100 broken (decisions use u)
            ranking = pd.Series(two_sided_score(s, good_val), index=info.index)
        else:
            ranking = s  # higher = more defective already, no ties from the percentile scale

        test, test_only = roles == "test", roles == "test_only"
        detected = flagged[test & defect].mean()          # sensitivity
        passed_good = 1 - flagged[test & ~defect].mean()  # specificity
        metric_rows.append({
            "run": run,
            # AUROC needs no threshold: on the raw score, or on u (ties broken) for two-sided scores
            "auroc": roc_auc_score(defect[test], ranking[test]),
            "balanced_accuracy": (detected + passed_good) / 2,
            "har": harmonic_mean([passed_good, detected]),
            "false_alarms_good": 1 - passed_good,
            "detected_defects": detected,
            **{f"detected_{label}": flagged[test_only & (info["label"] == label)].mean()
               for label in TEST_ONLY_LABELS},
        })
        shown = test | test_only
        prediction_parts.append(pd.DataFrame({
            "run": run, "file": info.index[shown], "label": info["label"][shown].to_numpy(),
            "is_defect": defect[shown].to_numpy(), "role": roles[shown].to_numpy(),
            "score": s[shown].to_numpy(), "unusualness": u[shown].to_numpy(),
            "flagged": flagged[shown].to_numpy(),
            "ranking_score": ranking[shown].to_numpy(),
        }))
    return pd.DataFrame(metric_rows).set_index("run"), pd.concat(prediction_parts, ignore_index=True)


def detection_table(predictions: pd.DataFrame, label_order: list[str]) -> pd.DataFrame:
    """Return the detection table: how often every label was flagged, as mean and SD over the runs.

    Args:
        predictions (pd.DataFrame): from evaluate_method (columns run, file, label, flagged).
        label_order (list[str]): labels from top to bottom, e.g. ["good", "blur_light", ...].

    Returns:
        pd.DataFrame: one row per label with should_be ("passed" for good, else "flagged"),
            images (number of different images), flagged_pct_mean and flagged_pct_sd over the runs.
            For good, flagged_pct is the false-alarm rate; for the others, the detection rate.
    """
    per_run = predictions.groupby(["label", "run"])["flagged"].mean().unstack("run") * 100
    table = pd.DataFrame({
        "should_be": ["passed" if label == "good" else "flagged" for label in label_order],
        "images": predictions.groupby("label")["file"].nunique().reindex(label_order),
        "flagged_pct_mean": per_run.mean(axis=1).reindex(label_order),
        "flagged_pct_sd": per_run.std(axis=1).reindex(label_order),
    }, index=label_order)
    return table.round(1)


def plot_evaluation(
    metrics: pd.DataFrame, predictions: pd.DataFrame, label_order: list[str], title: str
) -> plt.Figure:
    """Draw the result of one method: ROC curves, confusion matrix and detection rate per label.

    Args:
        metrics (pd.DataFrame): from evaluate_method (one row per run; column auroc).
        predictions (pd.DataFrame): from evaluate_method (columns run, label, is_defect, role,
            flagged, ranking_score).
        label_order (list[str]): labels from top to bottom in the bar chart, e.g. ["good", ...].
        title (str): name of the method, e.g. "VoL of the whole picture (two-sided)".

    Returns:
        plt.Figure: three panels:
            1. ROC curve of every run (test fold, A-D images) with its AUROC; the dot is the
               95th-percentile cut (false alarms on good vs share of defects detected);
            2. confusion matrix good/defective x passed/flagged, summed over the 4 runs;
            3. share of flagged images per label (mean +- SD over the runs); for good this is the
               false-alarm rate, the dashed line is the 5 % target.
    """
    fig, (ax_roc, ax_cm, ax_det) = plt.subplots(
        1, 3, figsize=(21, 6.5), constrained_layout=True, gridspec_kw={"width_ratios": [1, 0.8, 1.2]}
    )
    test = predictions[predictions["role"] == "test"]

    # 1. ROC curves, one per run
    for run, d in test.groupby("run"):
        fpr, tpr, _ = roc_curve(d["is_defect"], d["ranking_score"])
        ax_roc.plot(fpr, tpr, color=RUN_COLOURS[run], lw=2, label=f"run {run}: AUROC {metrics.loc[run, 'auroc']:.2f}")
        ax_roc.plot(d.loc[~d["is_defect"], "flagged"].mean(), d.loc[d["is_defect"], "flagged"].mean(), "o",
                    color=RUN_COLOURS[run], ms=9, mec="white", mew=2)
    ax_roc.plot([0, 1], [0, 1], ls="--", lw=1, color=REFERENCE_COLOUR)
    ax_roc.set(xlabel="false alarms on good (false-positive rate)", ylabel="defects detected (true-positive rate)",
               title="ROC curve per run (dot = 95th-percentile cut)", xlim=(0, 1), ylim=(0, 1.01))
    ax_roc.legend(loc="lower right", frameon=False)

    # 2. Confusion matrix, summed over the runs (every A-D image is tested once)
    truth = test["is_defect"].map({False: "good", True: "defective"})
    decision = test["flagged"].map({False: "passed", True: "flagged"})
    counts = pd.crosstab(truth, decision).reindex(index=["good", "defective"], columns=["passed", "flagged"],
                                                  fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0)
    ax_cm.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, f"{counts.iloc[i, j]}\n({100 * shares.iloc[i, j]:.0f} %)", ha="center", va="center",
                       fontsize=13, color="white" if shares.iloc[i, j] > 0.5 else "#0b0b0b")
    ax_cm.set_xticks([0, 1], ["passed", "flagged"])
    ax_cm.set_yticks([0, 1], ["good", "defective"])
    ax_cm.set(xlabel="decision", ylabel="true label", title="Confusion matrix (all 4 runs, test images A-D)")

    # 3. Share flagged per label
    table = detection_table(predictions, label_order)
    colours = [REFERENCE_COLOUR if label == "good" else SERIES_COLOUR for label in label_order]
    mean_pct, sd_pct = table["flagged_pct_mean"].fillna(0), table["flagged_pct_sd"].fillna(0)
    err_low, err_high = np.minimum(sd_pct, mean_pct), np.minimum(sd_pct, 100 - mean_pct)  # stay in 0-100 %
    ax_det.barh(label_order, mean_pct, xerr=[err_low, err_high], color=colours,
                error_kw=dict(ecolor="#52514e", lw=1, capsize=3))
    for y, (value, high) in enumerate(zip(mean_pct, err_high)):
        ax_det.text(value + high + 2, y, f"{value:.0f} %", va="center", fontsize=9)  # right of the error bar
    ax_det.axvline(100 - FLAG_PERCENTILE, ls="--", lw=1, color=REFERENCE_COLOUR)
    ax_det.invert_yaxis()
    ax_det.set(xlim=(0, 115), xlabel="flagged [%] (mean ± SD over runs)",
               title="Flagged per label (good = false alarms; dashed = 5 % target)")
    ax_det.spines[["top", "right"]].set_visible(False)

    mean = metrics.mean()
    fig.suptitle(f"{title}:  AUROC {mean['auroc']:.2f},  balanced accuracy {mean['balanced_accuracy']:.2f},  "
                 f"HAR {mean['har']:.2f},  false alarms on good {100 * mean['false_alarms_good']:.0f} %  "
                 "(mean over 4 runs)", fontsize=13)
    return fig


def plot_detection_heatmap(summary: pd.DataFrame, label_order: list[str], title: str) -> plt.Figure:
    """Compare many methods: % flagged per label as a heat map, next to each method's AUROC.

    Args:
        summary (pd.DataFrame): one row per method, with a column "method" (row name, e.g.
            "vol | whole"), one column per label with the % flagged (mean over runs), and the
            columns auroc and auroc_sd.
        label_order (list[str]): label columns from left to right, e.g. ["good", "blur_light", ...].
        title (str): figure title.

    Returns:
        plt.Figure: left, the heat map (rows = methods, columns = labels; the good column is the
            false-alarm rate, the others the detection rate); right, AUROC (mean +- SD over runs).
    """
    values = summary[label_order].to_numpy(dtype=float)
    n = len(summary)
    fig, (ax_map, ax_auc) = plt.subplots(1, 2, figsize=(18, 0.32 * n + 2.5), constrained_layout=True,
                                         gridspec_kw={"width_ratios": [4, 1]}, sharey=True)
    ax_map.imshow(values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(n):
        for j in range(len(label_order)):
            ax_map.text(j, i, f"{values[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if values[i, j] > 60 else "#0b0b0b")
    ax_map.set_xticks(range(len(label_order)), ["good\n(false alarms)"] + label_order[1:], rotation=45,
                      ha="right")
    ax_map.set_yticks(range(n), summary["method"])
    ax_map.add_patch(Rectangle((-0.5, -0.5), 1, n, fill=False, ec=REFERENCE_COLOUR, lw=2))  # good column
    ax_map.set_title("% of images flagged per label (mean over 4 runs)")

    ax_auc.barh(range(n), summary["auroc"], xerr=summary["auroc_sd"], color=SERIES_COLOUR,
                error_kw=dict(ecolor="#52514e", lw=1, capsize=2))
    ax_auc.axvline(0.5, ls="--", lw=1, color=REFERENCE_COLOUR)  # chance level
    ax_auc.set(xlim=(0.4, 1.0), xlabel="AUROC (mean ± SD)", title="AUROC")
    ax_auc.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=13)
    return fig


def anomaly_scores(detector, x: np.ndarray) -> np.ndarray:
    """Return the anomaly score of every row (higher = more unusual) for scikit-learn and PyOD detectors.

    Args:
        detector: a trained detector, e.g. LedoitWolf, OneClassSVM, IsolationForest,
            LocalOutlierFactor(novelty=True), or any PyOD detector.
        x (np.ndarray): standardised features, shape (n_rows, n_features), e.g. (47592, 95).

    Returns:
        np.ndarray: shape (n_rows,), higher = less like the good training images.
    """
    if hasattr(detector, "mahalanobis"):        # scikit-learn covariance: squared Mahalanobis distance
        return detector.mahalanobis(x)
    if hasattr(detector, "decision_scores_"):   # PyOD: decision_function is already "higher = unusual"
        return detector.decision_function(x)
    return -detector.score_samples(x)           # scikit-learn: score_samples is "higher = normal"


def one_class_scores(table: pd.DataFrame, info: pd.DataFrame, make_detector,
                     progress: tqdm | None = None, step: str = "") -> pd.DataFrame:
    """Train a one-class detector on the good training rows of every run and score all rows.

    In every run: the features are standardised with the mean and SD of the good training rows,
    the detector learns from those rows only, and then scores every row.

    Args:
        table (pd.DataFrame): one row per whole picture or per tile, with a column "file" and the
            feature columns (a "tile_idx" column is ignored), e.g. data["WHOLE_FEATURES"].
        info (pd.DataFrame): index = file, columns fold and label (images that are not excluded).
        make_detector (callable): makes a new untrained detector, e.g. ONE_CLASS_DETECTORS["LOF"].
        progress (tqdm | None): progress bar to advance by one step per run (None = no progress shown).
        step (str): what is trained, shown next to the bar, e.g. "LOF | tiles".

    Returns:
        pd.DataFrame: same index as table, one column per run (1-4) with the anomaly score.
    """
    feature_cols = [c for c in table.columns if c not in ("file", "tile_idx")]
    x_all = table[feature_cols].to_numpy(dtype=float)
    fold, label = table["file"].map(info["fold"]), table["file"].map(info["label"])
    scores = {}
    for run in FOLDS:
        if progress is not None:
            progress.set_postfix_str(f"{step} | run {run} of {len(FOLDS)}")
        train_good = ((run_roles(fold, run) == "train") & (label == "good")).to_numpy()
        scaler = StandardScaler().fit(x_all[train_good])
        x = scaler.transform(x_all)
        detector = make_detector()
        detector.fit(x[train_good])
        scores[run] = anomaly_scores(detector, x)
        if progress is not None:
            progress.update(1)
    return pd.DataFrame(scores, index=table.index)


def run_one_class_detectors(
    detectors: dict, whole: pd.DataFrame, tiles: pd.DataFrame, info: pd.DataFrame, desc: str = "one-class"
) -> dict[tuple[str, str], pd.DataFrame]:
    """Run every detector on whole pictures and on tiles, and return image scores per run.

    Args:
        detectors (dict): name -> function that makes a new detector, e.g. ONE_CLASS_DETECTORS.
        whole (pd.DataFrame): whole-picture features, e.g. data["WHOLE_FEATURES"].
        tiles (pd.DataFrame): tile features, e.g. data["TILE_FEATURES"].
        info (pd.DataFrame): index = file, columns fold and label.
        desc (str): label of the progress bar, e.g. "handcrafted one-class".

    Returns:
        dict[tuple[str, str], pd.DataFrame]: (detector name, input) -> image scores (index = file,
            columns = runs 1-4); input is "whole", "tiles max" (most unusual tile) or "tiles mean".
    """
    return run_models(detectors, whole, tiles, info, one_class_scores, desc)


def supervised_scores(table: pd.DataFrame, info: pd.DataFrame, make_model,
                      progress: tqdm | None = None, step: str = "") -> pd.DataFrame:
    """Train a classifier (good vs defective) on the training rows of every run and score all rows.

    In every run the model learns from all training rows (good and defective images of the two
    training folds; tiles get the label of their image) and then gives every row its probability of
    being defective.

    Args:
        table (pd.DataFrame): one row per whole picture or per tile, with a column "file" and the
            feature columns (a "tile_idx" column is ignored), e.g. data["WHOLE_FEATURES"].
        info (pd.DataFrame): index = file, columns fold and is_defect (images that are not excluded).
        make_model (callable): makes a new untrained classifier, e.g. SUPERVISED_MODELS["LightGBM"].
        progress (tqdm | None): progress bar to advance by one step per run (None = no progress shown).
        step (str): what is trained, shown next to the bar, e.g. "Random forest | tiles".

    Returns:
        pd.DataFrame: same index as table, one column per run (1-4) with P(defective), 0-1.
    """
    feature_cols = [c for c in table.columns if c not in ("file", "tile_idx")]
    x = table[feature_cols].to_numpy(dtype=float)
    fold = table["file"].map(info["fold"])
    y = table["file"].map(info["is_defect"]).astype(int).to_numpy()
    scores = {}
    for run in FOLDS:
        if progress is not None:
            progress.set_postfix_str(f"{step} | run {run} of {len(FOLDS)}")
        train = (run_roles(fold, run) == "train").to_numpy()
        model = make_model()
        model.fit(x[train], y[train])
        scores[run] = model.predict_proba(x)[:, 1]
        if progress is not None:
            progress.update(1)
    return pd.DataFrame(scores, index=table.index)


def run_models(
    models: dict, whole: pd.DataFrame, tiles: pd.DataFrame, info: pd.DataFrame, score_fn, desc: str = "models"
) -> dict[tuple[str, str], pd.DataFrame]:
    """Run every model on whole pictures and on tiles, and return image scores per run.

    The progress bar counts every training: models x 2 inputs (whole, tiles) x 4 runs, and shows
    next to it what is trained right now, e.g. "Random forest | tiles | run 2 of 4".

    Args:
        models (dict): name -> function that makes a new model, e.g. SUPERVISED_MODELS.
        whole (pd.DataFrame): whole-picture features, e.g. data["WHOLE_FEATURES"].
        tiles (pd.DataFrame): tile features, e.g. data["TILE_FEATURES"].
        info (pd.DataFrame): index = file, columns fold, label, is_defect.
        score_fn (callable): one_class_scores or supervised_scores.
        desc (str): label of the progress bar, e.g. "resnet18 supervised".

    Returns:
        dict[tuple[str, str], pd.DataFrame]: (model name, input) -> image scores (index = file,
            columns = runs 1-4); input is "whole", "tiles max" or "tiles mean".
    """
    results = {}
    progress = tqdm(total=len(models) * 2 * len(FOLDS), desc=desc, unit="step")
    for name, make in models.items():
        whole_scores = score_fn(whole, info, make, progress, f"{name} | whole")
        results[(name, "whole")] = whole_scores.set_axis(whole["file"].to_numpy())
        tile_scores = score_fn(tiles, info, make, progress, f"{name} | tiles")
        for agg in ("max", "mean"):
            results[(name, f"tiles {agg}")] = tile_scores.groupby(tiles["file"].to_numpy()).agg(agg)
    progress.close()
    return results


def summary_row(method: str, metrics: pd.DataFrame, predictions: pd.DataFrame, label_order: list[str]) -> dict:
    """Return one row of a method comparison table (means over the runs and % flagged per label).

    Args:
        method (str): name of the method, e.g. "LOF | tiles max".
        metrics (pd.DataFrame): from evaluate_method.
        predictions (pd.DataFrame): from evaluate_method.
        label_order (list[str]): labels, e.g. ["good", "blur_light", ...].

    Returns:
        dict: method, auroc, auroc_sd, balanced_accuracy, har, false_alarms_pct,
            false_alarms_sd_pct and the % flagged for every label.
    """
    return {
        "method": method,
        "auroc": metrics["auroc"].mean(), "auroc_sd": metrics["auroc"].std(),
        "balanced_accuracy": metrics["balanced_accuracy"].mean(), "har": metrics["har"].mean(),
        "false_alarms_pct": 100 * metrics["false_alarms_good"].mean(),
        "false_alarms_sd_pct": 100 * metrics["false_alarms_good"].std(),
        **detection_table(predictions, label_order)["flagged_pct_mean"].to_dict(),
    }


def multiclass_probabilities(
    table: pd.DataFrame, info: pd.DataFrame, make_model, classes: list[str],
    progress: tqdm | None = None, step: str = "",
) -> dict[int, np.ndarray]:
    """Train a multi-class model (which label?) on the training rows of every run; return probabilities.

    Args:
        table (pd.DataFrame): one row per whole picture or tile, with "file" and the feature columns.
        info (pd.DataFrame): index = file, columns fold and label (images that are not excluded).
        make_model (callable): makes a new untrained classifier, e.g. SUPERVISED_MODELS["LightGBM"].
        classes (list[str]): the labels the model can choose, e.g. good + the 10 defect types.
        progress (tqdm | None): progress bar to advance by one step per run (None = no progress shown).
        step (str): what is trained, shown next to the bar, e.g. "LightGBM | whole".

    Returns:
        dict[int, np.ndarray]: run (1-4) -> probabilities of shape (n_rows, n_classes), columns in
            the order of classes (each row sums to 1).
    """
    feature_cols = [c for c in table.columns if c not in ("file", "tile_idx")]
    x = table[feature_cols].to_numpy(dtype=float)
    fold, label = table["file"].map(info["fold"]), table["file"].map(info["label"])
    probabilities = {}
    for run in FOLDS:
        if progress is not None:
            progress.set_postfix_str(f"{step} | run {run} of {len(FOLDS)}")
        train = ((run_roles(fold, run) == "train") & label.isin(classes)).to_numpy()
        model = make_model()
        model.fit(x[train], label[train].to_numpy())
        proba = model.predict_proba(x)
        order = [list(model.classes_).index(c) for c in classes]  # reorder columns to `classes`
        probabilities[run] = proba[:, order]
        if progress is not None:
            progress.update(1)
    return probabilities


def run_multiclass_models(
    models: dict, whole: pd.DataFrame, tiles: pd.DataFrame, info: pd.DataFrame, classes: list[str],
    desc: str = "defect type",
) -> dict[tuple[str, str], dict[int, pd.DataFrame]]:
    """Run every multi-class model on whole pictures and on tiles; return image probabilities per run.

    The progress bar counts every training (models x 2 inputs x 4 runs) and shows what is trained now.

    Args:
        models (dict): name -> function that makes a new classifier, e.g. SUPERVISED_MODELS.
        whole (pd.DataFrame): whole-picture features. tiles (pd.DataFrame): tile features.
        info (pd.DataFrame): index = file, columns fold and label.
        classes (list[str]): the labels to choose from, e.g. good + 10 defect types.
        desc (str): label of the progress bar, e.g. "handcrafted defect type".

    Returns:
        dict: (model name, input) -> {run: DataFrame (index = file, columns = classes) of probabilities}.
            Inputs: "whole"; "tiles mean" (average of the 12 tile probabilities); "tiles max" (for every
            label its highest probability over the 12 tiles, which favours local defects).
    """
    results = {}
    progress = tqdm(total=len(models) * 2 * len(FOLDS), desc=desc, unit="step")
    for name, make in models.items():
        whole_p = multiclass_probabilities(whole, info, make, classes, progress, f"{name} | whole")
        tile_p = multiclass_probabilities(tiles, info, make, classes, progress, f"{name} | tiles")
        results[(name, "whole")] = {
            run: pd.DataFrame(p, index=whole["file"].to_numpy(), columns=classes) for run, p in whole_p.items()
        }
        for agg in ("mean", "max"):
            results[(name, f"tiles {agg}")] = {
                run: pd.DataFrame(p, columns=classes).groupby(tiles["file"].to_numpy()).agg(agg)
                for run, p in tile_p.items()
            }
    progress.close()
    return results


def evaluate_multiclass(
    probabilities: dict[int, pd.DataFrame], info: pd.DataFrame, classes: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the defect-type task (T2): the predicted label is the most likely one.

    Args:
        probabilities (dict[int, pd.DataFrame]): run -> probabilities (index = file, columns = classes).
        info (pd.DataFrame): index = file, columns fold and label.
        classes (list[str]): the labels, e.g. good + 10 defect types.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            metrics: one row per run, on the test fold (A-D images): accuracy, mean_recall (balanced
                accuracy over the 11 labels), har (harmonic mean of the 11 recalls), group_accuracy
                (right group: good / acquisition / preparation / field_position), and recall_<label>;
            predictions: one row per image and run (test and test-only images): run, file, label,
                predicted, role.
    """
    metric_rows, parts = [], []
    for run, proba in probabilities.items():
        roles = run_roles(info["fold"], run)
        predicted = proba.idxmax(axis=1).reindex(info.index)
        test = roles == "test"
        truth, guess = info["label"][test], predicted[test]
        recalls = {c: (guess[truth == c] == c).mean() for c in classes}
        metric_rows.append({
            "run": run,
            "accuracy": (guess == truth).mean(),
            "mean_recall": np.mean(list(recalls.values())),
            "har": harmonic_mean(list(recalls.values())),
            "group_accuracy": (guess.map(LABEL_GROUPS) == truth.map(LABEL_GROUPS)).mean(),
            **{f"recall_{c}": r for c, r in recalls.items()},
        })
        shown = test | (roles == "test_only")
        parts.append(pd.DataFrame({"run": run, "file": info.index[shown], "label": info["label"][shown].to_numpy(),
                                   "predicted": predicted[shown].to_numpy(), "role": roles[shown].to_numpy()}))
    return pd.DataFrame(metric_rows).set_index("run"), pd.concat(parts, ignore_index=True)


def plot_confusion_matrix(predictions: pd.DataFrame, classes: list[str], extra_rows: list[str],
                          title: str) -> plt.Figure:
    """Draw the confusion matrix of the defect-type task, summed over the runs.

    Args:
        predictions (pd.DataFrame): from evaluate_multiclass (columns label, predicted, role).
        classes (list[str]): the labels (rows and columns), e.g. good + 10 defect types.
        extra_rows (list[str]): labels shown below the matrix, e.g. TEST_ONLY_LABELS: never trained on,
            so their row only shows what the model calls them.
        title (str): figure title.

    Returns:
        plt.Figure: rows = true label, columns = predicted label; each cell = % of the row (count below);
            the diagonal = recognised correctly.
    """
    rows = classes + extra_rows
    counts = pd.crosstab(predictions["label"], predictions["predicted"]).reindex(index=rows, columns=classes,
                                                                                fill_value=0)
    shares = counts.div(counts.sum(axis=1).replace(0, 1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(13, 0.62 * len(rows) + 2.5), constrained_layout=True)
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(classes)):
            if counts.iloc[i, j] > 0:
                ax.text(j, i, f"{shares.iloc[i, j]:.0f}\n({counts.iloc[i, j]})", ha="center", va="center",
                        fontsize=8, color="white" if shares.iloc[i, j] > 60 else "#0b0b0b")
    ax.axhline(len(classes) - 0.5, color="#0b0b0b", lw=2)  # line between trained labels and extra rows
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.set(xlabel="predicted label", ylabel="true label",
           title=f"{title}\n% of each row (count), all 4 runs; below the line: never trained on")
    return fig


class ModelInputs(Dataset):
    """For every image: its 12 tiles and its whole picture, as uint8 arrays (read in parallel workers).

    Args:
        paths (list[Path]): image files, e.g. the 3,966 images that are not excluded.
        roi (tuple[int, int, int, int]): (x0, y0, x1, y1) of the ROI in px, e.g. ROI.
        tile (int): tile side in px, e.g. 384.
        whole_size (tuple[int, int]): whole-picture (width, height) in px, e.g. (768, 576).

    Returns (per item):
        tuple[np.ndarray, np.ndarray]: tiles of shape (12, 384, 384, 3) and the whole picture of shape
            (576, 768, 3), RGB uint8.
    """

    def __init__(self, paths: list[Path], roi: tuple[int, int, int, int], tile: int, whole_size: tuple[int, int]):
        self.paths, self.roi, self.tile, self.whole_size = paths, roi, tile, whole_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        roi_img = load_roi(self.paths[i], self.roi)
        return cut_tiles(roi_img, self.tile), shrink_roi(roi_img, self.whole_size)


def make_backbone(timm_name: str) -> torch.nn.Module:
    """Load a pretrained network from timm (downloaded once into data/weights) for feature extraction.

    Args:
        timm_name (str): e.g. "resnet18.a1_in1k" or "vit_small_patch14_dinov2.lvd142m".

    Returns:
        torch.nn.Module: the network in eval mode on DEVICE. CNNs return their feature maps of the last
            two stages; ViTs accept any input size divisible by the patch size.
    """
    if "vit" in timm_name:
        model = timm.create_model(timm_name, pretrained=True, num_classes=0, dynamic_img_size=True)
    else:
        model = timm.create_model(timm_name, pretrained=True, features_only=True, out_indices=(3, 4))
    return model.eval().to(DEVICE)


def backbone_features(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return one feature vector per image from a pretrained network (no training, no gradients).

    Args:
        model (torch.nn.Module): from make_backbone.
        images (torch.Tensor): uint8 batch on DEVICE, shape (n, height, width, 3), RGB.

    Returns:
        torch.Tensor: float32, shape (n, 768). CNN: the last two feature maps averaged over the image
            and concatenated. ViT (patch 14): the image is centre-cropped to a multiple of 14 px (no
            resizing), then the summary (CLS) token and the mean of the patch tokens are concatenated.
    """
    config = model.pretrained_cfg
    mean = torch.tensor(config["mean"], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(config["std"], device=DEVICE).view(1, 3, 1, 1)
    x = (images.permute(0, 3, 1, 2).float() / 255 - mean) / std  # (n, 3, h, w), normalised like ImageNet
    is_vit = hasattr(model, "patch_embed")
    # Half precision only for the ViT: the CNN's internal values overflow fp16 (NaN), so it runs in fp32
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16,
                                                enabled=is_vit and DEVICE == "cuda"):
        if is_vit:
            patch = model.patch_embed.patch_size[0]
            h, w = x.shape[2] // patch * patch, x.shape[3] // patch * patch
            top, left = (x.shape[2] - h) // 2, (x.shape[3] - w) // 2
            tokens = model.forward_features(x[:, :, top:top + h, left:left + w])
            features = torch.cat([tokens[:, 0], tokens[:, model.num_prefix_tokens:].mean(dim=1)], dim=1)
        else:  # CNN
            maps = model(x)
            features = torch.cat([m.mean(dim=(2, 3)) for m in maps], dim=1)
    features = features.float()
    if not torch.isfinite(features).all():
        raise ValueError("Non-finite deep features (NaN or inf)")
    return features


def extract_deep_features(paths: list[Path], timm_name: str, batch_images: int, workers: int,
                          parts_dir: Path, use_saved: bool, chunk: int = 200) -> dict:
    """Compute the deep features of every tile and whole picture with one pretrained network.

    Saved by image name and resumable: features are saved in chunks of `chunk` images (each chunk file
    lists its image names). With use_saved = True, images that already have saved features are not
    computed again, so after an interruption or after adding images only the missing ones are
    computed. A later chunk replaces an earlier one for the same image.

    Args:
        paths (list[Path]): image files, e.g. the images that are not excluded.
        timm_name (str): e.g. "vit_small_patch14_dinov2.lvd142m".
        batch_images (int): images per GPU batch, e.g. 4 (= 48 tiles + 4 whole pictures).
        workers (int): processes that read the images, e.g. DATA_WORKERS.
        parts_dir (Path): folder for the saved chunks, e.g. DEEP_FEATURES_DIR / "dinov2_vits14_parts".
        use_saved (bool): True = reuse saved features (e.g. USE_SAVED_RESULTS); False = compute all
            images again (new chunks, which then replace the old ones).
        chunk (int): images per saved chunk, e.g. 200.

    Returns:
        dict: "files" (list[str], unique image names in the order of paths), "tiles" (np.ndarray
            float32 of shape (n, 12, 768)), "whole" (np.ndarray float32 of shape (n, 768)).
    """
    parts_dir.mkdir(parents=True, exist_ok=True)
    names = [image_name(p) for p in paths]
    saved = {}  # image name -> (tile features (12, 768), whole-picture features (768,))
    if use_saved:
        for part in sorted(parts_dir.glob("part_*.npz")):  # oldest first, so newer chunks win
            with np.load(part) as z:
                for name, tiles, whole in zip(z["files"], z["tiles"], z["whole"]):
                    saved[str(name)] = (tiles, whole)
    todo = [p for p, name in zip(paths, names) if name not in saved]
    if todo:
        model = make_backbone(timm_name)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # new chunk files sort after the older ones
        progress = tqdm(total=len(todo), desc=timm_name.split(".")[0], unit="image")
        for k, start in enumerate(range(0, len(todo), chunk)):
            chunk_paths = todo[start:start + chunk]
            loader = DataLoader(ModelInputs(chunk_paths, ROI, CROP_SIZE, WHOLE_SIZE),
                                batch_size=batch_images, num_workers=workers)
            tile_parts, whole_parts = [], []
            for tiles, whole in loader:
                n = tiles.shape[0]
                tile_feats = backbone_features(model, tiles.to(DEVICE).flatten(0, 1))  # (n * 12, 768)
                tile_parts.append(tile_feats.view(n, -1, tile_feats.shape[1]).cpu().numpy())
                whole_parts.append(backbone_features(model, whole.to(DEVICE)).cpu().numpy())
                progress.update(n)
            chunk_names = [image_name(p) for p in chunk_paths]
            chunk_tiles, chunk_whole = np.concatenate(tile_parts), np.concatenate(whole_parts)
            np.savez(parts_dir / f"part_{stamp}_{k:03d}.npz", files=np.array(chunk_names),
                     tiles=chunk_tiles, whole=chunk_whole)
            saved.update({name: (t, w) for name, t, w in zip(chunk_names, chunk_tiles, chunk_whole)})
        progress.close()
        del model
        torch.cuda.empty_cache()
    return {"files": names,
            "tiles": np.stack([saved[name][0] for name in names]),
            "whole": np.stack([saved[name][1] for name in names])}


def deep_feature_tables(features: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn the deep features of one backbone into the same table format as the handcrafted features.

    Args:
        features (dict): from extract_deep_features ("files", "tiles" (n, 12, d), "whole" (n, d)).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: whole (one row per image: "file" + d feature columns
            "d000", "d001", ...) and tiles (one row per tile: "file", "tile_idx" + the same columns),
            ready for run_models().
    """
    n, n_tiles, d = features["tiles"].shape
    columns = [f"d{i:03d}" for i in range(d)]
    whole = pd.DataFrame(features["whole"], columns=columns)
    whole.insert(0, "file", features["files"])
    tiles = pd.DataFrame(features["tiles"].reshape(n * n_tiles, d), columns=columns)
    tiles.insert(0, "file", np.repeat(features["files"], n_tiles))
    tiles.insert(1, "tile_idx", np.tile(np.arange(n_tiles), n))
    return whole, tiles


def make_patchcore() -> PatchcoreModel:
    """Create anomalib's PatchCore model with a pretrained backbone and an empty memory bank, on DEVICE.

    Returns:
        PatchcoreModel: backbone PATCHCORE_BACKBONE (e.g. "resnet18.a1_in1k", weights downloaded once into
            data/weights), patch features from PATCHCORE_LAYERS, tile score reweighted with
            PATCHCORE_NEIGHBORS bank neighbours.
    """
    return PatchcoreModel(layers=PATCHCORE_LAYERS, backbone=PATCHCORE_BACKBONE, pre_trained=True,
                          num_neighbors=PATCHCORE_NEIGHBORS).to(DEVICE)


def tile_batches(paths: list[Path], desc: str, batch_images: int = DEEP_BATCH_IMAGES) -> Iterator[torch.Tensor]:
    """Yield the 12 native tiles of the images, a few images at a time, with a progress bar.

    Args:
        paths (list[Path]): image files, e.g. the ~320 good training images of one run.
        desc (str): label of the progress bar, e.g. "run 1: scoring".
        batch_images (int): images per batch, e.g. DEEP_BATCH_IMAGES = 4 (48 tiles).

    Yields:
        torch.Tensor: uint8 RGB tiles of shape (n_images, 12, 384, 384, 3), read in DATA_WORKERS processes.
    """
    loader = DataLoader(ModelInputs(paths, ROI, CROP_SIZE, WHOLE_SIZE), batch_size=batch_images,
                        num_workers=DATA_WORKERS)
    with tqdm(total=len(paths), desc=desc, unit="image") as progress:
        for tiles, _ in loader:
            yield tiles
            progress.update(tiles.shape[0])


def anomaly_run_images(info: pd.DataFrame, run: int) -> tuple[pd.Index, pd.Index]:
    """Return the images an anomaly model learns from and the images it scores in one run.

    Learns from: the good images of the two training folds. Scores: the test fold, the test-only images
    (F, U, M) and the good validation images (for the cut). Not scored: the training images and the
    defective validation images (the evaluation does not use them).

    Args:
        info (pd.DataFrame): index = file, columns fold and label (images that are not excluded),
            e.g. eval_info.
        run (int): evaluation run, 1-4 (= the test fold).

    Returns:
        tuple[pd.Index, pd.Index]: names of the training images (e.g. 312) and of the scored images
            (e.g. 2,100).
    """
    roles = run_roles(info["fold"], run)
    good = info["label"] == "good"
    train = info.index[(roles == "train") & good]
    scored = info.index[roles.isin(["test", "test_only"]) | ((roles == "validation") & good)]
    return train, scored


def anomaly_input(tiles: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """Turn uint8 tiles into the normalised float input of an anomalib model's backbone (no resizing).

    Args:
        tiles (torch.Tensor): uint8 RGB tiles of shape (n_images, 12, 384, 384, 3), e.g. from tile_batches.
        model (torch.nn.Module): an anomalib model (PatchCore, DFM or PaDiM); its backbone's ImageNet
            mean and SD are used.

    Returns:
        torch.Tensor: float32 on DEVICE, shape (n_images * 12, 3, 384, 384).
    """
    config = model.feature_extractor.feature_extractor.pretrained_cfg
    mean = torch.tensor(config["mean"], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(config["std"], device=DEVICE).view(1, 3, 1, 1)
    x = tiles.to(DEVICE).flatten(0, 1).permute(0, 3, 1, 2).float() / 255
    return ((x - mean) / std).contiguous()  # standard memory order (DFM reshapes the feature maps)


def patchcore_memory_bank(model: PatchcoreModel, paths: list[Path], seed: int, desc: str) -> torch.Tensor:
    """Build the PatchCore memory bank, a catalogue of good patches, from good training images.

    1. The backbone describes every tile with a 48 x 48 grid of patches (384 numbers each for ResNet-18;
       anomalib averages every patch with its 3 x 3 neighbours).
    2. A random PATCHCORE_KEEP_SHARE of the 12 x 2304 = 27,648 patches of every image is kept
       (all patches of ~320 images would need ~13 GB).
    3. anomalib's coreset selection (k-center greedy) picks PATCHCORE_BANK_SIZE of the kept patches so
       that every kept patch has a close one among them.

    Args:
        model (PatchcoreModel): from make_patchcore.
        paths (list[Path]): good training images, e.g. the ~320 good images of the two training folds.
        seed (int): random seed of the patch sampling and the coreset start, e.g. SEED + run.
        desc (str): label of the progress bar, e.g. "run 1: memory bank from 312 good images".

    Returns:
        torch.Tensor: float32 on DEVICE, shape (PATCHCORE_BANK_SIZE, 384).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)  # anomalib's coreset selection also draws from numpy's global random generator
    model.train()  # anomalib: in training mode the model returns the patch features (backbone stays in eval)
    kept = []
    with torch.no_grad():
        for tiles in tile_batches(paths, desc):
            features = model(anomaly_input(tiles, model))  # (n_images * 12 * 2304, 384)
            model.embedding_store.clear()  # anomalib would keep all features; only a random sample is kept
            n_keep = round(PATCHCORE_KEEP_SHARE * len(features))
            kept.append(features[torch.randperm(len(features), device=features.device)[:n_keep]])
    kept = torch.cat(kept)
    sampler = KCenterGreedy(embedding=kept, sampling_ratio=(PATCHCORE_BANK_SIZE + 0.5) / len(kept))
    return sampler.sample_coreset()  # shows its own progress bar ("Selecting Coreset Indices")


def anomaly_tile_scores(model: torch.nn.Module, paths: list[Path], desc: str,
                        batch_images: int = DEEP_BATCH_IMAGES) -> np.ndarray:
    """Score every tile with a fitted anomalib model (PatchCore, DFM or PaDiM). Higher = more unusual.

    - PatchCore: distance of the tile's most unusual patch to its nearest patch in the memory bank,
      slightly reweighted by anomalib (by how isolated that bank patch is among its
      PATCHCORE_NEIGHBORS neighbours);
    - DFM: how badly the main patterns of the good tiles rebuild the tile's features
      (feature reconstruction error);
    - PaDiM: the largest Mahalanobis distance of a patch from the good patches at the same position
      (after anomalib's smoothing).

    Args:
        model (torch.nn.Module): a fitted anomalib model, e.g. PatchCore with its memory bank set.
        paths (list[Path]): images to score, e.g. the ~2,100 images of one run.
        desc (str): label of the progress bar, e.g. "run 1: scoring".
        batch_images (int): images per GPU batch, e.g. DEEP_BATCH_IMAGES = 4 or PADIM_BATCH_IMAGES = 1.

    Returns:
        np.ndarray: float32, shape (n_images, 12): one score per tile, in tile_idx order.
    """
    model.eval()
    scores = []
    with torch.no_grad():
        for tiles in tile_batches(paths, desc, batch_images):
            out = model(anomaly_input(tiles, model))
            scores.append(out.pred_score.reshape(tiles.shape[0], -1).cpu().numpy())
    return np.concatenate(scores)


def run_patchcore(info: pd.DataFrame, paths: pd.Series, run: int) -> dict:
    """Train PatchCore on the good training images of one run and score the images that run needs
    (see anomaly_run_images).

    Args:
        info (pd.DataFrame): index = file, columns fold and label (images that are not excluded),
            e.g. eval_info.
        paths (pd.Series): index = file, the path of every image.
        run (int): evaluation run, 1-4 (= the test fold).

    Returns:
        dict: "bank" (np.ndarray float32 of shape (PATCHCORE_BANK_SIZE, 384): the memory bank) and
            "tile_scores" (pd.DataFrame: index = file of the scored images, columns 0-11 = tile_idx).
    """
    train, scored = anomaly_run_images(info, run)
    model = make_patchcore()
    model.memory_bank = patchcore_memory_bank(model, [Path(paths[f]) for f in train], SEED + run,
                                              f"run {run}: memory bank from {len(train)} good images")
    tile_scores = anomaly_tile_scores(model, [Path(paths[f]) for f in scored], f"run {run}: scoring")
    bank = model.memory_bank.cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return {"bank": bank, "tile_scores": pd.DataFrame(tile_scores, index=scored)}


def make_dfm() -> DFMModel:
    """Create anomalib's DFM model with a pretrained backbone (not fitted yet), on DEVICE.

    Returns:
        DFMModel: backbone DFM_BACKBONE, features from DFM_LAYER averaged over DFM_POOLING x DFM_POOLING
            cells, PCA keeping DFM_VARIANCE of the variation, score = feature reconstruction error ("fre").
    """
    return DFMModel(backbone=DFM_BACKBONE, layer=DFM_LAYER, pre_trained=True, pooling_kernel_size=DFM_POOLING,
                    n_comps=DFM_VARIANCE, score_type="fre").to(DEVICE)


def fit_dfm(model: DFMModel, paths: list[Path], desc: str) -> None:
    """Fit DFM on good training images: a PCA of the features of all their tiles (anomalib's own fit).

    Every tile becomes 9,216 numbers (its layer3 feature map averaged down to 6 x 6 cells). For ~320
    images that is 3,840 x 9,216 numbers (~140 MB), small enough for anomalib to keep all of them. The
    PCA then finds the main patterns of the good tiles; a tile they rebuild badly is unusual.

    Args:
        model (DFMModel): from make_dfm.
        paths (list[Path]): good training images, e.g. the ~320 good images of the two training folds.
        desc (str): label of the progress bar, e.g. "run 1: fit on 312 good images".
    """
    model.train()  # anomalib: in training mode the model keeps the tile features (backbone stays in eval)
    with torch.no_grad():
        for tiles in tile_batches(paths, desc):
            model(anomaly_input(tiles, model))
        model.fit()


def make_padim() -> PadimModel:
    """Create anomalib's PaDiM model with a pretrained backbone (not fitted yet), on DEVICE.

    Returns:
        PadimModel: backbone PADIM_BACKBONE, features from PADIM_LAYERS (all enlarged to the 96 x 96 grid
            of layer1), PADIM_FEATURES randomly chosen channels (seed torch first for the same choice).
    """
    return PadimModel(backbone=PADIM_BACKBONE, layers=PADIM_LAYERS, pre_trained=True,
                      n_features=PADIM_FEATURES).to(DEVICE)


def fit_padim(model: PadimModel, paths: list[Path], desc: str) -> None:
    """Fit PaDiM on good training images: one Gaussian (mean and covariance) per patch position.

    anomalib keeps all training features before fitting (3,840 tiles x 100 x 96 x 96 numbers, ~14 GB
    for ~320 images). Here the sums are added up batch by batch instead (in float64), which gives the
    same Gaussians as anomalib: covariance = (sum of x x^T - n mean mean^T) / (n - 1) + 0.01 I
    (anomalib's regularisation, plus its 1e-5 I), then inverted.

    Args:
        model (PadimModel): from make_padim.
        paths (list[Path]): good training images, e.g. the ~320 good images of the two training folds.
        desc (str): label of the progress bar, e.g. "run 1: fit on 312 good images".
    """
    model.train()  # anomalib: in training mode the model returns the patch features (backbone stays in eval)
    n = 0
    with torch.no_grad():
        for tiles in tile_batches(paths, desc, PADIM_BATCH_IMAGES):
            features = model(anomaly_input(tiles, model))  # (n_tiles, 100, 96, 96)
            model.memory_bank.clear()  # anomalib would keep all features
            x = features.flatten(2).double().permute(2, 1, 0)  # (positions, 100, n_tiles)
            if n == 0:
                sum_x = torch.zeros(x.shape[:2], dtype=torch.float64, device=x.device)
                sum_xx = torch.zeros(x.shape[0], x.shape[1], x.shape[1], dtype=torch.float64, device=x.device)
            sum_x += x.sum(dim=2)
            sum_xx.baddbmm_(x, x.transpose(1, 2))  # adds x x^T at every position, in place
            n += x.shape[2]
        mean = sum_x / n  # (positions, 100)
        sum_xx.baddbmm_(mean.unsqueeze(2), mean.unsqueeze(1), alpha=-n).div_(n - 1)  # now the covariance
        covariance = sum_xx.float()
        del sum_xx
        covariance += (0.01 + 1e-5) * torch.eye(covariance.shape[1], device=covariance.device)
        model.gaussian.mean = mean.T.float().contiguous()             # (100, positions), as anomalib stores it
        model.gaussian.inv_covariance = torch.linalg.inv(covariance)  # (positions, 100, 100)


def run_anomaly_model(info: pd.DataFrame, paths: pd.Series, run: int, make_model, fit_model,
                      batch_images: int) -> pd.DataFrame:
    """Fit an anomalib model (DFM or PaDiM) on the good training images of one run and score the images
    that run needs (see anomaly_run_images).

    Args:
        info (pd.DataFrame): index = file, columns fold and label (images that are not excluded),
            e.g. eval_info.
        paths (pd.Series): index = file, the path of every image.
        run (int): evaluation run, 1-4 (= the test fold).
        make_model (callable): makes the unfitted model, e.g. make_dfm.
        fit_model (callable): fits it on good images, e.g. fit_dfm.
        batch_images (int): images per GPU batch when scoring, e.g. DEEP_BATCH_IMAGES or PADIM_BATCH_IMAGES.

    Returns:
        pd.DataFrame: index = file of the scored images, columns 0-11 = tile score (tile_idx).
    """
    train, scored = anomaly_run_images(info, run)
    torch.manual_seed(SEED)  # PaDiM: the same random channel subset in every run
    model = make_model()
    fit_model(model, [Path(paths[f]) for f in train], f"run {run}: fit on {len(train)} good images")
    tile_scores = anomaly_tile_scores(model, [Path(paths[f]) for f in scored], f"run {run}: scoring", batch_images)
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(tile_scores, index=scored)


def final_anomaly_scores(name: str, info: pd.DataFrame, paths: pd.Series, target: int | tuple[int, int],
                         bank: np.ndarray | None = None, n_check: int = 4) -> pd.DataFrame:
    """Score the pictures the final evaluation still needs from DFM or PatchCore (tile scores).

    target = a run (int): the run's own model, fitted on the good pictures of its training folds as in the
        sections above (DFM is fitted again, the same way; PatchCore uses its saved memory bank), scores the
        defective pictures of the validation fold, never scored before (needed to choose methods on
        non-test data), plus n_check good validation pictures it scored before (to check that the model is
        the same: their new scores must equal the saved ones).
    target = a fold pair (tuple): a new model, fitted on the good pictures of these two folds, scores all
        pictures (good and defective) of the other two folds: the out-of-fold scores of training folds.
        PatchCore's memory bank uses the seed SEED + 10 * a + b, e.g. 13 for folds 1 + 3.

    Args:
        name (str): "DFM" or "PatchCore".
        info (pd.DataFrame): index = file, columns fold, label (images that are not excluded), e.g. eval_info.
        paths (pd.Series): index = file, the image path, e.g. image_paths.
        target (int | tuple[int, int]): a run (1-4) or a fold pair, e.g. (1, 3).
        bank (np.ndarray | None): PatchCore only, for a run: its saved memory bank (patchcore_runs[run]["bank"]).
        n_check (int): good validation pictures scored again for the check (runs only), e.g. 4.

    Returns:
        pd.DataFrame: index = file of the scored pictures, columns 0-11 = tile score (tile_idx).

    Raises:
        ValueError: if name is not "DFM" or "PatchCore".
    """
    if name not in ("DFM", "PatchCore"):
        raise ValueError(f"name must be 'DFM' or 'PatchCore', got {name}")
    good = info["label"] == "good"
    if isinstance(target, int):
        roles = run_roles(info["fold"], target)
        train = info.index[(roles == "train") & good]
        scored = (info.index[(roles == "validation") & ~good].tolist()
                  + sorted(info.index[(roles == "validation") & good])[:n_check])
        seed, label = SEED + target, f"run {target}"
    else:
        in_pair = info["fold"].isin(target)
        train = info.index[in_pair & good]
        scored = info.index[~in_pair & info["fold"].notna()].tolist()
        seed, label = SEED + 10 * target[0] + target[1], f"folds {target[0]} + {target[1]}"
    train_paths, scored_paths = [Path(paths[f]) for f in train], [Path(paths[f]) for f in scored]
    if name == "DFM":
        model = make_dfm()
        fit_dfm(model, train_paths, f"DFM {label}: fit on {len(train)} good images")
    else:
        model = make_patchcore()
        model.memory_bank = (torch.from_numpy(bank).to(DEVICE) if bank is not None else patchcore_memory_bank(
            model, train_paths, seed, f"PatchCore {label}: memory bank from {len(train)} good images"))
    tile_scores = anomaly_tile_scores(model, scored_paths, f"{name} {label}: scoring {len(scored)} images")
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(tile_scores, index=scored)


def patchcore_map(model: PatchcoreModel, path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the anomaly map of one image: the patch distances of its 12 tiles, put back into the ROI.

    Args:
        model (PatchcoreModel): with its memory bank set (model.memory_bank).
        path (Path): image file, e.g. Path(".../A/A01/A01_0125.png").

    Returns:
        tuple[np.ndarray, np.ndarray]: the ROI (uint8, shape (1152, 1536, 3)) and its anomaly map (float32,
            shape (1152, 1536): distance of every patch to its nearest good patch, enlarged to pixels
            and smoothed by anomalib).
    """
    roi_img = load_roi(path, ROI)
    tiles = torch.from_numpy(cut_tiles(roi_img, CROP_SIZE)).unsqueeze(0)  # (1, 12, 384, 384, 3)
    model.eval()
    with torch.no_grad():
        maps = model(anomaly_input(tiles, model)).anomaly_map.squeeze(1).cpu().numpy()  # (12, 384, 384)
    rows, cols = roi_img.shape[0] // CROP_SIZE, roi_img.shape[1] // CROP_SIZE
    # inverse of cut_tiles: (rows * cols, t, t) -> (rows, cols, t, t) -> (rows, t, cols, t) -> (rows * t, cols * t)
    full = maps.reshape(rows, cols, CROP_SIZE, CROP_SIZE).transpose(0, 2, 1, 3)
    return roi_img, full.reshape(rows * CROP_SIZE, cols * CROP_SIZE)


def plot_anomaly_maps(examples: list[dict], vmin: float, vmax: float, title: str) -> plt.Figure:
    """Draw example images next to their anomaly maps, two examples per row.

    Args:
        examples (list[dict]): one dict per example with label, file, run, roi (ROI image),
            map (anomaly map divided by the run's cut), tile_flagged (12 bools: tile above the cut)
            and flagged (bool: the image's decision).
        vmin (float): bottom of the colour scale in units of the cut, e.g. 0.5 (typical good patches).
        vmax (float): top of the colour scale in units of the cut, e.g. 1.5.
        title (str): figure title.

    Returns:
        plt.Figure: per example, left the image (thin white lines = the 12 tiles; blue frame = a tile
            above the cut, i.e. a tile that makes the image flagged) and right its anomaly map
            (white = like the good patches, dark blue = far from all of them; 1 = the run's cut).
    """
    n_rows = (len(examples) + 1) // 2
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 3.9 * n_rows), constrained_layout=True)
    for ax in axes.flat:
        ax.axis("off")
    for k, ex in enumerate(examples):
        ax_img, ax_map = axes.flat[2 * k], axes.flat[2 * k + 1]
        ax_img.imshow(ex["roi"])
        rows, cols = ex["roi"].shape[0] // CROP_SIZE, ex["roi"].shape[1] // CROP_SIZE
        for c in range(1, cols):
            ax_img.axvline(c * CROP_SIZE, color="white", lw=0.6, alpha=0.7)
        for r in range(1, rows):
            ax_img.axhline(r * CROP_SIZE, color="white", lw=0.6, alpha=0.7)
        for t in np.flatnonzero(ex["tile_flagged"]):
            r, c = divmod(t, cols)
            ax_img.add_patch(Rectangle((c * CROP_SIZE + 6, r * CROP_SIZE + 6), CROP_SIZE - 12, CROP_SIZE - 12,
                                       fill=False, ec=SERIES_COLOUR, lw=2.5))
        decision = "flagged" if ex["flagged"] else "passed"
        ax_img.set_title(f"{ex['label']}: {ex['file']} (run {ex['run']}), {decision}", fontsize=10)
        image = ax_map.imshow(ex["map"], cmap="Blues", vmin=vmin, vmax=vmax)
        ax_map.set_title(f"anomaly map, {int(np.sum(ex['tile_flagged']))} of 12 tiles above the cut", fontsize=10)
    fig.colorbar(image, ax=axes, shrink=0.5, label="patch distance / cut of the run (1 = cut)")
    fig.suptitle(title, fontsize=13)
    return fig


def run_unusualness(scores: pd.DataFrame, info: pd.DataFrame, reference: str = "validation") -> pd.DataFrame:
    """Return every image's unusualness (0-100) in every run, as evaluate_method computes it.

    u = percentile of the image's score among the scores of the good validation images of the run
    (one-sided, higher score = more defective), so u > FLAG_PERCENTILE is the method's own decision.
    Example: a score above 152 of the 156 good validation scores gives u = 97.4 (flagged).

    Args:
        scores (pd.DataFrame): index = file, one column per run (1-4), e.g. method_scores["Mahalanobis | whole"].
        info (pd.DataFrame): index = file (images that are not excluded), columns fold, label.
        reference (str): "validation" (4 slides) or "non-test" (12 slides), as in evaluate_method.

    Returns:
        pd.DataFrame: index = info.index, one column per run with u; NaN where the method has no
            score (e.g. training images, which DFM and PatchCore do not score).
    """
    ref_roles = reference_roles(reference)
    out = {}
    for run in scores.columns:
        roles = run_roles(info["fold"], run)
        s = scores[run].reindex(info.index)
        good_val = s[roles.isin(ref_roles) & (info["label"] == "good")]
        out[run] = pd.Series(unusualness(s, good_val, two_sided=False), index=info.index).where(s.notna())
    return pd.DataFrame(out)


def train_folds(run: int) -> list[int]:
    """Return the two training folds of a run (neither its test fold nor its validation fold).

    Args:
        run (int): the run (= its test fold), 1-4.

    Returns:
        list[int]: e.g. run 1 (test 1, validation 2) -> [3, 4].
    """
    return [f for f in FOLDS if f not in (run, VALIDATION_FOLD[run])]


def fold_pair_scores(table: pd.DataFrame, info: pd.DataFrame, make_model, pair: tuple[int, int],
                     kind: str) -> np.ndarray:
    """Train one model on a pair of folds and score every row (the same steps as one_class_scores /
    supervised_scores, only with a different pair of training folds).

    Args:
        table (pd.DataFrame): one row per whole picture or tile, with "file" and the feature columns.
        info (pd.DataFrame): index = file, columns fold, label, is_defect.
        make_model (callable): makes a new untrained model, e.g. ONE_CLASS_DETECTORS["LOF"].
        pair (tuple[int, int]): the two training folds, e.g. (1, 3).
        kind (str): "one-class" (learns from the good rows of the pair, features standardised on them)
            or "supervised" (learns from all rows of the pair; returns P(defective)).

    Returns:
        np.ndarray: one score per row of table, higher = more defective.
    """
    feature_cols = [c for c in table.columns if c not in ("file", "tile_idx")]
    x_all = table[feature_cols].to_numpy(dtype=float)
    train = table["file"].map(info["fold"]).isin(pair).to_numpy()
    if kind == "one-class":
        good = (table["file"].map(info["label"]) == "good").to_numpy()
        scaler = StandardScaler().fit(x_all[train & good])
        x = scaler.transform(x_all)
        detector = make_model()
        detector.fit(x[train & good])
        return anomaly_scores(detector, x)
    y = table["file"].map(info["is_defect"]).astype(int).to_numpy()
    model = make_model()
    model.fit(x_all[train], y[train])
    return model.predict_proba(x_all)[:, 1]


def run_extra_pairs(models: dict, whole: pd.DataFrame, tiles: pd.DataFrame, info: pd.DataFrame, kind: str,
                    pairs: list[tuple[int, int]], desc: str) -> dict:
    """Train every model on the fold pairs no run trains on, on whole pictures and tiles; return image scores.

    Args:
        models (dict): name -> function that makes a new model, e.g. SUPERVISED_MODELS.
        whole (pd.DataFrame): whole-picture features. tiles (pd.DataFrame): tile features.
        info (pd.DataFrame): index = file, columns fold, label, is_defect.
        kind (str): "one-class" or "supervised", as in fold_pair_scores.
        pairs (list[tuple[int, int]]): e.g. [(1, 3), (2, 4)].
        desc (str): label of the progress bar, e.g. "handcrafted supervised".

    Returns:
        dict: (model name, input) -> {pair: image scores (pd.Series, index = file)}; input is "whole",
            "tiles max" or "tiles mean", as in run_models.
    """
    results = {}
    progress = tqdm(total=len(models) * 2 * len(pairs), desc=desc, unit="model")
    for name, make in models.items():
        for input_name, table in [("whole", whole), ("tiles", tiles)]:
            for pair in pairs:
                progress.set_postfix_str(f"{name} | {input_name} | folds {pair[0]} + {pair[1]}")
                s = pd.Series(fold_pair_scores(table, info, make, pair, kind))
                if input_name == "whole":
                    results.setdefault((name, "whole"), {})[pair] = s.set_axis(table["file"].to_numpy())
                else:
                    for agg in ("max", "mean"):
                        results.setdefault((name, f"tiles {agg}"), {})[pair] = (
                            s.groupby(table["file"].to_numpy()).agg(agg))
                progress.update(1)
    progress.close()
    return results


def out_of_fold_scores(scores: pd.DataFrame, extra: dict, info: pd.DataFrame) -> pd.DataFrame:
    """Return a method's scores in which every training picture is scored by a model that never saw it.

    In run k the model learned from the two training folds, so it rates their pictures too normal. Here
    each training fold j gets the scores of the model trained on the other training fold and the
    validation fold instead (the same amount of data, and never the test fold). These models are either
    another run's own model, or one of the extra fold-pair models. Example, run 1 (test 1, validation 2,
    training 3 and 4): fold 3 is scored by the model of folds 2 + 4 (extra), fold 4 by that of folds
    2 + 3 (= run 4's model). Test, validation and test-only pictures keep the run's own scores.

    Args:
        scores (pd.DataFrame): index = file, columns = runs 1-4: the method's own scores (scikit-learn:
            all pictures; anomaly models: at least the test, test-only and validation pictures).
        extra (dict): {pair: image scores (pd.Series)} for the pairs no run trains on, from run_extra_pairs.
        info (pd.DataFrame): index = file, column fold.

    Returns:
        pd.DataFrame: index = info.index, columns = runs: the run's own scores, with the training pictures'
            scores replaced (or filled in, where the run's model did not score them).
    """
    by_pair = {tuple(sorted(train_folds(run))): scores[run] for run in scores.columns}
    by_pair.update(extra)
    out = {}
    for run in scores.columns:
        s = scores[run].reindex(info.index)
        for j in train_folds(run):
            other = [f for f in train_folds(run) if f != j][0]
            rows = info.index[info["fold"] == j]
            s.loc[rows] = by_pair[tuple(sorted((other, VALIDATION_FOLD[run])))].reindex(rows).to_numpy()
        out[run] = s
    return pd.DataFrame(out)


class WholePictures(Dataset):
    """Whole pictures (the ROI shrunk to WHOLE_SIZE) with their targets, for training or scoring a CNN.

    Args:
        paths (list[Path]): image files, e.g. the ~1,700 training images of one run.
        targets (np.ndarray): int array of shape (n, 2): [is_defect (0/1), label index in the classes
            (-1 = not a training label, e.g. frozen)], e.g. [[0, 0], [1, 2], ...].
        augment_images (bool): True (training) = every time a picture is read, one of the 4 versions in
            AUG_WHOLE is picked at random (original, h-flip, v-flip, 180 deg); False = always the original.

    Returns (per item):
        tuple[torch.Tensor, torch.Tensor]: the picture, uint8 RGB of shape (576, 768, 3), and its targets (2,).
    """

    def __init__(self, paths: list[Path], targets: np.ndarray, augment_images: bool):
        self.paths, self.targets, self.augment_images = paths, targets, augment_images

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        picture = shrink_roi(load_roi(self.paths[i], ROI), WHOLE_SIZE)
        if self.augment_images:  # torch's random numbers: seeded per worker by the DataLoader's generator
            k_rot, flip = AUG_WHOLE[int(torch.randint(len(AUG_WHOLE), (1,)))]
            picture = augment(picture, k_rot, flip)
        return torch.from_numpy(picture), torch.from_numpy(self.targets[i])


class TwoHeadCNN(torch.nn.Module):
    """A pretrained timm CNN with two outputs: good vs defective (T1) and which label (T2).

    The network's pooled features (512 numbers for ResNet-18) feed two linear layers: one number for
    "defective?" and one number per label.

    Gradient checkpointing (optional, for ResNet-50 on 4 GB): while training, the activations of layer1-4
    are not stored but recomputed in the backward pass. The gradients are the same (tested), but every
    BatchNorm layer in layer1-4 then runs twice per batch and would update its saved statistics twice.
    Its momentum is therefore lowered from 0.1 to 1 - sqrt(0.9) = 0.0513: two updates with the same batch
    = one update with 0.1 (tested: same running means). In eval mode checkpointing is off.

    Args:
        timm_name (str): e.g. "resnet18.a1_in1k" (weights downloaded once into data/weights); a ResNet.
        n_classes (int): number of labels of the second output, e.g. 11 (good + 10 defect types).
        grad_checkpointing (bool): True = save GPU memory while training (about 20% slower).

    Returns (forward):
        torch.Tensor: logits of shape (n, 1 + n_classes): column 0 = defective (sigmoid -> P(defective)),
            columns 1.. = the labels (softmax -> P(label)).
    """

    def __init__(self, timm_name: str, n_classes: int, grad_checkpointing: bool = False):
        super().__init__()
        self.backbone = timm.create_model(timm_name, pretrained=True, num_classes=0)  # -> pooled features
        self.defect_head = torch.nn.Linear(self.backbone.num_features, 1)
        self.label_head = torch.nn.Linear(self.backbone.num_features, n_classes)
        config = self.backbone.pretrained_cfg
        self.register_buffer("mean", torch.tensor(config["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(config["std"]).view(1, 3, 1, 1))
        self.grad_checkpointing = grad_checkpointing
        if grad_checkpointing:
            for layer in (self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4):
                for module in layer.modules():
                    if isinstance(module, torch.nn.BatchNorm2d):
                        module.momentum = 1 - (1 - module.momentum) ** 0.5

    def train(self, mode: bool = True) -> "TwoHeadCNN":
        """Switch training (True) or eval mode (False); gradient checkpointing, if chosen, is on only while training."""
        super().train(mode)
        self.backbone.set_grad_checkpointing(mode and self.grad_checkpointing)
        return self

    def prepare(self, pictures: torch.Tensor) -> torch.Tensor:
        """Turn uint8 RGB pictures (n, h, w, 3) on the model's device into the normalised input (n, 3, h, w)."""
        return (pictures.permute(0, 3, 1, 2).float() / 255 - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return torch.cat([self.defect_head(features), self.label_head(features)], dim=1)


def cnn_loader(paths: list[Path], targets: np.ndarray, train: bool, seed: int = SEED) -> DataLoader:
    """Return a DataLoader of whole pictures for a CNN, read in DATA_WORKERS processes.

    Args:
        paths (list[Path]): image files. targets (np.ndarray): (n, 2), as in WholePictures.
        train (bool): True = shuffled and randomly augmented; False = in order, not augmented.
        seed (int): seed of the order and the augmentation, e.g. different for every epoch, so that
            an epoch repeated after an interruption is the same.

    Returns:
        DataLoader: batches of CNN_BATCH pictures (n, 576, 768, 3) uint8 and their targets (n, 2).
    """
    return DataLoader(WholePictures(paths, targets, augment_images=train), batch_size=CNN_BATCH, shuffle=train,
                      num_workers=DATA_WORKERS, generator=torch.Generator().manual_seed(seed))


def cnn_loss(logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor,
             class_weights: torch.Tensor) -> torch.Tensor:
    """Return the CNN's loss: good-vs-defective loss + label loss, each balanced with class weights.

    Args:
        logits (torch.Tensor): from TwoHeadCNN, shape (n, 1 + n_classes).
        targets (torch.Tensor): shape (n, 2): is_defect, label index.
        pos_weight (torch.Tensor): weight of a defective picture in the first loss, e.g. 0.206 in run 1
            (= 296 good / 1,438 defective training pictures, so both groups count the same).
        class_weights (torch.Tensor): weight of every label in the second loss, shape (n_classes,),
            e.g. 0.53 for good (296 pictures in run 1) and about 1.0 for a defect type (~160 pictures).

    Returns:
        torch.Tensor: the loss (a single number), e.g. 2.9 at the start (0.69 + 2.2 for 11 labels).
    """
    defect_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, 0], targets[:, 0].float(),
                                                                       pos_weight=pos_weight)
    label_loss = torch.nn.functional.cross_entropy(logits[:, 1:], targets[:, 1], weight=class_weights)
    return defect_loss + label_loss


def cnn_epoch(model: TwoHeadCNN, loader: DataLoader, pos_weight: torch.Tensor, class_weights: torch.Tensor,
              progress: tqdm, optimizer: torch.optim.Optimizer | None = None,
              scheduler: torch.optim.lr_scheduler.LRScheduler | None = None) -> float:
    """Run one pass over the pictures: training (with an optimizer) or validation (without); return the mean loss.

    Training: the weights are updated after every CNN_ACCUMULATE batches (64 pictures), and the learning
    rate then takes one step of the scheduler. Validation: no gradients, BatchNorm uses its saved statistics.

    Args:
        model (TwoHeadCNN): the network on DEVICE.
        loader (DataLoader): from cnn_loader.
        pos_weight, class_weights (torch.Tensor): as in cnn_loss.
        progress (tqdm): progress bar, advanced by the number of pictures.
        optimizer (torch.optim.Optimizer | None): None = validation.
        scheduler (torch.optim.lr_scheduler.LRScheduler | None): learning-rate schedule (training only).

    Returns:
        float: mean loss per picture, e.g. 0.35.

    Raises:
        ValueError: if a loss is NaN or infinite.
    """
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for i, (pictures, targets) in enumerate(loader):
            loss = cnn_loss(model(model.prepare(pictures.to(DEVICE))), targets.to(DEVICE), pos_weight, class_weights)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss (NaN or inf)")
            if training:
                (loss / CNN_ACCUMULATE).backward()
                if (i + 1) % CNN_ACCUMULATE == 0 or i + 1 == len(loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
            total += loss.item() * len(pictures)
            count += len(pictures)
            progress.update(len(pictures))
    return total / count


def cnn_predict(model: TwoHeadCNN, paths: list[Path], desc: str) -> np.ndarray:
    """Return P(defective) and P(label) of every picture (no augmentation, no gradients).

    Args:
        model (TwoHeadCNN): the trained network on DEVICE.
        paths (list[Path]): image files, e.g. the test, validation and test-only images of one run.
        desc (str): label of the progress bar, e.g. "run 1: scoring".

    Returns:
        np.ndarray: float array of shape (n, 1 + n_classes): column 0 = P(defective), then P(label)
            for every label (these sum to 1), e.g. [0.98, 0.01, 0.95, ...].
    """
    loader = cnn_loader(paths, np.zeros((len(paths), 2), dtype=np.int64), train=False)
    model.eval()
    parts = []
    with torch.inference_mode(), tqdm(total=len(paths), desc=desc, unit="image") as progress:
        for pictures, _ in loader:
            logits = model(model.prepare(pictures.to(DEVICE)))
            parts.append(torch.cat([torch.sigmoid(logits[:, :1]), torch.softmax(logits[:, 1:], dim=1)], dim=1).cpu())
            progress.update(len(pictures))
    return torch.cat(parts).numpy()


class PictureTiles(Dataset):
    """The 12 native tiles of every picture (not augmented), for finding the key tiles and for scoring (design B).

    Args:
        paths (list[Path]): image files, e.g. the 1,734 training pictures of run 1.

    Returns (per item):
        torch.Tensor: uint8 RGB tiles of shape (12, 384, 384, 3), in tile_idx order.
    """

    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        return torch.from_numpy(cut_tiles(load_roi(self.paths[i], ROI), CROP_SIZE))


class KeyTiles(Dataset):
    """One tile per picture, its key tile, with the picture's targets, randomly augmented (design B training).

    Args:
        paths (list[Path]): image files, e.g. the 1,734 training pictures of run 1.
        key_tiles (np.ndarray): tile_idx (0-11) of every picture's key tile, shape (n,).
        targets (np.ndarray): int array of shape (n, 2), as in WholePictures.

    Returns (per item):
        tuple[torch.Tensor, torch.Tensor]: the tile (384, 384, 3) uint8 in one of its 8 versions
            (AUG_TILE, picked at random) and the picture's targets (2,).
    """

    def __init__(self, paths: list[Path], key_tiles: np.ndarray, targets: np.ndarray):
        self.paths, self.key_tiles, self.targets = paths, key_tiles, targets

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        tile = cut_tiles(load_roi(self.paths[i], ROI), CROP_SIZE)[self.key_tiles[i]]
        k_rot, flip = AUG_TILE[int(torch.randint(len(AUG_TILE), (1,)))]  # seeded like WholePictures
        return torch.from_numpy(augment(tile, k_rot, flip)), torch.from_numpy(self.targets[i])


def cnn_tile_logits(model: TwoHeadCNN, paths: list[Path], progress: tqdm) -> torch.Tensor:
    """Return the CNN's raw outputs for all 12 tiles of every picture (eval mode, no gradients, no augmentation).

    Args:
        model (TwoHeadCNN): the network on DEVICE.
        paths (list[Path]): image files, e.g. the 875 validation pictures of run 1.
        progress (tqdm): progress bar, advanced by the number of pictures.

    Returns:
        torch.Tensor: float tensor on the CPU of shape (n_pictures, 12, 1 + n_classes): logits per tile
            (column 0 = defective, then the labels), e.g. (875, 12, 12). The key tile of a picture is the
            one with the highest column 0.
    """
    loader = DataLoader(PictureTiles(paths), batch_size=CNN_SCORE_PICTURES, num_workers=DATA_WORKERS)
    model.eval()
    parts = []
    with torch.inference_mode():
        for tiles in loader:  # (b, 12, 384, 384, 3)
            logits = model(model.prepare(tiles.flatten(0, 1).to(DEVICE)))  # (b * 12, 1 + n_classes)
            parts.append(logits.view(len(tiles), -1, logits.shape[1]).cpu())
            progress.update(len(tiles))
    return torch.cat(parts)


def key_tile_outputs(tile_logits: torch.Tensor) -> torch.Tensor:
    """Return every picture's outputs = those of its key tile (the tile with the highest "defective" logit).

    Args:
        tile_logits (torch.Tensor): from cnn_tile_logits, shape (n, 12, 1 + n_classes).

    Returns:
        torch.Tensor: shape (n, 1 + n_classes). Example: tile logits 0 of one picture [-3.1, ..., 2.4 (tile 7),
            ...] -> its row is tile 7's logits, so P(defective) = sigmoid(2.4) = 0.92, the highest of its tiles.
    """
    return tile_logits[torch.arange(len(tile_logits)), tile_logits[:, :, 0].argmax(dim=1)]


def train_cnn(timm_name: str, info: pd.DataFrame, paths: pd.Series, run: int, classes: list[str],
              checkpoint_dir: Path, resume: bool, grad_checkpointing: bool = False, design: str = "whole") -> dict:
    """Fine-tune a two-output CNN on the pictures of one run (design A or B), then score the images the run uses.

    Training pictures: all labels of the two training folds (good and defective); validation pictures:
    the validation fold. After every epoch the validation loss is computed and two checkpoints are saved
    in checkpoint_dir: last.pt (weights + optimizer, to resume) and best.pt (the weights of the epoch with
    the lowest validation loss so far). Early stopping: training stops before epoch CNN_EPOCHS when the
    best epoch is CNN_PATIENCE epochs back, e.g. best epoch 6 -> stop after epoch 11 (also when resuming).
    At the end, best.pt scores the test, validation and test-only images. The random seed is fixed
    (SEED + run); cuDNN on the GPU may still make two trainings differ slightly.

    Design "whole" (A): the network sees the whole picture (WHOLE_SIZE). Design "tiles" (B): the network
    sees single native tiles, and a picture's outputs are those of its key tile, the tile with the highest
    "defective" output (so P(defective) of the picture = the highest of its 12 tiles). Every epoch of B:
    (1) score all 12 tiles of every training picture (eval mode, no gradients) to find the key tiles;
    (2) train on the key tiles with their pictures' targets (one tile per picture, randomly rotated or
    flipped); (3) validation loss = the loss of the validation pictures' key tiles. Only the picture's
    label is used: in a corner picture, only the tile the network finds most defective learns "corner".

    Args:
        timm_name (str): e.g. "resnet18.a1_in1k".
        info (pd.DataFrame): index = file (images that are not excluded), columns fold, label, is_defect.
        paths (pd.Series): index = file, the image path, e.g. image_paths.
        run (int): the run (= its test fold), 1-4.
        classes (list[str]): the labels of the second output, e.g. good + the 10 defect types.
        checkpoint_dir (Path): folder of last.pt and best.pt, e.g. CNN_DIR / "resnet18_whole_run1_<KEY>".
        resume (bool): True = continue after the last finished epoch if last.pt exists (e.g.
            USE_SAVED_RESULTS); False = train from the start and replace the checkpoints.
        grad_checkpointing (bool): True = save GPU memory, as in TwoHeadCNN (e.g. for ResNet-50).
        design (str): "whole" (design A) or "tiles" (design B).

    Returns:
        dict: "history" (pd.DataFrame, one row per epoch: epoch, train_loss, val_loss, lr), "best_epoch"
            (int, from 1), "scores" (pd.DataFrame, index = file of the test, validation and test-only
            images; column "p_defect" = P(defective), and one column per label with P(label)); design B
            also "tile_scores" (pd.DataFrame, same index, columns 0-11 = tile_idx: P(defective) of every tile).

    Raises:
        ValueError: if design is not "whole" or "tiles".
    """
    if design not in ("whole", "tiles"):
        raise ValueError(f"design must be 'whole' or 'tiles', got {design}")
    roles = run_roles(info["fold"], run)
    label_index = info["label"].map({c: i for i, c in enumerate(classes)}).fillna(-1).astype(int)
    targets = np.stack([info["is_defect"].astype(int).to_numpy(), label_index.to_numpy()], axis=1)
    all_paths = np.array([Path(p) for p in paths.reindex(info.index)], dtype=object)
    train, val = (roles == "train").to_numpy(), (roles == "validation").to_numpy()

    # Class weights from the training pictures: good and defective count the same, and so do all labels
    y = targets[train]
    pos_weight = torch.tensor((y[:, 0] == 0).sum() / (y[:, 0] == 1).sum(), dtype=torch.float32, device=DEVICE)
    counts = np.bincount(y[:, 1], minlength=len(classes))
    class_weights = torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32, device=DEVICE)

    torch.manual_seed(SEED + run)
    model = TwoHeadCNN(timm_name, len(classes), grad_checkpointing).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CNN_LR, weight_decay=CNN_WEIGHT_DECAY)
    updates_per_epoch = int(np.ceil(np.ceil(train.sum() / CNN_BATCH) / CNN_ACCUMULATE))
    total_updates = CNN_EPOCHS * updates_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(  # rises during the first epoch, then cosine down to 0
        optimizer,  # (a plain Python float, so that last.pt can be loaded safely)
        lambda step: float(min(1.0, (step + 1) / updates_per_epoch) * 0.5 * (1 + np.cos(np.pi * step / total_updates))))

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_file, best_file = checkpoint_dir / "last.pt", checkpoint_dir / "best.pt"
    history, best_loss, best_epoch = [], float("inf"), 0
    if resume and last_file.exists():
        state = torch.load(last_file, map_location=DEVICE)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, best_loss, best_epoch = state["history"], state["best_loss"], state["best_epoch"]
        print(f"Resumed after epoch {len(history)} of {CNN_EPOCHS}: {last_file}")

    # Pictures per epoch for the progress bar; design B reads every training picture twice (key tiles, training)
    per_epoch = int(train.sum() + val.sum()) + (int(train.sum()) if design == "tiles" else 0)
    progress = tqdm(total=CNN_EPOCHS * per_epoch, initial=len(history) * per_epoch,
                    desc=f"{timm_name.split('.')[0]} {design} run {run}", unit="image")
    for epoch in range(len(history), CNN_EPOCHS):
        if epoch - best_epoch >= CNN_PATIENCE:  # no lower validation loss for CNN_PATIENCE epochs: stop
            break
        done = f" | last val loss {history[-1]['val_loss']:.3f}, best {best_loss:.3f} (epoch {best_epoch})" if history else ""
        progress.set_postfix_str(f"epoch {epoch + 1} of {CNN_EPOCHS}{done}")
        if design == "whole":
            train_loss = cnn_epoch(model, cnn_loader(list(all_paths[train]), targets[train], train=True,
                                                     seed=SEED + 100 * run + epoch),
                                   pos_weight, class_weights, progress, optimizer, scheduler)
            val_loss = cnn_epoch(model, cnn_loader(list(all_paths[val]), targets[val], train=False),
                                 pos_weight, class_weights, progress)
        else:
            key_tiles = cnn_tile_logits(model, list(all_paths[train]), progress)[:, :, 0].argmax(dim=1).numpy()
            key_loader = DataLoader(KeyTiles(list(all_paths[train]), key_tiles, targets[train]), batch_size=CNN_BATCH,
                                    shuffle=True, num_workers=DATA_WORKERS,
                                    generator=torch.Generator().manual_seed(SEED + 100 * run + epoch))
            train_loss = cnn_epoch(model, key_loader, pos_weight, class_weights, progress, optimizer, scheduler)
            val_outputs = key_tile_outputs(cnn_tile_logits(model, list(all_paths[val]), progress))
            val_loss = cnn_loss(val_outputs.to(DEVICE), torch.from_numpy(targets[val]).to(DEVICE),
                                pos_weight, class_weights).item()
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                        "lr": float(optimizer.param_groups[0]["lr"])})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch + 1
            torch.save(model.state_dict(), best_file)
        temporary = checkpoint_dir / "last.tmp"  # written first, so an interruption never leaves a broken last.pt
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "history": history, "best_loss": best_loss,
                    "best_epoch": best_epoch}, temporary)
        temporary.replace(last_file)
    if len(history) < CNN_EPOCHS:  # stopped early: the bar ends at 100 % and says why
        progress.total = progress.n
        progress.set_postfix_str(f"stopped after epoch {len(history)}: best val loss {best_loss:.3f} "
                                 f"(epoch {best_epoch}), no lower one in the {CNN_PATIENCE} epochs after it")
    progress.close()

    model.load_state_dict(torch.load(best_file, map_location=DEVICE))
    scored = (roles != "train").to_numpy()
    result = {"history": pd.DataFrame(history), "best_epoch": best_epoch}
    if design == "whole":
        probabilities = cnn_predict(model, list(all_paths[scored]), f"run {run}: scoring")
    else:
        with tqdm(total=int(scored.sum()), desc=f"run {run}: scoring", unit="image") as scoring:
            tile_logits = cnn_tile_logits(model, list(all_paths[scored]), scoring)
        key = key_tile_outputs(tile_logits)
        probabilities = torch.cat([torch.sigmoid(key[:, :1]), torch.softmax(key[:, 1:], dim=1)], dim=1).numpy()
        result["tile_scores"] = pd.DataFrame(torch.sigmoid(tile_logits[:, :, 0]).numpy(), index=info.index[scored])
    del model, optimizer
    torch.cuda.empty_cache()
    result["scores"] = pd.DataFrame(probabilities, index=info.index[scored], columns=["p_defect"] + classes)
    return result


def cnn_gradcam(model: TwoHeadCNN, path: Path, output: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a whole picture, its Grad-CAM map for one output of the CNN, and the CNN's probabilities.

    Grad-CAM weighs every channel of the last feature map (layer4: 24 x 18 cells for 768 x 576 px) by how
    much it raises the chosen output, keeps the positive part, enlarges it to the picture size and scales
    it to 0-1 within the picture. Dark = the region that raises the output most.

    Args:
        model (TwoHeadCNN): the trained network on DEVICE (a ResNet: it has backbone.layer4).
        path (Path): image file.
        output (int): 0 = the "defective" output; 1 + i = the output of label classes[i].

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: the picture (576, 768, 3) uint8, the map (576, 768)
            float 0-1, and the probabilities (1 + n_classes,) as from cnn_predict.
    """
    picture = shrink_roi(load_roi(path, ROI), WHOLE_SIZE)
    x = model.prepare(torch.from_numpy(picture).unsqueeze(0).to(DEVICE))
    with GradCAM(model=model, target_layers=[model.backbone.layer4[-1]]) as cam:
        heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(output)])[0]
    with torch.inference_mode():
        logits = model(x)[0]
    probabilities = torch.cat([torch.sigmoid(logits[:1]), torch.softmax(logits[1:], dim=0)]).cpu().numpy()
    return picture, heat, probabilities


def cnn_tile_gradcam(model: TwoHeadCNN, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a picture, its Grad-CAM map put together from its 12 tiles (design B), and the CNN's outputs.

    Every tile gets its own Grad-CAM map of the "defective" output (scaled 0-1 within the tile by Grad-CAM),
    weighted by the tile's P(defective); the 12 maps are put side by side and scaled to 0-1 within the
    picture. So a tile with P = 0.05 stays almost white, and the darkest spot lies in the most defective tile.

    Args:
        model (TwoHeadCNN): the trained network on DEVICE (a ResNet: it has backbone.layer4).
        path (Path): image file.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: the picture shrunk to WHOLE_SIZE (576, 768, 3)
            uint8, the map (576, 768) float 0-1, P(defective) of the 12 tiles (12,), and the picture's
            probabilities (1 + n_classes,) as in train_cnn (those of its key tile).
    """
    roi_img = load_roi(path, ROI)
    x = model.prepare(torch.from_numpy(cut_tiles(roi_img, CROP_SIZE)).to(DEVICE))  # (12, 3, 384, 384)
    model.eval()
    with torch.inference_mode():
        logits = model(x).cpu()
    tile_p = torch.sigmoid(logits[:, 0]).numpy()
    key = logits[int(logits[:, 0].argmax())]
    probabilities = torch.cat([torch.sigmoid(key[:1]), torch.softmax(key[1:], dim=0)]).numpy()
    with GradCAM(model=model, target_layers=[model.backbone.layer4[-1]]) as cam:
        maps = [cam(input_tensor=x[i:i + 1], targets=[ClassifierOutputTarget(0)])[0] * tile_p[i] for i in range(len(x))]
    n_cols = (ROI[2] - ROI[0]) // CROP_SIZE  # 4 tiles per row
    stitched = np.block([maps[r * n_cols:(r + 1) * n_cols] for r in range(len(maps) // n_cols)])  # (1152, 1536)
    heat = cv2.resize(stitched / max(stitched.max(), 1e-7), WHOLE_SIZE, interpolation=cv2.INTER_AREA)
    return shrink_roi(roi_img, WHOLE_SIZE), heat, tile_p, probabilities


def tile_flags(tile_scores: pd.DataFrame, picture_scores: pd.Series, info: pd.DataFrame, run: int) -> pd.DataFrame:
    """Return which tiles are flagged in one run (design B): those above the run's picture cut.

    The cut is the same as for the pictures: a tile is flagged if its P(defective) is above the 95th
    percentile of the good validation pictures' scores (unusualness > FLAG_PERCENTILE). A picture's score is
    its highest tile score, so a picture is flagged exactly when at least one of its tiles is flagged.
    Example: cut 0.62; tile scores 0.95, 0.90, 0.70 and nine below 0.1 -> 3 flagged tiles.

    Args:
        tile_scores (pd.DataFrame): index = file, columns 0-11 (tile_idx), P(defective) of every tile in this run.
        picture_scores (pd.Series): index = file, the pictures' P(defective) in this run (for the cut).
        info (pd.DataFrame): index = file (images that are not excluded), columns fold, label.
        run (int): the run, 1-4.

    Returns:
        pd.DataFrame: bool, same shape as tile_scores: True = flagged tile.
    """
    roles = run_roles(info["fold"], run)
    s = picture_scores.reindex(info.index)
    good_val = s[(roles == "validation") & (info["label"] == "good")]
    return tile_scores.apply(lambda column: pd.Series(
        unusualness(column, good_val, two_sided=False) > FLAG_PERCENTILE, index=column.index))


def plot_training_curves(runs: dict[int, dict], title: str) -> plt.Figure:
    """Draw the training and validation loss of every run per epoch, with the epoch that was kept.

    Args:
        runs (dict[int, dict]): run -> result of train_cnn ("history", "best_epoch").
        title (str): figure title, e.g. "CNN resnet18, whole picture".

    Returns:
        plt.Figure: left, the loss on the (augmented) training pictures; right, on the validation fold,
            with a dot at the kept epoch. Same log scale on both, so they can be compared.
    """
    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True, sharey=True)
    for run, result in runs.items():
        h = result["history"]
        ax_train.plot(h["epoch"], h["train_loss"], color=RUN_COLOURS[run], lw=2, label=f"run {run}")
        ax_val.plot(h["epoch"], h["val_loss"], color=RUN_COLOURS[run], lw=2,
                    label=f"run {run}: kept epoch {result['best_epoch']}")
        best = h.loc[h["epoch"] == result["best_epoch"], "val_loss"].iloc[0]
        ax_val.plot(result["best_epoch"], best, "o", color=RUN_COLOURS[run], ms=9, mec="white", mew=2)
    ax_train.set(yscale="log", xlabel="epoch", ylabel="loss (defective + label)", title="training pictures (augmented)")
    ax_val.set(xlabel="epoch", title="validation fold (dot = kept epoch: lowest loss)")
    for ax in (ax_train, ax_val):
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=13)
    return fig


def plot_gradcam(examples: list[dict], title: str,
                 map_label: str = "Grad-CAM of the \"defective\" output (0-1 within the picture)") -> plt.Figure:
    """Draw whole pictures next to their Grad-CAM maps, two examples per row.

    Args:
        examples (list[dict]): one dict per picture with label, file, run, picture (576, 768, 3),
            map (576, 768) 0-1, p_defect (float), predicted (str, most likely label), flagged (bool).
            Design B also: tile_p (12 floats, P(defective) per tile) and tile_flagged (12 bools): the
            picture then shows its 4 x 3 tiles with their P(defective), flagged tiles framed in FLAG_COLOUR.
        title (str): figure title.
        map_label (str): label of the colour bar.

    Returns:
        plt.Figure: per example the picture (title: label, file, run, decision) and its map (title:
            P(defective) and the predicted label), with one shared colour bar.
    """
    n_rows = int(np.ceil(len(examples) / 2))
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 3.4 * n_rows), constrained_layout=True)
    for ax in axes.ravel():
        ax.axis("off")
    for i, ex in enumerate(examples):
        ax_img, ax_map = axes[i // 2, 2 * (i % 2)], axes[i // 2, 2 * (i % 2) + 1]
        ax_img.imshow(ex["picture"])
        decision = "flagged" if ex["flagged"] else "passed"
        ax_img.set_title(f"{ex['label']}: {ex['file']} (run {ex['run']}), {decision}", fontsize=10)
        if "tile_p" in ex:  # design B: tile grid with P(defective); flagged tiles drawn last, on top
            size = ex["picture"].shape[0] / 3  # tile side in the shown picture: 576 / 3 = 192 px
            for t in sorted(range(len(ex["tile_p"])), key=lambda t: ex["tile_flagged"][t]):
                r, c = divmod(t, 4)
                flagged_tile = ex["tile_flagged"][t]
                ax_img.add_patch(Rectangle((c * size, r * size), size, size, fill=False,
                                           edgecolor=FLAG_COLOUR if flagged_tile else "white",
                                           lw=3 if flagged_tile else 0.8))
                ax_img.text(c * size + 8, r * size + 8, f"{ex['tile_p'][t]:.2f}", color="white", fontsize=8,
                            va="top", bbox=dict(facecolor="black", alpha=0.5, pad=1, lw=0))
        image = ax_map.imshow(ex["map"], cmap="Blues", vmin=0, vmax=1)
        ax_map.set_title(f"P(defective) {ex['p_defect']:.2f}, predicted label: {ex['predicted']}", fontsize=10)
    fig.colorbar(image, ax=axes, shrink=0.5, label=map_label)
    fig.suptitle(title, fontsize=13)
    return fig


# %% [markdown]
# # Loading data

# %%
# Raw images (4000 PNG) followed by the 3.2 % fresh milk images (605 JPG in EXTRA_DIR)
data["RAW_PICTURES"] = (sorted(p for p in RAW_DIR.rglob("*") if p.is_file())
                        + sorted(p for p in EXTRA_DIR.rglob("*") if p.is_file()))
print(len(data["RAW_PICTURES"]), "files")

# %%
data["PICTURES_BY_LABEL"] = {}
for p in data["RAW_PICTURES"]:
    data["PICTURES_BY_LABEL"].setdefault(label_of(p), []).append(p)

{label: len(paths) for label, paths in data["PICTURES_BY_LABEL"].items()}

# %% [markdown]
# **Excluded images (bad quality).** After a visual review, some images were found to be of bad
# quality (e.g. blurry images labelled `good`) and are removed from the dataset. The list with the
# label and the reason for each image is in `data/excluded_images.csv`. Only the dataset in this
# notebook changes; the raw files are not touched.
#
# Visual review of all classes, completed 2026-11-08:
#
# | Label | Result |
# |---|---|
# | good | 31 excluded (blurry, accidental defocus) |
# | blur_light | 1 excluded |
# | bubbles | 2 excluded |
# | blur_heavy, corner, overstained, undiluted, overdiluted, frozen, uht | all OK |
# | filter_yellow, filter_blue | all OK; also confirmed by a mean-colour check (no overlap with other classes) |
#
# Images that have a second defect (e.g. a blurry `filter_yellow` image) stay in their class.

# %%
data["EXCLUDED"] = pd.read_csv(EXCLUDED_CSV)
excluded_names = set(data["EXCLUDED"]["file"])

# Catch typos: every excluded name must exist in the raw data
unknown = excluded_names - {image_name(p) for p in data["RAW_PICTURES"]}
if unknown:
    raise ValueError(f"Excluded files not found in the raw data: {sorted(unknown)}")

data["PICTURES_BY_LABEL"] = {
    label: [p for p in paths if image_name(p) not in excluded_names]
    for label, paths in data["PICTURES_BY_LABEL"].items()
}
# Fingerprint of the images in use: part of the file names of saved model results, so that they
# are recomputed automatically when images are added or excluded
data["KEY"] = data_key([image_name(p) for paths in data["PICTURES_BY_LABEL"].values() for p in paths])
print(len(excluded_names), "images excluded | images in use:", data["KEY"])
{label: len(paths) for label, paths in data["PICTURES_BY_LABEL"].items()}

# %%
# Copy the raw images to MFG_DIR/<label>/ (about 38 GB; files that already exist are skipped)
copy_summary = load_or_compute(
    CACHE_DIR / f"copy_summary_{data['KEY']}.pkl",
    lambda: copy_pictures_by_label(data["PICTURES_BY_LABEL"], MFG_DIR, overwrite=False),
    USE_SAVED_RESULTS,
)
copy_summary

# %%
# First image of every label (full resolution, axes in pixels), 4 per row
n_rows = -(-len(data["PICTURES_BY_LABEL"]) // 4)  # round up, e.g. 14 labels -> 4 rows
fig, axes = plt.subplots(n_rows, 4, figsize=(20, 4 * n_rows), constrained_layout=True)
for ax, (label, paths) in zip(axes.flat, data["PICTURES_BY_LABEL"].items()):
    img = cv2.cvtColor(cv2.imread(str(paths[0])), cv2.COLOR_BGR2RGB)  # OpenCV reads BGR
    ax.imshow(img)
    ax.set_title(f"{label}  ({image_name(paths[0])})")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
for ax in axes.flat[len(data["PICTURES_BY_LABEL"]):]:  # empty panels at the end
    ax.set_visible(False)

fig_path = RESULTS_DIR / "example_images.png"
if fig_path.exists() and not OVERWRITE:
    print(f"Not saved, file exists (set OVERWRITE = True to replace): {fig_path}")
else:
    fig.savefig(fig_path, dpi=100)
    print("Saved", fig_path)
plt.show()

# %% [markdown]
# # Creating manifest
#
# One row per raw image (all 4605, incl. the 3.2 % milk; the excluded images are kept and flagged)
# with its label, the preparation details from the preparation log (CLAUDE.md §7.4) and the fold
# (§9 Step 6). F, U and M have no fold: they are test only. Saved to `data/manifest.csv`; later steps
# read it.

# %%
data["MANIFEST"] = build_manifest(data["RAW_PICTURES"], set(data["EXCLUDED"]["file"]))
data["MANIFEST"].head()

# %%
# Check: raw images per label vs CLAUDE.md §7.3, and images used (not excluded) per fold
m = data["MANIFEST"]
manifest_check = m.groupby("label", sort=False).agg(raw=("file", "size"), excluded=("excluded", "sum"))
manifest_check["expected_raw"] = pd.Series(EXPECTED_COUNTS)
manifest_check["ok"] = manifest_check["raw"] == manifest_check["expected_raw"]

used = m[~m["excluded"]].assign(
    fold_name=lambda d: ("fold " + d["fold"].astype("string")).fillna("test only")
)
manifest_check = manifest_check.join(
    used.pivot_table(index="label", columns="fold_name", values="file", aggfunc="size", fill_value=0)
)
if not manifest_check["ok"].all():
    print("COUNTS DIFFER FROM CLAUDE.md §7.3 - check before going on!")
manifest_check

# %%
# Save data/manifest.csv (an existing, different file is only replaced when OVERWRITE = True)
manifest_text = data["MANIFEST"].to_csv(index=False)
if not MANIFEST_CSV.exists():
    MANIFEST_CSV.write_text(manifest_text)
    print("Saved", MANIFEST_CSV)
elif MANIFEST_CSV.read_text() == manifest_text:
    print("Unchanged:", MANIFEST_CSV)
elif OVERWRITE:
    MANIFEST_CSV.write_text(manifest_text)
    print("Replaced", MANIFEST_CSV)
else:
    print(f"NOT saved: {MANIFEST_CSV} exists and differs (set OVERWRITE = True to replace it)")

# %% [markdown]
# # ROI
#
# The edges of every frame are softer than the centre (field curvature, vignetting), so only a
# central region of interest (ROI) is used. Sharpness map: variance of the Laplacian in 64 x 64 px
# cells, averaged over all good images (after exclusions), divided by the sharpness in the
# centre (central 256 x 256 px). The sharp area is where this relative sharpness is at least 80 %.
# The ROI (4 x 3 tiles of 384 px, centred) should lie inside the sharp area.

# %%
# All good images (after exclusions) are used for the sharpness map (takes about 35 s)
data["ROI_IMAGES"] = data["PICTURES_BY_LABEL"]["good"]

# Average sharpness map, relative to the centre of the frame
mean_map = load_or_compute(
    CACHE_DIR / "sharpness_map.pkl",
    lambda: np.mean([sharpness_map(p, SHARP_TILE) for p in tqdm(data["ROI_IMAGES"], desc="sharpness map")],
                    axis=0),
    USE_SAVED_RESULTS,
)
rows, cols = mean_map.shape  # (24, 32)
centre = mean_map[rows // 2 - 2:rows // 2 + 2, cols // 2 - 2:cols // 2 + 2].mean()
data["SHARPNESS_MAP"] = mean_map / centre

print(len(data["ROI_IMAGES"]), "good images used")

# %%
rel = data["SHARPNESS_MAP"]

# Sharp extent along x (averaged over the central third of the rows) and along y (central third of columns)
x_profile = rel[rows // 3:2 * rows // 3].mean(axis=0)
y_profile = rel[:, cols // 3:2 * cols // 3].mean(axis=1)
sharp_x = sharp_extent(x_profile, ROI_THRESHOLD, SHARP_TILE)
sharp_y = sharp_extent(y_profile, ROI_THRESHOLD, SHARP_TILE)

# Relative sharpness inside the ROI
x0, y0, x1, y1 = ROI
inside = rel[y0 // SHARP_TILE:y1 // SHARP_TILE, x0 // SHARP_TILE:x1 // SHARP_TILE]

pd.DataFrame({
    "value": [
        f"x {sharp_x[0]}-{sharp_x[1]}, y {sharp_y[0]}-{sharp_y[1]}",
        f"x {x0}-{x1}, y {y0}-{y1}",
        f"{inside.min():.2f}",
        f"{(inside >= ROI_THRESHOLD).mean():.0%}",
    ]
}, index=[
    f"sharp extent through the centre (>= {ROI_THRESHOLD:.0%}) [px]",
    "ROI [px]",
    "lowest relative sharpness inside the ROI",
    f"share of ROI cells >= {ROI_THRESHOLD:.0%}",
])

# %%
fig, (ax_map, ax_x, ax_y) = plt.subplots(
    1, 3, figsize=(20, 5.5), gridspec_kw={"width_ratios": [1.6, 1, 1]}, constrained_layout=True
)

# Map with the 80 % contour, the ROI and the first guess
h, w = rows * SHARP_TILE, cols * SHARP_TILE
im = ax_map.imshow(rel, extent=(0, w, h, 0), cmap="viridis")
ax_map.contour(
    (np.arange(cols) + 0.5) * SHARP_TILE, (np.arange(rows) + 0.5) * SHARP_TILE, rel,
    levels=[ROI_THRESHOLD], colors="white", linewidths=1.5,
)
ax_map.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="red", lw=2, label="ROI"))
gx0, gy0, gx1, gy1 = ROI_FIRST_GUESS
ax_map.add_patch(Rectangle((gx0, gy0), gx1 - gx0, gy1 - gy0, fill=False, color="orange", lw=1.5,
                           ls="--", label="first guess"))
ax_map.plot([], [], color="white", label=f"{ROI_THRESHOLD:.0%} contour")
ax_map.legend(loc="lower right")
ax_map.set(title="Relative sharpness (1 = centre)", xlabel="x [px]", ylabel="y [px]")
fig.colorbar(im, ax=ax_map, shrink=0.8)

# Profiles through the centre
for ax, profile, (lo, hi), axis in [(ax_x, x_profile, (x0, x1), "x"), (ax_y, y_profile, (y0, y1), "y")]:
    ax.plot((np.arange(profile.size) + 0.5) * SHARP_TILE, profile, marker="o")
    ax.axhline(ROI_THRESHOLD, color="grey", ls="--", label=f"{ROI_THRESHOLD:.0%}")
    ax.axvspan(lo, hi, color="red", alpha=0.1, label="ROI")
    ax.set(title=f"Profile along {axis} (central third)", xlabel=f"{axis} [px]",
           ylabel="relative sharpness")
    ax.legend()

fig_path = RESULTS_DIR / "roi_sharpness_map.png"
if fig_path.exists() and not OVERWRITE:
    print(f"Not saved, file exists (set OVERWRITE = True to replace): {fig_path}")
else:
    fig.savefig(fig_path, dpi=200)
    print("Saved", fig_path)
plt.show()

# %%
# The ROI cut into tiles of CROP_SIZE px, drawn on the first image of every label (4 per row)
x0, y0, x1, y1 = ROI
n_cols = (x1 - x0) // CROP_SIZE            # 4
n_tiles = n_cols * ((y1 - y0) // CROP_SIZE)  # 12

n_rows = -(-len(data["PICTURES_BY_LABEL"]) // 4)  # round up, e.g. 14 labels -> 4 rows
fig, axes = plt.subplots(n_rows, 4, figsize=(24, 5 * n_rows), constrained_layout=True)
for ax in axes.flat[len(data["PICTURES_BY_LABEL"]):]:  # empty panels at the end
    ax.set_visible(False)
for ax, (label, paths) in zip(axes.flat, data["PICTURES_BY_LABEL"].items()):
    path = paths[0]
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)[y0:y1, x0:x1]  # ROI only
    ax.imshow(img, extent=(x0, x1, y1, y0))  # axes in px of the full frame
    for x in range(x0, x1 + 1, CROP_SIZE):
        ax.axvline(x, color="yellow", lw=2)
    for y in range(y0, y1 + 1, CROP_SIZE):
        ax.axhline(y, color="yellow", lw=2)
    for k in range(n_tiles):
        row, col = divmod(k, n_cols)  # tile number -> (row, column), e.g. 5 -> (1, 1)
        ax.text(x0 + col * CROP_SIZE + 15, y0 + row * CROP_SIZE + 55, str(k),
                color="yellow", fontsize=15, weight="bold")
    ax.set_title(f"{label} ({image_name(path)})", fontsize=14)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
fig.suptitle(f"ROI (x {x0}-{x1}, y {y0}-{y1}) cut into {n_tiles} tiles of {CROP_SIZE} x {CROP_SIZE} px",
             fontsize=16)

fig_path = RESULTS_DIR / "tiles_on_examples.png"
if fig_path.exists() and not OVERWRITE:
    print(f"Not saved, file exists (set OVERWRITE = True to replace): {fig_path}")
else:
    fig.savefig(fig_path, dpi=100)
    print("Saved", fig_path)
plt.show()

# %% [markdown]
# # Model data
#
# What the models get (CLAUDE.md §9 Step 2), for every image that is not excluded:
# - the ROI cut into 12 tiles of 384 x 384 px (tile models, anomaly detection),
# - the whole ROI (1536 x 1152 px) scaled to 768 x 576 px (whole-picture model).

# %% [markdown]
# ## Tile table
#
# One row per tile, built from `data/manifest.csv`: labels, folds, `test_only` and the excluded flag
# come from there (excluded images get no tiles). `x0, y0, x1, y1` are in px of the full frame.

# %%
manifest = pd.read_csv(MANIFEST_CSV, dtype={"fold": "Int64"})
data["TILES"] = build_tile_table(manifest, ROI, CROP_SIZE)
print(f"{len(data['TILES'])} tiles from {data['TILES']['file'].nunique()} images")
data["TILES"].head(12)  # the 12 tiles of the first image

# %%
# Check: tiles per label and fold (= 12 x the images used per label and fold in the manifest check)
data["TILES"].assign(
    fold_name=lambda d: ("fold " + d["fold"].astype("string")).fillna("test only")
).pivot_table(index="label", columns="fold_name", values="file", aggfunc="size", fill_value=0)

# %% [markdown]
# ## ROI and tiles of one image
#
# What the models get from one image (`EXAMPLE_IMAGE`): the ROI (1536 x 1152 px) cut into 12 tiles
# of 384 x 384 px at full resolution, and the whole ROI scaled to 768 x 576 px (half size).

# %%
example_rows = data["TILES"][data["TILES"]["file"] == EXAMPLE_IMAGE]
example_path = Path(example_rows["path"].iloc[0])
roi_img = load_roi(example_path, ROI)
tiles = cut_tiles(roi_img, CROP_SIZE)
whole = shrink_roi(roi_img, WHOLE_SIZE)

# Check: every tile is exactly the image cut at the coordinates in the tile table
full_img = cv2.cvtColor(cv2.imread(str(example_path)), cv2.COLOR_BGR2RGB)
tiles_match = all(
    np.array_equal(tiles[r.tile_idx], full_img[r.y0:r.y1, r.x0:r.x1]) for r in example_rows.itertuples()
)
print(f"ROI {roi_img.shape}, tiles {tiles.shape}, whole picture {whole.shape}")
print("Tiles match the tile-table coordinates:", tiles_match)

# %%
fig = plt.figure(figsize=(24, 8.5), constrained_layout=True)
fig_tiles, fig_whole = fig.subfigures(1, 2, width_ratios=[1, 1])

# Left: the 12 tiles, as the tile models see them (384 x 384 px each)
for r, ax in zip(example_rows.itertuples(), fig_tiles.subplots(3, 4).flat):
    ax.imshow(tiles[r.tile_idx])
    ax.set_title(f"tile {r.tile_idx}  (x {r.x0}-{r.x1}, y {r.y0}-{r.y1})", fontsize=9)
    ax.tick_params(labelsize=7)
fig_tiles.suptitle(f"12 tiles of {CROP_SIZE} x {CROP_SIZE} px (full resolution), axes in px of the tile")

# Right: the whole ROI scaled to WHOLE_SIZE, as the whole-picture model sees it
ax = fig_whole.subplots()
ax.imshow(whole)
ax.set(xlabel="x [px]", ylabel="y [px]")
fig_whole.suptitle(f"Whole ROI scaled to {WHOLE_SIZE[0]} x {WHOLE_SIZE[1]} px (half size)")
fig.suptitle(f"What the models get from {EXAMPLE_IMAGE} ({example_rows['label'].iloc[0]})", fontsize=15)

fig_path = RESULTS_DIR / "model_inputs_example.png"
if fig_path.exists() and not OVERWRITE:
    print(f"Not saved, file exists (set OVERWRITE = True to replace): {fig_path}")
else:
    fig.savefig(fig_path, dpi=100)
    print("Saved", fig_path)
plt.show()

# %% [markdown]
# ## Augmentation
#
# Used only when training (never on validation or test data): flips and rotations by 90, 180 and
# 270 deg. They only move pixels, so no detail is blurred and no colour changes; the label stays true.
# A tile has 8 versions (`AUG_TILE`); the 768 x 576 whole picture has 4 (`AUG_WHOLE`), because a
# 90 deg rotation would turn it into 576 x 768.

# %%
aug_tile = tiles[AUG_EXAMPLE_TILE]  # tiles and whole of EXAMPLE_IMAGE, from the cell above
tile_versions = [augment(aug_tile, k, flip) for k, flip in AUG_TILE]
whole_versions = [augment(whole, k, flip) for k, flip in AUG_WHOLE]

# Check: every version has exactly the same pixel values (only their positions change),
# and all versions are different from each other
same_pixels = all(
    np.array_equal(np.sort(v, axis=None), np.sort(orig, axis=None))
    for orig, versions in [(aug_tile, tile_versions), (whole, whole_versions)]
    for v in versions
)
n_distinct = len({v.tobytes() for v in tile_versions}), len({v.tobytes() for v in whole_versions})
print("Tile versions:", [v.shape for v in tile_versions][0], "x", len(tile_versions),
      "| whole versions:", [v.shape for v in whole_versions][0], "x", len(whole_versions))
print("Same pixel values in every version:", same_pixels)
print(f"Distinct versions: tile {n_distinct[0]} of 8, whole {n_distinct[1]} of 4")

# %%
fig = plt.figure(figsize=(24, 9), constrained_layout=True)
fig_t, fig_w = fig.subfigures(1, 2, width_ratios=[1, 1.1])

# Left: the 8 versions of one tile
for ax, (k, flip), v in zip(fig_t.subplots(2, 4).flat, AUG_TILE, tile_versions):
    ax.imshow(v)
    ax.set_title(augment_name(k, flip), fontsize=12)
    ax.tick_params(labelsize=7)
fig_t.suptitle(f"Tile {AUG_EXAMPLE_TILE}: 8 versions ({CROP_SIZE} x {CROP_SIZE} px)")

# Right: the 4 versions of the whole picture
for ax, (k, flip), v in zip(fig_w.subplots(2, 2).flat, AUG_WHOLE, whole_versions):
    ax.imshow(v)
    ax.set_title(augment_name(k, flip), fontsize=12)
    ax.tick_params(labelsize=7)
fig_w.suptitle(f"Whole picture: 4 versions ({WHOLE_SIZE[0]} x {WHOLE_SIZE[1]} px)")
fig.suptitle(f"Augmentation of {EXAMPLE_IMAGE} ({example_rows['label'].iloc[0]})", fontsize=15)

fig_path = RESULTS_DIR / "augmentation_example.png"
if fig_path.exists() and not OVERWRITE:
    print(f"Not saved, file exists (set OVERWRITE = True to replace): {fig_path}")
else:
    fig.savefig(fig_path, dpi=100)
    print("Saved", fig_path)
plt.show()

# %% [markdown]
# ## Speed test
#
# How long it takes to make the model inputs, to decide whether a cache on disk is needed.
# Timed on `SPEED_TEST_IMAGES` random images (one CPU core). Reading a file the first time is slower
# than reading it again (the operating system keeps recently read files in RAM), so repeated runs
# can be faster.

# %%
rng = np.random.default_rng(SEED)
images_used = data["TILES"].drop_duplicates("file")  # one row per image that is not excluded
speed_sample = images_used.iloc[sorted(rng.choice(len(images_used), SPEED_TEST_IMAGES, replace=False))]
load_times = load_or_compute(
    CACHE_DIR / "load_times.pkl",
    lambda: measure_load_times([Path(p) for p in speed_sample["path"]], ROI, CROP_SIZE, WHOLE_SIZE),
    USE_SAVED_RESULTS,
)
load_times.agg(["mean", "median", "max"]).T.round(1).rename_axis("ms per image")

# %%
# What this means for one pass over all images (e.g. one training epoch)
ms = load_times.mean()
n_images = len(images_used)  # 3966
load_options = pd.DataFrame(
    {
        "ms per image": [
            ms["read_roi"] + ms["cut_tiles"],  # 12 tiles, cut from the full PNG each time
            ms["read_roi"] + ms["shrink"],      # whole picture, scaled from the full PNG each time
            ms["decode_tile_pngs"],             # 12 tiles read from a tile cache
            ms["decode_whole_png"],             # whole picture read from a 768 x 576 cache
        ],
        "cache on disk": ["none", "none", "about 21 GB", "about 1-1.5 GB"],
    },
    index=[
        "12 tiles, on the fly",
        "whole picture, on the fly",
        "12 tiles, from cache",
        "whole picture, from cache",
    ],
)
load_options["images/s (1 core)"] = 1000 / load_options["ms per image"]
load_options[f"images/s ({DATA_WORKERS} workers)"] = load_options["images/s (1 core)"] * DATA_WORKERS
load_options[f"1 pass, {DATA_WORKERS} workers [s]"] = n_images / load_options[f"images/s ({DATA_WORKERS} workers)"]
load_options.round(1)

# %% [markdown]
# # Evaluation
#
# The same exam for every method (CLAUDE.md §9 Step 6):
# - 4 runs: test on fold k, choose the percentile threshold and settings on the validation fold
#   (the next fold, cyclic), train on the other two folds. F and U are tested in every run.
# - Every score is turned into a percentile of the good validation images; an image is flagged as
#   defective above the 95th percentile (about 5 % false alarms on good).
# - Metrics per image: AUROC, balanced accuracy, HAR, recall per class, false alarms on good,
#   detection table; supervised methods also the defect-type confusion matrix.

# %% [markdown]
# ## Fold plan
#
# Which preparations train, validate and test in each of the 4 runs.

# %%
preps = data["MANIFEST"].drop_duplicates("prep_id").set_index("prep_id")["fold"]  # prep -> fold
fold_plan = pd.DataFrame({
    f"run {k}": run_roles(preps, k).groupby(run_roles(preps, k)).apply(lambda s: ", ".join(s.index))
    for k in FOLDS
}).T[["train", "validation", "test", "test_only"]]
fold_plan

# %%
# Images used (not excluded) per role in every run, and checks
images = data["MANIFEST"][~data["MANIFEST"]["excluded"]]
image_roles = pd.DataFrame({f"run {k}": run_roles(images["fold"], k) for k in FOLDS})
display(pd.DataFrame({
    run: image_roles[run].value_counts() for run in image_roles
}).T[["train", "validation", "test", "test_only"]].assign(
    good_in_validation=[((image_roles[run] == "validation") & (images["label"] == "good")).sum()
                        for run in image_roles]
))

in_folds = images["fold"].notna()
checks = {
    "every image (A-D) is tested in exactly 1 run": ((image_roles[in_folds] == "test").sum(axis=1) == 1).all(),
    "every image (A-D) is validation in exactly 1 run": ((image_roles[in_folds] == "validation").sum(axis=1) == 1).all(),
    "F, U and M are 'test_only' in every run": (image_roles[~in_folds] == "test_only").all(axis=None),
    "no preparation is in two roles within a run": all(
        images.assign(role=image_roles[run]).groupby("prep_id")["role"].nunique().max() == 1 for run in image_roles
    ),
}
pd.Series(checks, name="ok")

# %% [markdown]
# # Classical methods
#
# Handcrafted features computed on every tile (CLAUDE.md §9 Step 3), then turned into a
# good / defective decision.

# %% [markdown]
# ## Feature definitions
#
# **Notation.** A tile has $N = 384 \times 384$ pixels at positions $(x, y)$. $R, G, B \in [0, 255]$
# are the colour channels and $I(x, y) = 0.299R + 0.587G + 0.114B$ is the grey level (as in OpenCV).
# $\ast$ is convolution (sliding a small weight matrix over the image).
#
# ### 1. Focus (sharpness)
# A sharp tile has strong, sudden changes of brightness at globule edges; blur smooths them away.
#
# **Variance of the Laplacian** (the Laplacian is large where brightness bends sharply):
# $$L = I \ast \begin{pmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{pmatrix}, \qquad
# \mathrm{VoL} = \frac{1}{N} \sum_{x,y} \bigl(L(x,y) - \bar{L}\bigr)^2$$
#
# **Tenengrad** (energy of the Sobel gradient):
# $$I_x = I \ast S, \quad I_y = I \ast S^{\top}, \quad
# S = \begin{pmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{pmatrix}, \qquad
# \mathrm{TEN} = \frac{1}{N} \sum_{x,y} \bigl(I_x^2 + I_y^2\bigr)$$
#
# **Brenner** (squared difference of pixels 2 apart, in both directions):
# $$\mathrm{BRE} = \frac{1}{N} \sum_{x,y} \Bigl[\bigl(I(x{+}2, y) - I(x, y)\bigr)^2
# + \bigl(I(x, y{+}2) - I(x, y)\bigr)^2\Bigr]$$
#
# All three drop when the tile is out of focus. They also grow with the number of globules
# (more edges), so they are only compared with good tiles, never used as absolute values.
#
# ### 2. Grey-level statistics
# **Entropy** of the grey-level histogram, with $p_k$ = share of pixels with grey level $k$:
# $$H = -\sum_{k=0}^{255} p_k \log_2 p_k \quad [\text{bits}]$$
# Low for flat, uniform tiles (haze, empty field), high for detailed ones.
#
# **Mean and standard deviation** of brightness:
# $$\mu = \frac{1}{N} \sum_{x,y} I(x,y), \qquad
# \sigma = \sqrt{\frac{1}{N} \sum_{x,y} \bigl(I(x,y) - \mu\bigr)^2}$$
#
# **RMS contrast**, relative to brightness: $C_{\mathrm{rms}} = \sigma / \mu$.
# (The classic RMS contrast is $\sigma$ of intensities scaled to $[0, 1]$, i.e. $\sigma / 255$,
# which would only repeat $\sigma$.)
#
# ### 3. Colour statistics
# **Mean channels** $\bar{R}, \bar{G}, \bar{B}$ and **chromaticities** (share of each channel,
# independent of brightness):
# $$r = \frac{\bar{R}}{\bar{R} + \bar{G} + \bar{B}}, \quad
# g = \frac{\bar{G}}{\bar{R} + \bar{G} + \bar{B}}, \quad
# b = \frac{\bar{B}}{\bar{R} + \bar{G} + \bar{B}} \quad (r + g + b = 1)$$
# plus the **ratio** $\bar{B} / \bar{R}$ (yellow filter: much below 1, blue filter: much above 1).
#
# **HSV**: mean saturation $\bar{S}$ and mean value $\bar{V}$. Hue $h$ is an angle, so its mean is
# circular, weighted by saturation (grey pixels have no meaningful hue):
# $$\bar{h} = \operatorname{atan2}\Bigl(\sum_{x,y} S \sin h, \; \sum_{x,y} S \cos h\Bigr)$$
# For the models, $\bar{h}$ is given as $(\cos \bar{h}, \sin \bar{h})$ so that 359° and 1° are close.
#
# ### 4. Globule area fraction (threshold segmentation)
# Globules differ from the smooth background, so each pixel's distance from the local background is
# thresholded:
# $$D = \bigl|I - I \ast \mathcal{G}_{\sigma}\bigr|, \qquad
# M(x,y) = \begin{cases} 1 & D(x,y) > t \\ 0 & \text{otherwise} \end{cases}, \qquad
# A = \frac{1}{N} \sum_{x,y} M(x,y)$$
# $\mathcal{G}_{\sigma}$ is a wide Gaussian blur (the local background, e.g. $\sigma = 15$ px) and $t$
# is **one fixed threshold** for all tiles, found with Otsu's method on good training tiles
# (a threshold per tile would always find "globules", even in an empty tile).
# Undiluted: $A$ high; overdiluted: $A$ low. The segmentation is checked visually before use.
#
# ### 5. Histograms
# Each histogram is divided by $N$ (sums to 1), so it describes the shape of the distribution:
# $$h_k = \frac{1}{N} \, \#\bigl\{(x, y) : \text{value}(x, y) \in \text{bin}_k\bigr\}$$
# - **Brightness** $I$: 16 bins of width 16.
# - **Hue** (each pixel weighted by its saturation, 18 bins of 20°) and **saturation** (16 bins).
# - **Gradient magnitude** $|\nabla I| = \sqrt{I_x^2 + I_y^2}$: 16 bins of $\log(1 + |\nabla I|)$,
#   because the distribution has a long tail; a sharp tile has more pixels in the high bins.
# - **LBP texture** (optional): each pixel gets a code from comparing it with its 8 neighbours:
# $$\mathrm{LBP} = \sum_{p=0}^{7} s(g_p - g_c) \, 2^p, \qquad
# s(u) = \begin{cases} 1 & u \ge 0 \\ 0 & u < 0 \end{cases}$$
#   ($g_c$ = centre pixel, $g_p$ = neighbours). Rotation-invariant "uniform" codes give 10 bins.
#
# In total about 85 numbers per tile (about 95 with LBP).
#
# ### 6. From features to a decision
# - **Scaling**: for linear and distance-based models every feature is standardised with the
#   training preparations only: $z = (f - \mu_{\text{train}}) / \sigma_{\text{train}}$.
#   Tree models (random forest, gradient boosting) do not need it.
# - **Single feature (mode a)**: a defect can move a feature either way (blur lowers VoL, undiluted
#   raises it), so a tile is flagged when the feature leaves the band of good validation tiles:
#   $f < q_{2.5\%}$ or $f > q_{97.5\%}$, which gives about 5 % false alarms on good by design.
# - **Image score** from the 12 tile scores $s_1, \dots, s_{12}$:
#   $s_{\text{img}} = \max_t s_t$ (catches local defects such as bubbles) or
#   $s_{\text{img}} = \frac{1}{12} \sum_t s_t$ (steadier for whole-image defects).

# %% [markdown]
# ## Features of example tiles
#
# The features computed on one tile per label (tile `FEATURE_EXAMPLE_TILE` of the first image of
# each label), to check them visually before computing them for all 47,592 tiles.
# The globule threshold $t$ is found once with Otsu's method on all good tiles (like the ROI, which
# used all good images) and then used for every tile; the whole pictures (half resolution) get
# their own threshold, found the same way on all good whole pictures.

# %%
# Globule thresholds t: Otsu's method on D of ALL good tiles, and separately of all good whole pictures
good_paths = [Path(p) for p in data["TILES"].loc[data["TILES"]["label"] == "good", "path"].unique()]
counts = load_or_compute(
    CACHE_DIR / "globule_deviation_counts.pkl",
    lambda: run_parallel(
        partial(deviation_counts, roi=ROI, tile=CROP_SIZE, whole_size=WHOLE_SIZE,
                sigma_tile=GLOBULE_BLUR_SIGMA, sigma_whole=GLOBULE_BLUR_SIGMA_WHOLE),
        good_paths, DATA_WORKERS, desc="D of good images",
    ),
    USE_SAVED_RESULTS,
)
bin_centres = np.arange(256) + 0.5
data["GLOBULE_THRESHOLD"] = float(threshold_otsu(hist=(sum(c[0] for c in counts), bin_centres)))
data["GLOBULE_THRESHOLD_WHOLE"] = float(threshold_otsu(hist=(sum(c[1] for c in counts), bin_centres)))
print(f"Globule threshold t from {len(good_paths)} good images: tiles {data['GLOBULE_THRESHOLD']:.1f}, "
      f"whole pictures {data['GLOBULE_THRESHOLD_WHOLE']:.1f} grey levels")

# %%
# One example tile per label and its features
first_files = data["TILES"].drop_duplicates("label")["file"]  # first image of every label
feature_rows = data["TILES"][
    data["TILES"]["file"].isin(first_files) & (data["TILES"]["tile_idx"] == FEATURE_EXAMPLE_TILE)
]
example_tiles = {r.label: load_roi(Path(r.path), (r.x0, r.y0, r.x1, r.y1)) for r in feature_rows.itertuples()}
example_features = pd.DataFrame({
    label: tile_features(tile, data["GLOBULE_THRESHOLD"], GLOBULE_BLUR_SIGMA)
    for label, tile in example_tiles.items()
})  # rows = features, columns = labels
print(f"{len(example_features)} features per tile")
example_features[~example_features.index.str.startswith("hist_")].round(3)  # the 19 single values

# %%
# What the focus and globule features "see": maps on the same scale for every label
gray_ex = {label: cv2.cvtColor(t, cv2.COLOR_RGB2GRAY).astype(np.float64) for label, t in example_tiles.items()}
lap_ex = {label: np.abs(cv2.Laplacian(g, cv2.CV_64F)) for label, g in gray_ex.items()}
grad_ex = {label: np.hypot(cv2.Sobel(g, cv2.CV_64F, 1, 0), cv2.Sobel(g, cv2.CV_64F, 0, 1))
           for label, g in gray_ex.items()}
lap_max = np.percentile(lap_ex["good"], 99)    # common colour scale, set by the good tile
grad_max = np.percentile(grad_ex["good"], 99)

fig, axes = plt.subplots(len(example_tiles), 4, figsize=(13, 3.3 * len(example_tiles)), constrained_layout=True)
for row, label in zip(axes, example_tiles):
    f = example_features[label]
    mask = globule_deviation(gray_ex[label].astype(np.uint8), GLOBULE_BLUR_SIGMA) > data["GLOBULE_THRESHOLD"]
    overlay = np.dstack([gray_ex[label] / 255] * 3)
    overlay[mask] = 0.6 * overlay[mask] + 0.4 * np.array([0x2A, 0x78, 0xD6]) / 255  # SERIES_COLOUR
    panels = [
        (example_tiles[label], None, f"{label}\ntile {FEATURE_EXAMPLE_TILE}"),
        (lap_ex[label], lap_max, f"|Laplacian|   VoL = {f['vol']:.0f}"),
        (grad_ex[label], grad_max, f"gradient magnitude   TEN = {f['tenengrad']:.0f}"),
        (overlay, None, f"globule mask   A = {f['globule_area']:.3f}"),
    ]
    for ax, (img, vmax, title) in zip(row, panels):
        ax.imshow(img, cmap="Greys" if vmax else None, vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
fig.suptitle("Focus and globule features of one tile per label (maps on the scale of the good tile)",
             fontsize=13)
save_figure(fig, RESULTS_DIR / "feature_maps_example.png", OVERWRITE)
plt.show()

# %%
# Histograms of every label (blue) compared with the good tile (grey line)
hist_types = [  # (feature prefix, title, x-axis label, bin edges)
    ("hist_bright", "brightness", "grey level", np.linspace(0, 256, 17)),
    ("hist_hue", "hue (weighted by saturation)", "hue [deg]", np.linspace(0, 360, 19)),
    ("hist_sat", "saturation", "saturation", np.linspace(0, 256, 17)),
    ("hist_grad", "gradient magnitude", "log(1 + |grad I|)", np.linspace(0, 7.3, 17)),
    ("hist_lbp", "LBP texture codes", "uniform LBP code", np.arange(11) - 0.5),
]
fig, axes = plt.subplots(len(example_tiles), len(hist_types),
                         figsize=(20, 2.2 * len(example_tiles)), constrained_layout=True, sharex="col")
for i, (row, label) in enumerate(zip(axes, example_tiles)):
    for ax, (prefix, title, xlabel, edges) in zip(row, hist_types):
        values = example_features.loc[example_features.index.str.startswith(prefix), label].to_numpy()
        reference = example_features.loc[example_features.index.str.startswith(prefix), "good"].to_numpy()
        centres, width = (edges[:-1] + edges[1:]) / 2, np.diff(edges)
        ax.bar(centres, values, width=0.85 * width, color=SERIES_COLOUR, label="this label")
        ax.step(edges, np.r_[reference, reference[-1]], where="post", color=REFERENCE_COLOUR,
                lw=1.5, label="good (reference)")
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        if i == 0:                  # first row: column titles
            ax.set_title(title, fontsize=11)
        if i == len(axes) - 1:      # last row: x-axis labels
            ax.set_xlabel(xlabel)
    row[0].set_ylabel(f"{label}\nshare of pixels", fontsize=9)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False)
fig.suptitle(f"Histogram features of tile {FEATURE_EXAMPLE_TILE} of the first image of every label",
             fontsize=13)
save_figure(fig, RESULTS_DIR / "histograms_example.png", OVERWRITE)
plt.show()

# %% [markdown]
# ## Features of all images
#
# The 95 features for every tile (12 per image, full resolution) and for every whole picture
# (the ROI scaled to 768 x 576 px), for all images that are not excluded. Saved in `data/features/`
# by image name: only images without saved features are computed (all of them the first time,
# about 16 min in `DATA_WORKERS` parallel processes; later e.g. only newly added images). With
# `USE_SAVED_RESULTS = False` all images are computed again.

# %%
images = data["TILES"].drop_duplicates("file")[["file", "path"]]  # images in use
have_saved = TILE_FEATURES_CSV.exists() and WHOLE_FEATURES_CSV.exists() and USE_SAVED_RESULTS
saved_tiles = pd.read_csv(TILE_FEATURES_CSV) if have_saved else pd.DataFrame(columns=["file"])
saved_whole = pd.read_csv(WHOLE_FEATURES_CSV) if have_saved else pd.DataFrame(columns=["file"])
todo = images[~images["file"].isin(saved_whole["file"])]
print(f"{len(images) - len(todo)} images with saved features, {len(todo)} to compute")
if len(todo):
    t0 = time.time()
    results = run_parallel(
        partial(image_features, roi=ROI, tile=CROP_SIZE, whole_size=WHOLE_SIZE,
                thresholds=(data["GLOBULE_THRESHOLD"], data["GLOBULE_THRESHOLD_WHOLE"]),
                sigmas=(GLOBULE_BLUR_SIGMA, GLOBULE_BLUR_SIGMA_WHOLE)),
        [Path(p) for p in todo["path"]], DATA_WORKERS, desc="features",
    )
    print(f"Computed in {(time.time() - t0) / 60:.1f} min")
    new_tiles = pd.DataFrame([
        {"file": file, "tile_idx": k, **features}
        for file, (tile_list, _) in zip(todo["file"], results)
        for k, features in enumerate(tile_list)
    ])
    new_whole = pd.DataFrame([{"file": file, **whole} for file, (_, whole) in zip(todo["file"], results)])
    saved_tiles = pd.concat([saved_tiles, new_tiles], ignore_index=True) if len(saved_tiles) else new_tiles
    saved_whole = pd.concat([saved_whole, new_whole], ignore_index=True) if len(saved_whole) else new_whole
    save_table(saved_tiles, TILE_FEATURES_CSV, overwrite=True)  # saved features + the new ones
    save_table(saved_whole, WHOLE_FEATURES_CSV, overwrite=True)
# Keep only the images in use (e.g. not images excluded later)
data["TILE_FEATURES"] = saved_tiles[saved_tiles["file"].isin(images["file"])].reset_index(drop=True)
data["WHOLE_FEATURES"] = saved_whole[saved_whole["file"].isin(images["file"])].reset_index(drop=True)
print(f"Tiles: {data['TILE_FEATURES'].shape}, whole pictures: {data['WHOLE_FEATURES'].shape}")

# %%
# Median of every single-value feature per label, over all tiles and over all whole pictures
label_of_file = data["TILES"].drop_duplicates("file").set_index("file")["label"]
label_order = list(data["PICTURES_BY_LABEL"])  # good, blur_light, ..., frozen, uht, milk32_undiluted, milk32
single_features = [c for c in data["WHOLE_FEATURES"].columns if c != "file" and not c.startswith("hist_")]
feature_medians = {
    name: table.assign(label=table["file"].map(label_of_file))
               .groupby("label")[single_features].median().T[label_order]
    for name, table in [("tiles", data["TILE_FEATURES"]), ("whole pictures", data["WHOLE_FEATURES"])]
}
print("Median over all TILES of each label:")
display(feature_medians["tiles"].round(3))
print("Median over all WHOLE PICTURES (768 x 576) of each label:")
feature_medians["whole pictures"].round(3)

# %%
# Box plots of the 19 single-value features per label, over all tiles
fig = plot_feature_boxplots(data["TILE_FEATURES"], label_of_file, single_features, label_order,
                            "Single-value features over all TILES of every label (384 x 384 px)")
save_figure(fig, RESULTS_DIR / "boxplots_all_tiles.png", OVERWRITE)
plt.show()

# %%
# Box plots of the 19 single-value features per label, over all whole pictures
fig = plot_feature_boxplots(data["WHOLE_FEATURES"], label_of_file, single_features, label_order,
                            "Single-value features over all WHOLE PICTURES of every label (768 x 576 px)")
save_figure(fig, RESULTS_DIR / "boxplots_all_whole.png", OVERWRITE)
plt.show()

# %%
# Average histograms per label over all tiles, compared with good
hist_columns = [c for c in data["TILE_FEATURES"].columns if c.startswith("hist_")]
mean_hists_tiles = (data["TILE_FEATURES"].assign(label=data["TILE_FEATURES"]["file"].map(label_of_file))
                    .groupby("label")[hist_columns].mean().T[label_order])
fig = plot_label_histograms(mean_hists_tiles, "Average histograms over all TILES of every label (384 x 384 px)")
save_figure(fig, RESULTS_DIR / "histograms_all_tiles.png", OVERWRITE)
plt.show()

# %%
# Average histograms per label over all whole pictures, compared with good
mean_hists_whole = (data["WHOLE_FEATURES"].assign(label=data["WHOLE_FEATURES"]["file"].map(label_of_file))
                    .groupby("label")[hist_columns].mean().T[label_order])
fig = plot_label_histograms(mean_hists_whole,
                            "Average histograms over all WHOLE PICTURES of every label (768 x 576 px)")
save_figure(fig, RESULTS_DIR / "histograms_all_whole.png", OVERWRITE)
plt.show()

# %% [markdown]
# ## Plain variance of the Laplacian
#
# The evaluation (section `# Evaluation`) tried on the simplest possible detector, before it is used
# for all methods: **"an image is defective if its variance of the Laplacian (VoL, sharpness of the
# whole picture) is unusual compared with good images"**. Nothing is trained. In every run:
# 1. the VoL of the good images of the validation fold shows what "normal" is;
# 2. a test image is flagged as defective if its VoL is below the 2.5th or above the 97.5th
#    percentile of them (two-sided: blur lowers VoL, undiluted raises it);
# 3. the decisions are compared with the true labels.
#
# Result: the metrics per run (table) and the figure (ROC curves, confusion matrix, flagged per label).

# %%
eval_info = data["MANIFEST"].loc[~data["MANIFEST"]["excluded"]].set_index("file")[["fold", "label", "is_defect"]]
eval_labels = list(data["PICTURES_BY_LABEL"])  # good, blur_light, ..., frozen, uht, milk32_undiluted, milk32
demo_vol = data["WHOLE_FEATURES"].set_index("file")["vol"]  # VoL of every whole picture (cell above)
demo_metrics, demo_predictions = evaluate_method(
    pd.DataFrame({run: demo_vol for run in FOLDS}), eval_info, two_sided=True
)
demo_summary = pd.concat([demo_metrics, demo_metrics.agg(["mean", "std"])])
demo_summary.index = [f"run {run}" for run in demo_metrics.index] + ["mean", "SD"]
demo_summary.round(3)

# %%
fig = plot_evaluation(demo_metrics, demo_predictions, eval_labels,
                      "Plain variance of the Laplacian (whole picture, two-sided)")
save_figure(fig, RESULTS_DIR / "evaluation_example_vol.png", OVERWRITE)
plt.show()

# %% [markdown]
# ## Single features
#
# Every one of the 19 single-value features used alone as a detector, with the same evaluation as
# the plain variance of the Laplacian (4 runs, 95th-percentile cut on the good validation images).
# Each feature is tried on three inputs:
# - **whole**: the feature of the whole picture (768 x 576), two-sided;
# - **tiles max**: every tile is scored against the good validation tiles (two-sided, 0-100), and the
#   image gets the score of its most unusual tile;
# - **tiles mean**: the same, but the image gets the mean of its 12 tile scores.
#
# For example, with VoL and "tiles max", a bubbles image with one very sharp tile (crowded globules at
# the bubble edge) gets that tile's high score, even if its other 11 tiles look normal.
#
# Result: a table of the best feature/input pairs and a heat map of the % flagged per label for all
# 57 combinations (good column = false alarms, other columns = defects detected).

# %%
single_features = [c for c in data["WHOLE_FEATURES"].columns if c != "file" and not c.startswith("hist_")]
whole_by_file = data["WHOLE_FEATURES"].set_index("file")
single_rows = []
for feature in tqdm(single_features, desc="single features"):
    inputs = {"whole": (pd.DataFrame({run: whole_by_file[feature] for run in FOLDS}), True)}
    for agg in ("max", "mean"):
        inputs[f"tiles {agg}"] = (tile_image_scores(data["TILE_FEATURES"], feature, eval_info, agg), False)
    for input_name, (scores, two_sided) in inputs.items():
        metrics, predictions = evaluate_method(scores, eval_info, two_sided)
        flagged_pct = detection_table(predictions, eval_labels)["flagged_pct_mean"]
        single_rows.append({
            "method": f"{feature} | {input_name}", "feature": feature, "input": input_name,
            "auroc": metrics["auroc"].mean(), "auroc_sd": metrics["auroc"].std(),
            "balanced_accuracy": metrics["balanced_accuracy"].mean(), "har": metrics["har"].mean(),
            "false_alarms_pct": 100 * metrics["false_alarms_good"].mean(),
            "false_alarms_sd_pct": 100 * metrics["false_alarms_good"].std(),
            **flagged_pct.to_dict(),
        })
data["SINGLE_FEATURES"] = pd.DataFrame(single_rows)
save_table(data["SINGLE_FEATURES"].round(4), RESULTS_DIR / "single_features_summary.csv", OVERWRITE)

print("The 15 best feature/input pairs by AUROC (mean over 4 runs):")
data["SINGLE_FEATURES"].sort_values("auroc", ascending=False).head(15)[
    ["method", "auroc", "auroc_sd", "balanced_accuracy", "har", "false_alarms_pct", "false_alarms_sd_pct"]
].round(3)

# %%
fig = plot_detection_heatmap(data["SINGLE_FEATURES"], eval_labels,
                             "Single features as detectors: whole picture vs tiles (max / mean of 12 tiles)")
save_figure(fig, RESULTS_DIR / "single_features_heatmap.png", OVERWRITE)
plt.show()

# %% [markdown]
# ## One-class models
#
# Four detectors that use **all 95 features at once** and learn **only from good images** of the
# training slides (two folds per run). Default settings, no tuning:
# - **Mahalanobis**: distance of an image from the average good image, taking into account how the
#   features vary together (Ledoit-Wolf: a stabilised version for 95 features and few images);
# - **One-Class SVM**: draws a boundary around the good images; outside = unusual;
# - **Isolation Forest**: random splits of the features; unusual images are isolated in few splits;
# - **LOF**: compares how crowded an image's neighbourhood is with that of its 20 nearest good images.
#
# Each detector gives an anomaly score (higher = less like good). It is tried on the whole picture
# and on tiles (score of the most unusual tile, or the mean of the 12 tiles), and evaluated exactly
# like the single features: 95th-percentile cut on the good validation images, 4 runs.
#
# Result: a comparison table, the heat map of % flagged per label, and the full evaluation figure of
# the best detector.

# %%
one_class_image_scores = load_or_compute(
    CACHE_DIR / f"one_class_scores_{data['KEY']}.pkl",
    lambda: run_one_class_detectors(ONE_CLASS_DETECTORS, data["WHOLE_FEATURES"], data["TILE_FEATURES"], eval_info,
                                    desc="handcrafted one-class"),
    USE_SAVED_RESULTS,
)
one_class_results = {
    key: evaluate_method(scores, eval_info, two_sided=False) for key, scores in one_class_image_scores.items()
}
data["ONE_CLASS"] = pd.DataFrame([
    summary_row(f"{name} | {input_name}", metrics, predictions, eval_labels)
    for (name, input_name), (metrics, predictions) in one_class_results.items()
])
save_table(data["ONE_CLASS"].round(4), RESULTS_DIR / "one_class_summary.csv", OVERWRITE)
data["ONE_CLASS"].sort_values("auroc", ascending=False)[
    ["method", "auroc", "auroc_sd", "balanced_accuracy", "har", "false_alarms_pct", "false_alarms_sd_pct"]
].round(3)

# %%
fig = plot_detection_heatmap(data["ONE_CLASS"], eval_labels,
                             "One-class models on all 95 features (trained on good images only)")
save_figure(fig, RESULTS_DIR / "one_class_heatmap.png", OVERWRITE)
plt.show()

# %%
# Full evaluation figure of the best one-class detector (highest mean AUROC)
best_method = data["ONE_CLASS"].sort_values("auroc", ascending=False)["method"].iloc[0]
best_key = tuple(best_method.split(" | "))
fig = plot_evaluation(*one_class_results[best_key], eval_labels, f"Best one-class model: {best_method}")
save_figure(fig, RESULTS_DIR / "one_class_best.png", OVERWRITE)
plt.show()

# %% [markdown]
# ## Supervised models (good vs defective)
#
# Four classifiers on all 95 features that learn from **good and defective** training images (two
# folds per run), so they can learn what bubbles or undiluted fields look like, which the one-class
# models never see. Class weights balance the ~300 good against ~1,400 defective training images;
# default settings, no tuning:
# - **Logistic regression**: a weighted sum of the (standardised) features;
# - **Random forest**: 300 decision trees that vote;
# - **HistGradientBoosting** and **LightGBM**: trees added one after another, each fixing the
#   mistakes of the previous ones.
#
# For tiles, every tile gets the label of its image (a bubbles image has 12 "defective" tiles even if
# only some show the bubble); the image then gets the max or mean of its 12 tile probabilities.
# Each model gives P(defective); it is judged with the same rule as everything else (95th percentile of
# the good validation images, 4 runs), not with the usual 0.5.
#
# Result: a comparison table, the heat map, the best model's evaluation figure, and the best method of
# each family so far.

# %%
supervised_image_scores = load_or_compute(
    CACHE_DIR / f"supervised_scores_{data['KEY']}.pkl",
    lambda: run_models(SUPERVISED_MODELS, data["WHOLE_FEATURES"], data["TILE_FEATURES"], eval_info,
                       supervised_scores, desc="handcrafted supervised"),
    USE_SAVED_RESULTS,
)
supervised_results = {
    key: evaluate_method(scores, eval_info, two_sided=False) for key, scores in supervised_image_scores.items()
}
data["SUPERVISED"] = pd.DataFrame([
    summary_row(f"{name} | {input_name}", metrics, predictions, eval_labels)
    for (name, input_name), (metrics, predictions) in supervised_results.items()
])
save_table(data["SUPERVISED"].round(4), RESULTS_DIR / "supervised_summary.csv", OVERWRITE)
data["SUPERVISED"].sort_values("auroc", ascending=False)[
    ["method", "auroc", "auroc_sd", "balanced_accuracy", "har", "false_alarms_pct", "false_alarms_sd_pct"]
].round(3)

# %%
fig = plot_detection_heatmap(data["SUPERVISED"], eval_labels,
                             "Supervised models on all 95 features (trained on good and defective images)")
save_figure(fig, RESULTS_DIR / "supervised_heatmap.png", OVERWRITE)
plt.show()

# %%
# Full evaluation figure of the best supervised model (highest mean AUROC)
best_supervised = data["SUPERVISED"].sort_values("auroc", ascending=False)["method"].iloc[0]
fig = plot_evaluation(*supervised_results[tuple(best_supervised.split(" | "))], eval_labels,
                      f"Best supervised model: {best_supervised}")
save_figure(fig, RESULTS_DIR / "supervised_best.png", OVERWRITE)
plt.show()

# %%
# Best method of every family so far (highest mean AUROC), with its detection per label
family_best = pd.concat([
    table.sort_values("auroc", ascending=False).head(1).assign(family=family)
    for family, table in [("single feature", data["SINGLE_FEATURES"]), ("one-class", data["ONE_CLASS"]),
                          ("supervised", data["SUPERVISED"])]
]).set_index("family")
family_best[["method", "auroc", "balanced_accuracy", "har", "false_alarms_pct"] + eval_labels[1:]].round(2)

# %% [markdown]
# ## Supervised models (defect type)
#
# Task T2: not only "good or defective?" but **which problem**. The same four supervised models and
# 95 features, trained on the training folds, now choose **one of 11 labels** (good + the 10 defect
# types); the predicted label is simply the most likely one (no 95 % cut). Inputs: the whole picture;
# **tiles mean** (average of the 12 tiles' label probabilities); **tiles max** (for every label its
# highest probability over the 12 tiles, which helps local defects such as bubbles).
#
# Result: a table (accuracy, mean recall over the 11 labels, HAR, right group = good / acquisition /
# preparation / field position) and the **confusion matrix** of the best model: rows = true label,
# columns = predicted label, summed over the 4 runs. Frozen and UHT were never trained on, so their rows
# (below the line) only show what the model calls them.

# %%
t2_classes = [label for label in eval_labels if label not in TEST_ONLY_LABELS]  # good + 10 defect types
t2_probabilities = load_or_compute(
    CACHE_DIR / f"multiclass_probabilities_{data['KEY']}.pkl",
    lambda: run_multiclass_models(SUPERVISED_MODELS, data["WHOLE_FEATURES"], data["TILE_FEATURES"], eval_info,
                                  t2_classes, desc="handcrafted defect type"),
    USE_SAVED_RESULTS,
)
t2_results = {key: evaluate_multiclass(p, eval_info, t2_classes) for key, p in t2_probabilities.items()}
data["DEFECT_TYPE"] = pd.DataFrame([
    {"method": f"{name} | {input_name}",
     **{m: metrics[m].mean() for m in ["accuracy", "mean_recall", "har", "group_accuracy"]},
     "mean_recall_sd": metrics["mean_recall"].std()}
    for (name, input_name), (metrics, _) in t2_results.items()
])
save_table(data["DEFECT_TYPE"].round(4), RESULTS_DIR / "defect_type_summary.csv", OVERWRITE)
data["DEFECT_TYPE"].sort_values("mean_recall", ascending=False).round(3)

# %%
# Confusion matrix of the best model (highest mean recall over the 11 labels)
best_t2 = data["DEFECT_TYPE"].sort_values("mean_recall", ascending=False)["method"].iloc[0]
fig = plot_confusion_matrix(t2_results[tuple(best_t2.split(" | "))][1], t2_classes, TEST_ONLY_LABELS,
                            f"Defect type, best model: {best_t2}")
save_figure(fig, RESULTS_DIR / "defect_type_confusion.png", OVERWRITE)
plt.show()

# %%
# Recall per label of the best model (mean and SD over the 4 runs)
best_t2_metrics = t2_results[tuple(best_t2.split(" | "))][0]
pd.DataFrame({
    "recall_pct_mean": 100 * best_t2_metrics[[f"recall_{c}" for c in t2_classes]].mean().to_numpy(),
    "recall_pct_sd": 100 * best_t2_metrics[[f"recall_{c}" for c in t2_classes]].std().to_numpy(),
}, index=t2_classes).round(1)

# %% [markdown]
# # Deep features
#
# Pretrained networks turn every tile and every whole picture into 768 numbers ("deep features"),
# much richer than the 95 handcrafted features. The networks are not trained here, they only "look":
# - **ResNet-18** (trained on ImageNet photos): the feature maps of its last two stages, averaged;
# - **DINOv2 ViT-S/14** (self-supervised on 142 million images): its summary token and the average of
#   its patch tokens. It needs sizes divisible by 14, so tiles are centre-cropped to 378 x 378 px and
#   whole pictures to 756 x 574 px (cropping, no resizing).
#
# The features then go into the same one-class and supervised models as the handcrafted features.

# %% [markdown]
# ## Feature extraction

# %%
images_used = data["TILES"].drop_duplicates("file")
deep_features = {  # resumable: finished chunks of 200 images are saved and reused
    name: extract_deep_features([Path(p) for p in images_used["path"]], timm_name, DEEP_BATCH_IMAGES,
                                DATA_WORKERS, DEEP_FEATURES_DIR / f"{name}_parts", USE_SAVED_RESULTS)
    for name, timm_name in DEEP_BACKBONES.items()
}
pd.DataFrame({
    name: {"images": len(f["files"]), "tile features": f["tiles"].shape, "whole-picture features": f["whole"].shape}
    for name, f in deep_features.items()
}).T

# %% [markdown]
# ## One-class and supervised models on deep features
#
# The same one-class models (learn from good images only) and supervised models (learn from good and
# defective images) as for the handcrafted features, with the same evaluation, but now on the deep
# features of every backbone in `DEEP_BACKBONES` (768 numbers per tile or whole picture instead of 95).
# Uncommenting a backbone in Config and running again adds it automatically.
#
# Result: a table of all deep-feature methods, the heat map, and the comparison with the best
# handcrafted-feature methods: **are deep features better than the 95 handcrafted ones?**

# %%
deep_rows = []     # one summary row per method (backbone x model x input)
deep_results = {}  # method name -> (metrics per run, predictions), for the figures below
deep_image_scores = {}  # (backbone, family) -> image scores of all its models (for "Combining methods")
for backbone, features in deep_features.items():  # every backbone in DEEP_BACKBONES
    # Deep features as tables like the handcrafted ones: 768 columns d000-d767, per whole picture / per tile
    whole_table, tile_table = deep_feature_tables(features)
    for family, models, score_fn in [("one-class", ONE_CLASS_DETECTORS, one_class_scores),
                                     ("supervised", SUPERVISED_MODELS, supervised_scores)]:
        # Train every model in all 4 runs on whole pictures and on tiles; saved after the first time.
        # Progress bar: one step = one model trained on one input in one run (4 models x 2 inputs x 4 runs),
        # and next to it what is trained right now, e.g. "Random forest | tiles | run 2 of 4".
        scores = load_or_compute(
            CACHE_DIR / f"deep_{backbone}_{family}_{data['KEY']}.pkl",
            lambda: run_models(models, whole_table, tile_table, eval_info, score_fn, desc=f"{backbone} {family}"),
            USE_SAVED_RESULTS,
        )
        deep_image_scores[(backbone, family)] = scores
        # Same evaluation as everything else: 95th-percentile cut on good validation images, 4 runs
        for (name, input_name), image_scores in scores.items():
            method = f"{backbone} {name} | {input_name}"
            deep_results[method] = evaluate_method(image_scores, eval_info, two_sided=False)
            deep_rows.append({"family": f"deep {family}", **summary_row(method, *deep_results[method], eval_labels)})
data["DEEP"] = pd.DataFrame(deep_rows)
save_table(data["DEEP"].round(4), RESULTS_DIR / "deep_features_summary.csv", OVERWRITE)
data["DEEP"].sort_values("auroc", ascending=False)[
    ["family", "method", "auroc", "auroc_sd", "balanced_accuracy", "har", "false_alarms_pct", "false_alarms_sd_pct"]
].head(15).round(3)

# %%
fig = plot_detection_heatmap(data["DEEP"], eval_labels,
                             "One-class and supervised models on deep features (pretrained networks)")
save_figure(fig, RESULTS_DIR / "deep_features_heatmap.png", OVERWRITE)
plt.show()

# %%
# Best deep-feature method: full evaluation figure
best_deep = data["DEEP"].sort_values("auroc", ascending=False)["method"].iloc[0]
fig = plot_evaluation(*deep_results[best_deep], eval_labels, f"Best deep-feature method: {best_deep}")
save_figure(fig, RESULTS_DIR / "deep_features_best.png", OVERWRITE)
plt.show()

# %%
# Handcrafted vs deep features: the best method of every family (highest mean AUROC)
families = [("handcrafted one-class", data["ONE_CLASS"]), ("handcrafted supervised", data["SUPERVISED"])] + [
    (family, table) for family, table in data["DEEP"].groupby("family")
]
pd.concat([
    table.sort_values("auroc", ascending=False).head(1).assign(family=family) for family, table in families
]).set_index("family")[["method", "auroc", "balanced_accuracy", "har", "false_alarms_pct"] + eval_labels[1:]].round(2)

# %% [markdown]
# # Anomaly detection
#
# Models that learn only what good images look like (CLAUDE.md §9 Step 4) and flag everything that
# differs, also changes they have never seen (e.g. frozen milk). They compare small patches, so they
# also show *where* an image is unusual (anomaly map).

# %% [markdown]
# ## PatchCore
#
# anomalib's PatchCore on the 12 native 384-px tiles of every image (no shrinking):
# 1. a pretrained ResNet-18 describes every tile with a 48 x 48 grid of patches, each with 384 numbers
#    (its layer2 and layer3 feature maps; anomalib averages every patch with its 3 x 3 neighbours);
# 2. **training** (good images of the two training folds only, ~320 per run): a random 2.5 % of the
#    patches of every image is kept (~220,000), and anomalib's coreset selection picks the 10,000 that
#    cover the rest best: the **memory bank**, a catalogue of what good patches look like;
# 3. **scoring**: every patch of a tile looks for its nearest patch in the memory bank; the tile score is
#    the distance of its most unusual patch (slightly reweighted by anomalib); the image score is the
#    highest of its 12 tile scores ("tiles max") or their mean ("tiles mean");
# 4. the same evaluation as every other method: cut at the 95th percentile of the good validation images.
#
# Example: in a bubble image the patches on the bubble edge have no close good patch, so that tile gets
# a high score, the image is flagged and its anomaly map is dark on the bubble.
#
# Only the images a run uses are scored (test fold, test-only images, good validation images), not the
# training images, whose patches are in the memory bank. About 45 min for all 4 runs on the GTX 1650.
# Every run is saved when it is done (memory bank + tile scores, ~16 MB), so after an interruption the
# finished runs are loaded and the next one starts.

# %%
# PatchCore in all 4 runs: memory bank from the good training images, then scores of 12 tiles per image
image_paths = data["TILES"].drop_duplicates("file").set_index("file")["path"]
patchcore_name = PATCHCORE_BACKBONE.split(".")[0]  # "resnet18"
patchcore_runs = {
    run: load_or_compute(CACHE_DIR / f"patchcore_{patchcore_name}_run{run}_{data['KEY']}.pkl",
                         lambda: run_patchcore(eval_info, image_paths, run), USE_SAVED_RESULTS)
    for run in FOLDS
}
pd.DataFrame({run: {"memory bank": r["bank"].shape, "images scored": len(r["tile_scores"])}
              for run, r in patchcore_runs.items()})

# %%
# Image score = highest tile score ("tiles max") or mean of the 12 ("tiles mean"); then the common evaluation
patchcore_results = {}  # method name -> (metrics per run, predictions)
for agg in ("max", "mean"):
    image_scores = pd.DataFrame({run: r["tile_scores"].agg(agg, axis=1) for run, r in patchcore_runs.items()})
    patchcore_results[f"PatchCore {patchcore_name} | tiles {agg}"] = evaluate_method(image_scores, eval_info,
                                                                                    two_sided=False)
data["ANOMALY"] = pd.DataFrame([
    {"family": "anomaly detection", **summary_row(method, *result, eval_labels)}
    for method, result in patchcore_results.items()
])
save_table(data["ANOMALY"].round(4), RESULTS_DIR / "anomaly_summary.csv", OVERWRITE)
data["ANOMALY"].set_index("method")[
    ["auroc", "auroc_sd", "balanced_accuracy", "har", "false_alarms_pct", "false_alarms_sd_pct"] + eval_labels[1:]
].round(2)

# %%
# Best PatchCore variant: full evaluation figure
best_anomaly = data["ANOMALY"].sort_values("auroc", ascending=False)["method"].iloc[0]
fig = plot_evaluation(*patchcore_results[best_anomaly], eval_labels, f"Best anomaly detection: {best_anomaly}")
save_figure(fig, RESULTS_DIR / "anomaly_best.png", OVERWRITE)
plt.show()

# %%
# Anomaly maps: one random image per label, scored by the run in which it is a test image (F, U, M: run 1).
# Colour = patch distance divided by the run's cut (the 95th percentile of the good validation images'
# highest tile score); blue frame = tile above the cut (with "tiles max", one such tile flags the image)
max_predictions = patchcore_results[f"PatchCore {patchcore_name} | tiles max"][1].set_index(["run", "file"])
rng = np.random.default_rng(SEED)
model = make_patchcore()
examples = []
for label in eval_labels:
    files = sorted(eval_info.index[eval_info["label"] == label])
    file = files[rng.integers(len(files))]
    fold = eval_info.loc[file, "fold"]
    run = 1 if pd.isna(fold) else int(fold)
    tile_scores = patchcore_runs[run]["tile_scores"]
    roles = run_roles(eval_info["fold"], run)
    good_val_max = tile_scores.max(axis=1).reindex(eval_info.index[(roles == "validation")
                                                                   & (eval_info["label"] == "good")])
    model.memory_bank = torch.from_numpy(patchcore_runs[run]["bank"]).to(DEVICE)
    roi_img, anomaly_map = patchcore_map(model, Path(image_paths[file]))
    examples.append({
        "label": label, "file": file, "run": run, "roi": roi_img,
        "map": anomaly_map / np.percentile(good_val_max, FLAG_PERCENTILE),
        # the same rule as for the image: a tile is above the cut if its score alone would flag the image
        "tile_flagged": unusualness(tile_scores.loc[file], good_val_max, two_sided=False) > FLAG_PERCENTILE,
        "flagged": bool(max_predictions.loc[(run, file), "flagged"]),
    })
del model
torch.cuda.empty_cache()
fig = plot_anomaly_maps(examples, vmin=0.5, vmax=1.5,
                        title=f"PatchCore ({patchcore_name}, tiles max): one random image per label")
save_figure(fig, RESULTS_DIR / "anomaly_maps_patchcore.png", OVERWRITE)
plt.show()

# %%
# All families: the best method of each (highest mean AUROC)
families = [("handcrafted one-class", data["ONE_CLASS"]), ("handcrafted supervised", data["SUPERVISED"])] + [
    (family, table) for family, table in data["DEEP"].groupby("family")
] + [("anomaly detection", data["ANOMALY"])]
pd.concat([
    table.sort_values("auroc", ascending=False).head(1).assign(family=family) for family, table in families
]).set_index("family")[["method", "auroc", "balanced_accuracy", "har", "false_alarms_pct"] + eval_labels[1:]].round(2)

# %% [markdown]
# ## DFM and PaDiM
#
# Two more anomalib models with the same network (ResNet-18), tiles, runs and evaluation as PatchCore.
# Both learn from the good training images only:
# - **DFM** looks at every tile as a whole: its layer3 feature map averaged down to 6 x 6 cells
#   (9,216 numbers). A PCA learns the main patterns of the good tiles (those that explain 97 % of their
#   variation). Tile score = how badly these patterns rebuild the tile's numbers (reconstruction error).
#   Example: a tile whose cells are all shifted the same way (e.g. a colour cast) is rebuilt badly,
#   even if every single patch would still look normal on its own.
# - **PaDiM** compares patches position by position: for each of the 96 x 96 patch positions of a tile it
#   learns the average and spread (a Gaussian) of 100 feature numbers over all good tiles. Tile score =
#   the largest distance (Mahalanobis) of a patch from the good patches at the same position. Our fields
#   are chosen freely, so a position inside a tile has no fixed content: PaDiM is kept to test whether
#   this per-position idea suits our images.
#
# Image score: highest tile score ("tiles max") or mean of the 12 ("tiles mean"), as for PatchCore.
# About 10 min (DFM) and 15-20 min (PaDiM) for all 4 runs; saved per run (tile scores only).

# %%
# DFM and PaDiM in all 4 runs: fit on the good training images, then scores of 12 tiles per image
anomaly_models = {  # name -> (make the model, fit it on good images, backbone, images per GPU batch)
    "DFM": (make_dfm, fit_dfm, DFM_BACKBONE, DEEP_BATCH_IMAGES),
    "PaDiM": (make_padim, fit_padim, PADIM_BACKBONE, PADIM_BATCH_IMAGES),
}
anomaly_tile_tables = {}  # (name, run) -> tile scores (index = file, columns 0-11)
for name, (make, fit, backbone, batch_images) in anomaly_models.items():
    for run in FOLDS:
        anomaly_tile_tables[(name, run)] = load_or_compute(
            CACHE_DIR / f"{name.lower()}_{backbone.split('.')[0]}_run{run}_{data['KEY']}.pkl",
            lambda: run_anomaly_model(eval_info, image_paths, run, make, fit, batch_images), USE_SAVED_RESULTS)
pd.DataFrame({f"{name} run {run}": {"images scored": len(t)} for (name, run), t in anomaly_tile_tables.items()})

# %%
# All anomaly models (PatchCore above, DFM, PaDiM): image score from the tile scores, then the common evaluation
anomaly_results = dict(patchcore_results)  # method name -> (metrics per run, predictions)
for name, (_, _, backbone, _) in anomaly_models.items():
    for agg in ("max", "mean"):
        image_scores = pd.DataFrame({run: anomaly_tile_tables[(name, run)].agg(agg, axis=1) for run in FOLDS})
        anomaly_results[f"{name} {backbone.split('.')[0]} | tiles {agg}"] = evaluate_method(
            image_scores, eval_info, two_sided=False)
data["ANOMALY_ALL"] = pd.DataFrame([
    {"family": "anomaly detection", **summary_row(method, *result, eval_labels)}
    for method, result in anomaly_results.items()
])
save_table(data["ANOMALY_ALL"].round(4), RESULTS_DIR / "anomaly_all_summary.csv", OVERWRITE)
data["ANOMALY_ALL"].set_index("method")[
    ["auroc", "auroc_sd", "balanced_accuracy", "har", "false_alarms_pct", "false_alarms_sd_pct"] + eval_labels[1:]
].round(2)

# %%
fig = plot_detection_heatmap(data["ANOMALY_ALL"], eval_labels,
                             f"Anomaly detection (anomalib, {patchcore_name}, learned from good tiles only)")
save_figure(fig, RESULTS_DIR / "anomaly_all_heatmap.png", OVERWRITE)
plt.show()

# %%
# Best anomaly model: full evaluation figure
best_anomaly_all = data["ANOMALY_ALL"].sort_values("auroc", ascending=False)["method"].iloc[0]
fig = plot_evaluation(*anomaly_results[best_anomaly_all], eval_labels, f"Best anomaly detection: {best_anomaly_all}")
save_figure(fig, RESULTS_DIR / "anomaly_all_best.png", OVERWRITE)
plt.show()

# %%
# All families: the best method of each (highest mean AUROC), now with all three anomaly models
families = [("handcrafted one-class", data["ONE_CLASS"]), ("handcrafted supervised", data["SUPERVISED"])] + [
    (family, table) for family, table in data["DEEP"].groupby("family")
] + [("anomaly detection", data["ANOMALY_ALL"])]
pd.concat([
    table.sort_values("auroc", ascending=False).head(1).assign(family=family) for family, table in families
]).set_index("family")[["method", "auroc", "balanced_accuracy", "har", "false_alarms_pct"] + eval_labels[1:]].round(2)

# %% [markdown]
# # Combining methods
#
# No single method catches everything. DINOv2 logistic regression catches almost every defect but only
# about half of the frozen milk; DFM catches frozen milk (99 %) but almost no overstained or overdiluted
# fields. Two ways to combine methods, both from the saved scores (nothing is trained again):
# - **Max rule** (the two models in `MAX_RULE_PAIR`): every image gets each model's unusualness (0-100, its
#   percentile among the good validation images, as in every evaluation). The combined score is the higher
#   of the two. Then the usual rule: flagged if the combined score is above the 95th percentile of the good
#   validation images' combined scores, so the false-alarm target on validation stays at 5 %.
#   Example (run 1): the frozen image F02_0059 gets u = 45.5 from logistic regression (passed) and u = 100
#   from DFM, so its combined score is 100. The cut of run 1 is a combined score of 97.3: flagged.
# - **Voting** (the five models in `VOTERS`, the best method of each family): every model decides on its own
#   (flagged if its unusualness > 95), and the image is flagged if at least k of the 5 flag it, for every k
#   in `VOTES_NEEDED` (k = 1: any model is enough; k = 3: majority; k = 5: all five must agree).
#   Example (run 1): 3 of the 5 models flag F02_0059: flagged with k = 1, 2 or 3, passed with k = 4 or 5.
#   For voting, AUROC uses the number of votes (0-5) as the score, so it is the same for every k;
#   k only moves the cut along the same ROC curve.
#
# Result: a table and heat map of the five methods alone, the max rule and every vote, and the full
# evaluation figures of the max rule and of the majority vote.

# %%
# Image scores of every method by its name in the summary tables (e.g. "Mahalanobis | whole"), 4 runs each
method_scores = {f"{name} | {input_name}": s
                 for (name, input_name), s in {**one_class_image_scores, **supervised_image_scores}.items()}
for (backbone, family), scores in deep_image_scores.items():
    method_scores.update({f"{backbone} {name} | {input_name}": s for (name, input_name), s in scores.items()})
for agg in ("max", "mean"):  # anomaly models: image score from the 12 tile scores
    method_scores[f"PatchCore {patchcore_name} | tiles {agg}"] = pd.DataFrame(
        {run: r["tile_scores"].agg(agg, axis=1) for run, r in patchcore_runs.items()})
    for name, (_, _, backbone, _) in anomaly_models.items():
        method_scores[f"{name} {backbone.split('.')[0]} | tiles {agg}"] = pd.DataFrame(
            {run: anomaly_tile_tables[(name, run)].agg(agg, axis=1) for run in FOLDS})

# Unusualness (0-100) of every image in every run, for the models combined below
unusual = {method: run_unusualness(method_scores[method], eval_info) for method in dict.fromkeys(VOTERS + MAX_RULE_PAIR)}

combined_results = {}  # method name -> (metrics per run, predictions)
for method in VOTERS:  # every voter alone, for comparison
    combined_results[method] = evaluate_method(method_scores[method], eval_info, two_sided=False)
# Max rule: the higher of the two unusualness values, then the usual 95th-percentile cut
max_rule = np.maximum(unusual[MAX_RULE_PAIR[0]], unusual[MAX_RULE_PAIR[1]])
combined_results["max rule (2 models)"] = evaluate_method(max_rule, eval_info, two_sided=False)
# Voting: how many voters flag the image (0-5); flagged if at least k of them do
votes = sum((unusual[method] > FLAG_PERCENTILE).astype(float).where(unusual[method].notna()) for method in VOTERS)
for k in VOTES_NEEDED:
    combined_results[f"vote: at least {k} of {len(VOTERS)}"] = evaluate_method(votes, eval_info, two_sided=False,
                                                                               cut=k - 0.5)
data["COMBINED"] = pd.DataFrame([
    {"combination": "max rule" if method.startswith("max") else "voting" if method.startswith("vote") else "alone",
     **summary_row(method, *result, eval_labels)}
    for method, result in combined_results.items()
])
save_table(data["COMBINED"].round(4), RESULTS_DIR / "combined_summary.csv", OVERWRITE)
data["COMBINED"].set_index("method")[
    ["auroc", "auroc_sd", "balanced_accuracy", "har", "false_alarms_pct", "false_alarms_sd_pct"] + eval_labels[1:]
].round(2)

# %%
fig = plot_detection_heatmap(data["COMBINED"], eval_labels,
                             "Combining methods: the five voters alone, the max rule of two, and voting (at least k of 5)")
save_figure(fig, RESULTS_DIR / "combined_heatmap.png", OVERWRITE)
plt.show()

# %%
# Max rule: full evaluation figure
fig = plot_evaluation(*combined_results["max rule (2 models)"], eval_labels,
                      f"Max rule: {MAX_RULE_PAIR[0]} + {MAX_RULE_PAIR[1]}")
save_figure(fig, RESULTS_DIR / "combined_max_rule.png", OVERWRITE)
plt.show()

# %%
# Majority vote: full evaluation figure (ROC curve of the number of votes; the dot is "at least 3 of 5")
majority = len(VOTERS) // 2 + 1
fig = plot_evaluation(*combined_results[f"vote: at least {majority} of {len(VOTERS)}"], eval_labels,
                      f"Majority vote: at least {majority} of {len(VOTERS)} models")
save_figure(fig, RESULTS_DIR / "combined_majority_vote.png", OVERWRITE)
plt.show()

# %% [markdown]
# # CNNs
#
# Networks pretrained on ImageNet and **fine-tuned** on our pictures: all their weights are trained,
# unlike the deep features above, where the network only looks. One network does both tasks: its pooled
# features (512 numbers for ResNet-18) feed two outputs, "defective?" (T1) and "which of the 11 labels?"
# (T2: good + 10 defect types). The training loss is the sum of both, with class weights, so the ~300 good
# pictures count as much as the ~1,400 defective ones, and every label counts the same.

# %% [markdown]
# ## Design A: the whole picture
#
# Input: the whole ROI shrunk to 768 x 576 px (half resolution), in colour.
# - **Training** (every run): the two training folds, good and defective (~1,700 pictures). Every time a
#   picture is read, one of 4 exact versions is picked at random (original, h-flip, v-flip, 180 deg).
#   AdamW, at most 20 epochs, batches of 8 pictures, weights updated every 8 batches (64 pictures).
# - **Kept epoch and early stopping**: after every epoch the loss on the validation fold is computed; the
#   epoch with the lowest validation loss is kept, and training stops when 5 epochs in a row bring no
#   lower one (the test fold is never looked at). Runs 1-3 were trained for all 20 epochs before early
#   stopping was added; their kept epochs (6, 18, 4) are the ones early stopping keeps, so only their
#   training curves are longer.
# - **Evaluation** as for every method: P(defective) -> percentile among the good validation pictures,
#   flagged above 95 (T1); the most likely label (T2).
# - **ResNet-50** is trained exactly the same way. It does not fit batch 8 into the 4 GB GPU, so it
#   recomputes its intermediate results in the backward pass instead of storing them (gradient
#   checkpointing): same gradients, 2.8 GB instead of > 4 GB, about 20% slower.
#
# About 1.7 min per epoch for ResNet-18 and ~6 min for ResNet-50 on the GTX 1650. A checkpoint is saved
# after every epoch (data/cnn: ~180 MB per ResNet-18 run, ~380 MB per ResNet-50 run), so after an
# interruption Run All continues where it stopped.
#
# Result: the kept epochs, the training curves, the T1 table and evaluation figure, the T2 table and
# confusion matrix, Grad-CAM maps, and the best method of every family.

# %%
# Train (or load) every CNN in CNN_BACKBONES in all 4 runs; each run is saved when it is done
cnn_runs = {}  # (backbone, run) -> {"history", "best_epoch", "scores"}
for name, timm_name in CNN_BACKBONES.items():
    for run in FOLDS:
        cnn_runs[(name, run)] = load_or_compute(
            CACHE_DIR / f"cnn_{name}_whole_run{run}_{data['KEY']}.pkl",
            lambda: train_cnn(timm_name, eval_info, image_paths, run, t2_classes,
                              CNN_DIR / f"{name}_whole_run{run}_{data['KEY']}", resume=USE_SAVED_RESULTS,
                              grad_checkpointing=name in CNN_GRAD_CHECKPOINTING),
            USE_SAVED_RESULTS)
pd.DataFrame({f"{name} run {run}": {"epochs trained": len(r["history"]),
                                    "kept epoch": r["best_epoch"],
                                    "validation loss": round(r["history"]["val_loss"].min(), 3),
                                    "training loss then": round(r["history"].loc[r["best_epoch"] - 1, "train_loss"], 3),
                                    "images scored": len(r["scores"])}
              for (name, run), r in cnn_runs.items()})

# %%
# Training curves: is the network still learning, or does the validation loss rise again (overfitting)?
for name in CNN_BACKBONES:
    fig = plot_training_curves({run: cnn_runs[(name, run)] for run in FOLDS}, f"CNN {name}, whole picture")
    save_figure(fig, RESULTS_DIR / f"cnn_{name}_whole_training.png", OVERWRITE)
    plt.show()

# %%
# T1 (good vs defective): P(defective) with the common evaluation (95th percentile of good validation pictures)
cnn_results = {}  # method name -> (metrics per run, predictions)
for name in CNN_BACKBONES:
    image_scores = pd.DataFrame({run: cnn_runs[(name, run)]["scores"]["p_defect"] for run in FOLDS})
    cnn_results[f"CNN {name} | whole"] = evaluate_method(image_scores, eval_info, two_sided=False)
data["CNN"] = pd.DataFrame([
    {"family": "CNN", **summary_row(method, *result, eval_labels)} for method, result in cnn_results.items()
])
save_table(data["CNN"].round(4), RESULTS_DIR / "cnn_summary.csv", OVERWRITE)
data["CNN"].set_index("method")[
    ["auroc", "auroc_sd", "balanced_accuracy", "har", "false_alarms_pct", "false_alarms_sd_pct"] + eval_labels[1:]
].round(2)

# %%
# Full evaluation figure of every CNN
for method, result in cnn_results.items():
    fig = plot_evaluation(*result, eval_labels, method)
    file_name = method.lower().replace("cnn ", "cnn_").replace(" | ", "_")  # "CNN resnet18 | whole" -> cnn_resnet18_whole
    save_figure(fig, RESULTS_DIR / f"{file_name}.png", OVERWRITE)
    plt.show()

# %%
# T2 (defect type): the most likely of the 11 labels; same table as for the handcrafted features
cnn_t2_results = {
    f"CNN {name} | whole": evaluate_multiclass({run: cnn_runs[(name, run)]["scores"][t2_classes] for run in FOLDS},
                                               eval_info, t2_classes)
    for name in CNN_BACKBONES
}
data["CNN_DEFECT_TYPE"] = pd.DataFrame([
    {"method": method,
     **{m: metrics[m].mean() for m in ["accuracy", "mean_recall", "har", "group_accuracy"]},
     "mean_recall_sd": metrics["mean_recall"].std()}
    for method, (metrics, _) in cnn_t2_results.items()
])
save_table(data["CNN_DEFECT_TYPE"].round(4), RESULTS_DIR / "cnn_defect_type_summary.csv", OVERWRITE)
data["CNN_DEFECT_TYPE"].round(3)

# %%
# Confusion matrix of the best CNN (highest mean recall over the 11 labels), and its recall per label
best_cnn_t2 = data["CNN_DEFECT_TYPE"].sort_values("mean_recall", ascending=False)["method"].iloc[0]
fig = plot_confusion_matrix(cnn_t2_results[best_cnn_t2][1], t2_classes, TEST_ONLY_LABELS,
                            f"Defect type, {best_cnn_t2}")
save_figure(fig, RESULTS_DIR / "cnn_defect_type_confusion.png", OVERWRITE)
plt.show()
best_cnn_t2_metrics = cnn_t2_results[best_cnn_t2][0]
pd.DataFrame({
    "recall_pct_mean": 100 * best_cnn_t2_metrics[[f"recall_{c}" for c in t2_classes]].mean().to_numpy(),
    "recall_pct_sd": 100 * best_cnn_t2_metrics[[f"recall_{c}" for c in t2_classes]].std().to_numpy(),
}, index=t2_classes).round(1)

# %%
# Grad-CAM: where does the network look when it says "defective"? One random picture per label (the same
# pictures as in the PatchCore anomaly maps), scored by the run in which it is a test image (F, U, M: run 1).
# Every map is scaled to 0-1 within its picture, so a good picture also shows its "most defect-like" region.
for name, timm_name in CNN_BACKBONES.items():
    predictions = cnn_results[f"CNN {name} | whole"][1].set_index(["run", "file"])
    model = TwoHeadCNN(timm_name, len(t2_classes)).to(DEVICE)
    rng = np.random.default_rng(SEED)
    examples = []
    for label in eval_labels:
        files = sorted(eval_info.index[eval_info["label"] == label])
        file = files[rng.integers(len(files))]
        fold = eval_info.loc[file, "fold"]
        run = 1 if pd.isna(fold) else int(fold)
        model.load_state_dict(torch.load(CNN_DIR / f"{name}_whole_run{run}_{data['KEY']}" / "best.pt",
                                         map_location=DEVICE))
        picture, heat, probabilities = cnn_gradcam(model, Path(image_paths[file]), output=0)
        examples.append({"label": label, "file": file, "run": run, "picture": picture, "map": heat,
                         "p_defect": probabilities[0], "predicted": t2_classes[int(np.argmax(probabilities[1:]))],
                         "flagged": bool(predictions.loc[(run, file), "flagged"])})
    del model
    torch.cuda.empty_cache()
    fig = plot_gradcam(examples, f"Grad-CAM, CNN {name} (whole picture): one random picture per label")
    save_figure(fig, RESULTS_DIR / f"cnn_gradcam_{name}_whole.png", OVERWRITE)
    plt.show()

# %%
# All families: the best method of each (highest mean AUROC), now with the CNNs
families = [("handcrafted one-class", data["ONE_CLASS"]), ("handcrafted supervised", data["SUPERVISED"])] + [
    (family, table) for family, table in data["DEEP"].groupby("family")
] + [("anomaly detection", data["ANOMALY_ALL"]),
      ("combined", data["COMBINED"].loc[data["COMBINED"]["combination"] != "alone"]), ("CNN", data["CNN"])]
pd.concat([
    table.sort_values("auroc", ascending=False).head(1).assign(family=family) for family, table in families
]).set_index("family")[["method", "auroc", "balanced_accuracy", "har", "false_alarms_pct"] + eval_labels[1:]].round(2)

# %% [markdown]
# ## Design B: 12 tiles, "any tile defective"
#
# Input: the 12 tiles of the ROI (4 x 3 tiles of 384 px) at **full resolution** (design A: half). The
# network gives every tile its own P(defective); **a picture's P(defective) is the highest of its 12 tiles**,
# so a picture is defective if any tile is, and the tile scores show which tiles are defective. Only the
# picture's label is used (multiple instance learning: a picture is a bag of tiles); a corner picture does
# not teach "corner" to its 11 normal tiles.
# - **One epoch** (run 1): (1) score all 12 tiles of the 1,734 training pictures (20,808 tiles, no training)
#   and take each picture's **key tile**, the one with the highest P(defective); (2) train on the 1,734 key
#   tiles with their pictures' labels, everything else as in design A (random rotation or flip: 8 versions
#   of a square tile; class weights; updates of 64; learning rate; early stopping, kept epoch). Example: in a
#   corner picture the tile with the coverslip edge scores highest and learns "corner"; in a dense good
#   picture its most defect-like tile learns "good". (3) Validation loss: the key tiles of the 875
#   validation pictures (all 12 tiles scored). Same scheme as Campanella et al. (2019) for pathology slides.
# - **Outputs**: P(defective) of the picture (highest tile) -> the common evaluation (T1); the label of the
#   key tile (T2); P(defective) of all 12 tiles. A tile is **flagged** if its P(defective) is above the
#   picture cut of the run, so a picture is flagged exactly when at least one of its tiles is.
# - Networks in `CNN_TILE_BACKBONES` (ResNet-18 first); ResNet-50 with gradient checkpointing as in design A.
#
# About 3.5 min per epoch for ResNet-18 and ~9 min for ResNet-50 on the GTX 1650.
#
# Result: the kept epochs, training curves, the T1 table of all CNNs (designs A and B) and the evaluation
# figure, the T2 table and confusion matrix, the number of flagged tiles per label, and the pictures with
# their tile scores and Grad-CAM maps.

# %%
# Train (or load) every CNN in CNN_TILE_BACKBONES on tiles (design B) in all 4 runs; each run is saved when done
cnn_tile_runs = {}  # (backbone, run) -> {"history", "best_epoch", "scores", "tile_scores"}
for name in CNN_TILE_BACKBONES:
    timm_name = CNN_BACKBONES[name]
    for run in FOLDS:
        cnn_tile_runs[(name, run)] = load_or_compute(
            CACHE_DIR / f"cnn_{name}_tiles_run{run}_{data['KEY']}.pkl",
            lambda: train_cnn(timm_name, eval_info, image_paths, run, t2_classes,
                              CNN_DIR / f"{name}_tiles_run{run}_{data['KEY']}", resume=USE_SAVED_RESULTS,
                              grad_checkpointing=name in CNN_GRAD_CHECKPOINTING, design="tiles"),
            USE_SAVED_RESULTS)
pd.DataFrame({f"{name} run {run}": {"epochs trained": len(r["history"]),
                                    "kept epoch": r["best_epoch"],
                                    "validation loss": round(r["history"]["val_loss"].min(), 3),
                                    "training loss then": round(r["history"].loc[r["best_epoch"] - 1, "train_loss"], 3),
                                    "images scored": len(r["scores"])}
              for (name, run), r in cnn_tile_runs.items()})

# %%
# Training curves of design B (training loss: on the key tiles; validation loss: key tiles of the validation pictures)
for name in CNN_TILE_BACKBONES:
    fig = plot_training_curves({run: cnn_tile_runs[(name, run)] for run in FOLDS}, f"CNN {name}, 12 tiles")
    save_figure(fig, RESULTS_DIR / f"cnn_{name}_tiles_training.png", OVERWRITE)
    plt.show()

# %%
# T1 (good vs defective): picture P(defective) = highest tile; all CNNs (designs A and B) in one table
cnn_tile_results = {}  # method name -> (metrics per run, predictions)
for name in CNN_TILE_BACKBONES:
    image_scores = pd.DataFrame({run: cnn_tile_runs[(name, run)]["scores"]["p_defect"] for run in FOLDS})
    cnn_tile_results[f"CNN {name} | tiles"] = evaluate_method(image_scores, eval_info, two_sided=False)
data["CNN_ALL"] = pd.DataFrame([
    {"family": "CNN", **summary_row(method, *result, eval_labels)}
    for method, result in {**cnn_results, **cnn_tile_results}.items()
])
save_table(data["CNN_ALL"].round(4), RESULTS_DIR / "cnn_all_summary.csv", OVERWRITE)
data["CNN_ALL"].set_index("method")[
    ["auroc", "auroc_sd", "balanced_accuracy", "har", "false_alarms_pct", "false_alarms_sd_pct"] + eval_labels[1:]
].round(2)

# %%
# Full evaluation figure of every design-B CNN
for method, result in cnn_tile_results.items():
    fig = plot_evaluation(*result, eval_labels, method)
    file_name = method.lower().replace("cnn ", "cnn_").replace(" | ", "_")  # "CNN resnet18 | tiles" -> cnn_resnet18_tiles
    save_figure(fig, RESULTS_DIR / f"{file_name}.png", OVERWRITE)
    plt.show()

# %%
# T2 (defect type): the label of the key tile; all CNNs (designs A and B) in one table
cnn_tile_t2_results = {
    f"CNN {name} | tiles": evaluate_multiclass(
        {run: cnn_tile_runs[(name, run)]["scores"][t2_classes] for run in FOLDS}, eval_info, t2_classes)
    for name in CNN_TILE_BACKBONES
}
cnn_all_t2_results = {**cnn_t2_results, **cnn_tile_t2_results}
data["CNN_ALL_DEFECT_TYPE"] = pd.DataFrame([
    {"method": method,
     **{m: metrics[m].mean() for m in ["accuracy", "mean_recall", "har", "group_accuracy"]},
     "mean_recall_sd": metrics["mean_recall"].std()}
    for method, (metrics, _) in cnn_all_t2_results.items()
])
save_table(data["CNN_ALL_DEFECT_TYPE"].round(4), RESULTS_DIR / "cnn_all_defect_type_summary.csv", OVERWRITE)
data["CNN_ALL_DEFECT_TYPE"].round(3)

# %%
# Confusion matrix of the best CNN of both designs (highest mean recall over the 11 labels), and its recall per label
best_cnn_all_t2 = data["CNN_ALL_DEFECT_TYPE"].sort_values("mean_recall", ascending=False)["method"].iloc[0]
fig = plot_confusion_matrix(cnn_all_t2_results[best_cnn_all_t2][1], t2_classes, TEST_ONLY_LABELS,
                            f"Defect type, {best_cnn_all_t2}")
save_figure(fig, RESULTS_DIR / "cnn_all_defect_type_confusion.png", OVERWRITE)
plt.show()
best_cnn_all_t2_metrics = cnn_all_t2_results[best_cnn_all_t2][0]
pd.DataFrame({
    "recall_pct_mean": 100 * best_cnn_all_t2_metrics[[f"recall_{c}" for c in t2_classes]].mean().to_numpy(),
    "recall_pct_sd": 100 * best_cnn_all_t2_metrics[[f"recall_{c}" for c in t2_classes]].std().to_numpy(),
}, index=t2_classes).round(1)

# %%
# Which tiles are flagged? For every flagged test (and test-only) picture: how many of its 12 tiles are
# above the cut. Defects of the whole field (blur, filter, dilution) should flag many tiles; bubbles and
# corners only a few.
flagged_tile_tables = {}
for name in CNN_TILE_BACKBONES:
    predictions = cnn_tile_results[f"CNN {name} | tiles"][1]
    counts = []
    for run in FOLDS:
        result = cnn_tile_runs[(name, run)]
        flags = tile_flags(result["tile_scores"], result["scores"]["p_defect"], eval_info, run)
        flagged = predictions[(predictions["run"] == run) & predictions["flagged"]]
        counts.append(pd.DataFrame({"label": flagged["label"].to_numpy(),
                                    "tiles": flags.loc[flagged["file"]].sum(axis=1).to_numpy()}))
    counts = pd.concat(counts)
    print(f"{name}: every flagged picture has at least one flagged tile: {bool((counts['tiles'] >= 1).all())}")
    flagged_tile_tables[name] = counts.groupby("label")["tiles"].agg(
        flagged_pictures="size", mean_flagged_tiles="mean",
        pct_1_to_3_tiles=lambda t: 100 * (t <= 3).mean(), pct_all_12_tiles=lambda t: 100 * (t == 12).mean(),
    ).reindex(eval_labels)
data["CNN_FLAGGED_TILES"] = pd.concat(flagged_tile_tables, names=["network", "label"]).round(1)
save_table(data["CNN_FLAGGED_TILES"].reset_index(), RESULTS_DIR / "cnn_tiles_flagged_tiles.csv", OVERWRITE)
data["CNN_FLAGGED_TILES"]

# %%
# The same random pictures as in the design-A Grad-CAM figure: each with its 12 tile scores (red frame =
# flagged tile) and a Grad-CAM map put together from the tiles, each tile's map weighted by its P(defective)
for name in CNN_TILE_BACKBONES:
    predictions = cnn_tile_results[f"CNN {name} | tiles"][1].set_index(["run", "file"])
    model = TwoHeadCNN(CNN_BACKBONES[name], len(t2_classes)).to(DEVICE)
    rng = np.random.default_rng(SEED)
    examples = []
    for label in eval_labels:
        files = sorted(eval_info.index[eval_info["label"] == label])
        file = files[rng.integers(len(files))]
        fold = eval_info.loc[file, "fold"]
        run = 1 if pd.isna(fold) else int(fold)
        result = cnn_tile_runs[(name, run)]
        model.load_state_dict(torch.load(CNN_DIR / f"{name}_tiles_run{run}_{data['KEY']}" / "best.pt",
                                         map_location=DEVICE))
        picture, heat, tile_p, probabilities = cnn_tile_gradcam(model, Path(image_paths[file]))
        flags = tile_flags(result["tile_scores"], result["scores"]["p_defect"], eval_info, run).loc[file]
        examples.append({"label": label, "file": file, "run": run, "picture": picture, "map": heat,
                         "p_defect": probabilities[0], "predicted": t2_classes[int(np.argmax(probabilities[1:]))],
                         "flagged": bool(predictions.loc[(run, file), "flagged"]),
                         "tile_p": tile_p, "tile_flagged": flags.to_numpy()})
    del model
    torch.cuda.empty_cache()
    fig = plot_gradcam(examples, f"CNN {name}, 12 tiles: P(defective) per tile (red frame = flagged) and Grad-CAM",
                       map_label="Grad-CAM of each tile x its P(defective) (0-1 within the picture)")
    save_figure(fig, RESULTS_DIR / f"cnn_gradcam_{name}_tiles.png", OVERWRITE)
    plt.show()

# %% [markdown]
# # Benchmark
#
# One table of all methods, chosen and ranked by **balanced accuracy**: the mean of the detection rate on
# defective pictures and the pass rate on good pictures, at the cut each method really uses (95th percentile
# of the good validation pictures). AUROC says how well a score could separate good from defective with the
# best possible cut; balanced accuracy says how well the method does as a QC with its actual cut.
# Example: ResNet-50 has the higher AUROC (0.96 vs 0.95), ResNet-18 the higher balanced accuracy (0.90 vs
# 0.86, with fewer false alarms and more bubbles, undiluted and frozen milk caught).
# - **Best of each family**: the method with the highest balanced accuracy in each of the 6 families
#   (handcrafted one-class, handcrafted supervised, deep one-class, deep supervised, anomaly detection, CNN
#   with both designs: whole picture and 12 tiles).
# - **Votes of the best** (`BENCHMARK_VOTES`): the n family bests with the highest balanced accuracy each
#   decide on their own (unusualness > 95); an image is flagged if at least k of them flag it.
#   "At least 1 of 6": any of the 6 is enough. "At least 2 of 3": 2 of the 3 best must agree.
# - **Combined**: the best of the earlier combinations (max rule, votes of 5), also by balanced accuracy.
#
# The voters are chosen by their results on the test folds, so the votes are slightly optimistic.

# %%
# Best method of each family by balanced accuracy, and the votes of the best of them
for design, runs, names in [("whole", cnn_runs, CNN_BACKBONES), ("tiles", cnn_tile_runs, CNN_TILE_BACKBONES)]:
    for name in names:  # CNN image scores (P(defective)) of both designs, so the CNNs can vote too
        method_scores[f"CNN {name} | {design}"] = pd.DataFrame(
            {run: runs[(name, run)]["scores"]["p_defect"] for run in FOLDS})
families = [("handcrafted one-class", data["ONE_CLASS"]), ("handcrafted supervised", data["SUPERVISED"])] + [
    (family, table) for family, table in data["DEEP"].groupby("family")
] + [("anomaly detection", data["ANOMALY_ALL"]), ("CNN", data["CNN_ALL"])]
family_best = pd.concat([
    table.sort_values("balanced_accuracy", ascending=False).head(1).assign(family=family) for family, table in families
]).sort_values("balanced_accuracy", ascending=False)
ranked_best = family_best["method"].tolist()  # the 6 family bests, highest balanced accuracy first

unusual_best = {method: run_unusualness(method_scores[method], eval_info) for method in ranked_best}
benchmark_results = {}  # method name -> (metrics per run, predictions)
for n, k in BENCHMARK_VOTES:
    flags = sum((unusual_best[method] > FLAG_PERCENTILE).astype(float).where(unusual_best[method].notna())
                for method in ranked_best[:n])
    benchmark_results[f"vote: at least {k} of {n} best"] = evaluate_method(flags, eval_info, two_sided=False,
                                                                          cut=k - 0.5)
    print(f"At least {k} of {n} best:", ", ".join(ranked_best[:n]))

combined_best = data["COMBINED"].loc[data["COMBINED"]["combination"] != "alone"].sort_values(
    "balanced_accuracy", ascending=False).head(1).assign(family="combined")
votes_best = pd.DataFrame([{"family": "votes of the best", **summary_row(method, *result, eval_labels)}
                           for method, result in benchmark_results.items()])
benchmark_columns = ["family", "method", "auroc", "auroc_sd", "balanced_accuracy", "har", "false_alarms_pct",
                     "false_alarms_sd_pct"] + eval_labels[1:]
data["BENCHMARK"] = pd.concat([family_best, combined_best, votes_best])[benchmark_columns].sort_values(
    "balanced_accuracy", ascending=False)
save_table(data["BENCHMARK"].round(4), RESULTS_DIR / "benchmark_summary.csv", OVERWRITE)
data["BENCHMARK"].set_index("method").round(2)

# %% [markdown]
# # Final evaluation
#
# The numbers for the paper. Three changes against the sections above, all decided before these results
# were seen:
# 1. **Steadier cut (12 slides).** The cut of every method comes from the good pictures of all 12 slides that
#    are not tested in the run (the validation fold and both training folds), not only from the 4 validation
#    slides. The training pictures are scored by models that never saw them. The test fold is never used.
#    The CNNs keep the 4-slide cut (a 12-slide cut would need 8 more trainings per network).
# 2. **Choice on non-test data** (next step): every run chooses its methods using only the non-test folds.
# 3. **Statistics** (next step): confidence intervals by preparation and McNemar tests.
#
# The max-rule cut stays at the 95th percentile. PaDiM is not carried into the final evaluation: it models
# every patch position separately, which suits aligned objects, not freely chosen fields of view (its
# results above stay as the expected negative result).

# %% [markdown]
# ## Steadier cut: 12 slides
#
# Run 1 as an example: test fold 1, validation fold 2, training folds 3 and 4. The run's model learned from
# folds 3 and 4, so it rates their good pictures too normal. Fold 3 is therefore scored by a model trained on
# folds 2 + 4, and fold 4 by a model trained on folds 2 + 3 (the same amount of data, never the test fold).
# Four of the six fold pairs are the runs' own models; only the pairs 1 + 3 and 2 + 4 need new models: two more
# models for every scikit-learn method (handcrafted and deep features, one-class and supervised), about
# 15 min the first time (saved afterwards).
#
# Result: every method with the 4-slide and the 12-slide cut (false alarms, balanced accuracy, HAR, frozen
# milk), a figure of the change, and a check of the globule threshold.

# %%
# Extra fold-pair models of every scikit-learn method, then its out-of-fold scores
all_pairs = [(a, b) for a in sorted(FOLDS) for b in sorted(FOLDS) if a < b]
run_pairs = {tuple(sorted(train_folds(run))) for run in FOLDS}
extra_pairs = [pair for pair in all_pairs if pair not in run_pairs]
print(f"Fold pairs of the runs' own models: {sorted(run_pairs)}; extra models: {extra_pairs}")

final_scores, final_families = {}, {}  # method -> scores with out-of-fold training pictures; method -> family
for source, kind in [("handcrafted", "one-class"), ("handcrafted", "supervised")] + [
        (backbone, kind) for backbone in deep_features for kind in ("one-class", "supervised")]:
    models = ONE_CLASS_DETECTORS if kind == "one-class" else SUPERVISED_MODELS
    if source == "handcrafted":
        own = one_class_image_scores if kind == "one-class" else supervised_image_scores
        tables = lambda: (data["WHOLE_FEATURES"], data["TILE_FEATURES"])
    else:
        own = deep_image_scores[(source, kind)]
        tables = lambda: deep_feature_tables(deep_features[source])
    extra = load_or_compute(
        CACHE_DIR / f"extra_pairs_{source}_{kind}_{data['KEY']}.pkl",
        lambda: run_extra_pairs(models, *tables(), eval_info, kind, extra_pairs, desc=f"{source} {kind}"),
        USE_SAVED_RESULTS)
    for (name, input_name), image_scores in own.items():
        method = f"{name} | {input_name}" if source == "handcrafted" else f"{source} {name} | {input_name}"
        final_scores[method] = out_of_fold_scores(image_scores, extra[(name, input_name)], eval_info)
        final_families[method] = f"{'handcrafted' if source == 'handcrafted' else 'deep'} {kind}"
print(f"{len(final_scores)} scikit-learn methods with out-of-fold scores")

# %%
# Every scikit-learn method with the 4-slide cut (as in the sections above) and with the 12-slide cut
final_results = {method: evaluate_method(s, eval_info, two_sided=False, reference="non-test")
                 for method, s in final_scores.items()}  # method -> (metrics per run, predictions)
data["FINAL_SKLEARN"] = pd.DataFrame([{"family": final_families[method], **summary_row(method, *result, eval_labels)}
                                      for method, result in final_results.items()])
save_table(data["FINAL_SKLEARN"].round(4), RESULTS_DIR / "final_sklearn_summary.csv", OVERWRITE)
four = pd.DataFrame([summary_row(m, *evaluate_method(method_scores[m], eval_info, two_sided=False), eval_labels)
                     for m in final_scores]).set_index("method")
twelve = data["FINAL_SKLEARN"].set_index("method")
data["FINAL_CUT"] = pd.DataFrame({
    "family": twelve["family"],
    "false_alarms_4": four["false_alarms_pct"], "false_alarms_sd_4": four["false_alarms_sd_pct"],
    "false_alarms_12": twelve["false_alarms_pct"], "false_alarms_sd_12": twelve["false_alarms_sd_pct"],
    "balanced_accuracy_4": four["balanced_accuracy"], "balanced_accuracy_12": twelve["balanced_accuracy"],
    "har_4": four["har"], "har_12": twelve["har"], "frozen_4": four["frozen"], "frozen_12": twelve["frozen"],
})
save_table(data["FINAL_CUT"].reset_index().round(4), RESULTS_DIR / "final_cut_4_vs_12_slides.csv", OVERWRITE)
mean = data["FINAL_CUT"].mean(numeric_only=True)
print(f"Mean over {len(data['FINAL_CUT'])} methods, 4 -> 12 slides: false alarms {mean['false_alarms_4']:.1f} -> "
      f"{mean['false_alarms_12']:.1f} %, their spread between runs (SD) {mean['false_alarms_sd_4']:.1f} -> "
      f"{mean['false_alarms_sd_12']:.1f} points, balanced accuracy {mean['balanced_accuracy_4']:.3f} -> "
      f"{mean['balanced_accuracy_12']:.3f}")
data["FINAL_CUT"].sort_values("balanced_accuracy_12", ascending=False).head(20).round(3)

# %%
# Figure: every method's false alarms (mean over the 4 runs) against their spread between runs, 4 -> 12 slides
fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
cut_table = data["FINAL_CUT"]
for _, r in cut_table.iterrows():
    ax.plot([r["false_alarms_4"], r["false_alarms_12"]], [r["false_alarms_sd_4"], r["false_alarms_sd_12"]],
            color="#cfcfcf", lw=0.8, zorder=1)
ax.scatter(cut_table["false_alarms_4"], cut_table["false_alarms_sd_4"], s=18, color=REFERENCE_COLOUR,
           label="cut from 4 validation slides", zorder=2)
ax.scatter(cut_table["false_alarms_12"], cut_table["false_alarms_sd_12"], s=18, color=SERIES_COLOUR,
           label="cut from 12 non-test slides", zorder=3)
ax.set(xlabel="false alarms on good test pictures, mean of 4 runs (%)",
       ylabel="spread of the false alarms between runs (SD, points)",
       title=f"Steadier cut: {len(cut_table)} scikit-learn methods (line = the same method)")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
save_figure(fig, RESULTS_DIR / "final_cut_4_vs_12_slides.png", OVERWRITE)
plt.show()

# %% [markdown]
# **Globule threshold check.** The globule threshold $t$ of the handcrafted features was found once on all
# 609 good pictures, test folds included. Here it is found again in every run from the good pictures of the
# training folds only; if the values hardly differ, using one threshold for all runs does not matter.

# %%
# Globule threshold per run from the good pictures of the training folds only, against the one used.
# The deviation counts of the good pictures are loaded again under their own name (`counts` is reused later)
good_counts = load_or_compute(
    CACHE_DIR / "globule_deviation_counts.pkl",
    lambda: run_parallel(
        partial(deviation_counts, roi=ROI, tile=CROP_SIZE, whole_size=WHOLE_SIZE,
                sigma_tile=GLOBULE_BLUR_SIGMA, sigma_whole=GLOBULE_BLUR_SIGMA_WHOLE),
        good_paths, DATA_WORKERS, desc="D of good images",
    ),
    USE_SAVED_RESULTS,
)
path_fold = data["TILES"].drop_duplicates("path").set_index("path")["fold"]
threshold_rows = []
for run in FOLDS:
    keep = [i for i, p in enumerate(good_paths) if path_fold[str(p)] in train_folds(run)]
    threshold_rows.append({
        "good pictures": len(keep),
        "tiles": float(threshold_otsu(hist=(sum(good_counts[i][0] for i in keep), bin_centres))),
        "whole pictures": float(threshold_otsu(hist=(sum(good_counts[i][1] for i in keep), bin_centres))),
    })
threshold_check = pd.DataFrame(threshold_rows, index=[f"run {run}: training folds" for run in FOLDS])
threshold_check.loc["all good pictures (used)"] = [len(good_paths), data["GLOBULE_THRESHOLD"],
                                                   data["GLOBULE_THRESHOLD_WHOLE"]]
threshold_check.round(2)

# %% [markdown]
# ## Anomaly models (DFM, PatchCore): scores of the non-test pictures
#
# The sections above let DFM and PatchCore score only what the old evaluation needed: the test fold, the
# other milks and the good validation pictures. For the 12-slide cut and the choice on non-test data they
# also need:
# - **the defective validation pictures**, scored by each run's own model (DFM fitted again the same way,
#   PatchCore with its saved memory bank). Four good validation pictures per run are scored again as a
#   check: their scores must equal the saved ones;
# - **the training folds**, scored by models that never saw them. As for the scikit-learn methods, the runs'
#   own models cover four fold pairs (their test folds); two new models per method cover the pairs 1 + 3 and
#   2 + 4 (fitted on the good pictures of the pair, scoring all pictures of the other two folds).
#
# GPU, about 1 hour the first time (DFM ~20 min, PatchCore ~35 min; saved afterwards). PaDiM is not carried on
# (see above).

# %%
# Score the pictures still needed: defective validation pictures (runs' own models) and the extra fold pairs
final_anomaly_tiles = {}  # (name, run or pair) -> tile scores of the newly scored pictures
for name in ["DFM", "PatchCore"]:
    for target in list(FOLDS) + extra_pairs:
        label = f"run{target}" if isinstance(target, int) else f"folds{target[0]}{target[1]}"
        bank = patchcore_runs[target]["bank"] if (name == "PatchCore" and isinstance(target, int)) else None
        final_anomaly_tiles[(name, target)] = load_or_compute(
            CACHE_DIR / f"final_{name.lower()}_{label}_{data['KEY']}.pkl",
            lambda: final_anomaly_scores(name, eval_info, image_paths, target, bank), USE_SAVED_RESULTS)

# Check: the runs' own models give the saved scores again (good validation pictures scored twice)
saved_tiles = {("DFM", run): anomaly_tile_tables[("DFM", run)] for run in FOLDS}
saved_tiles.update({("PatchCore", run): patchcore_runs[run]["tile_scores"] for run in FOLDS})
for (name, run), saved in saved_tiles.items():
    new = final_anomaly_tiles[(name, run)]
    common = new.index.intersection(saved.index)
    largest = float((new.loc[common] - saved.loc[common]).abs().to_numpy().max())
    print(f"{name} run {run}: {len(common)} pictures scored again, largest difference {largest:.2e} "
          f"(scores up to {float(saved.loc[common].abs().to_numpy().max()):.3g})")

# %%
# DFM and PatchCore with the 12-slide cut (tiles max and tiles mean), next to the 4-slide cut
anomaly_rows = []
for name, saved_name in [("DFM", f"DFM {DFM_BACKBONE.split('.')[0]}"), ("PatchCore", f"PatchCore {patchcore_name}")]:
    own_tiles = {}
    for run in FOLDS:  # saved scores + the newly scored pictures (the check pictures keep their saved scores)
        both = pd.concat([saved_tiles[(name, run)], final_anomaly_tiles[(name, run)]])
        own_tiles[run] = both[~both.index.duplicated()]
    for agg in ("max", "mean"):
        method = f"{saved_name} | tiles {agg}"
        own = pd.DataFrame({run: t.agg(agg, axis=1) for run, t in own_tiles.items()})
        extra = {pair: final_anomaly_tiles[(name, pair)].agg(agg, axis=1) for pair in extra_pairs}
        final_scores[method] = out_of_fold_scores(own, extra, eval_info)
        final_families[method] = "anomaly detection"
        final_results[method] = evaluate_method(final_scores[method], eval_info, two_sided=False,
                                                reference="non-test")
        four = summary_row(method, *evaluate_method(method_scores[method], eval_info, two_sided=False), eval_labels)
        twelve = summary_row(method, *final_results[method], eval_labels)
        anomaly_rows.append({
            "method": method, "family": "anomaly detection",
            "false_alarms_4": four["false_alarms_pct"], "false_alarms_sd_4": four["false_alarms_sd_pct"],
            "false_alarms_12": twelve["false_alarms_pct"], "false_alarms_sd_12": twelve["false_alarms_sd_pct"],
            "balanced_accuracy_4": four["balanced_accuracy"], "balanced_accuracy_12": twelve["balanced_accuracy"],
            "har_4": four["har"], "har_12": twelve["har"], "frozen_4": four["frozen"], "frozen_12": twelve["frozen"],
        })
data["FINAL_ANOMALY"] = pd.DataFrame(anomaly_rows).set_index("method")
save_table(data["FINAL_ANOMALY"].reset_index().round(4), RESULTS_DIR / "final_anomaly_4_vs_12_slides.csv", OVERWRITE)
data["FINAL_ANOMALY"].round(3)

# %% [markdown]
# # Experiments

# %% [markdown]
# ## PyOD detectors (curiosity, not for the paper)
#
# Five more one-class detectors from the PyOD library, run exactly like the scikit-learn ones
# (all 95 features, good training images only, same evaluation), to see whether any of them beats
# the final one-class models:
# - **ECOD** and **COPOD**: for every feature, how far into the tail of the good distribution a value
#   lies, combined over all features; no settings at all;
# - **HBOS**: a histogram of every feature from the good images; rare bins = unusual;
# - **KNN**: distance to the 20th nearest good image;
# - **PCA**: how badly the image is described by the main patterns of the good images.

# %%
# Temporary experiment: the PyOD imports are kept here so the whole experiment can be removed in one piece
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.pca import PCA as PyodPCA

pyod_detectors = {
    "ECOD": lambda: ECOD(),
    "COPOD": lambda: COPOD(),
    "HBOS": lambda: HBOS(),
    "KNN": lambda: KNN(n_neighbors=20),
    # keep the main patterns (99 % of the variation): the histogram bins sum to 1, so the last
    # patterns have ~0 variance and PCA's score (divided by that variance) would explode
    "PCA": lambda: PyodPCA(n_components=0.99, svd_solver="full", random_state=SEED),
}
# PyOD on the 95 handcrafted features and on the deep features of every backbone in DEEP_BACKBONES
pyod_inputs = {"handcrafted": (data["WHOLE_FEATURES"], data["TILE_FEATURES"])}
pyod_inputs.update({backbone: deep_feature_tables(features) for backbone, features in deep_features.items()})
pyod_rows = []
for feature_set, (whole_table, tile_table) in pyod_inputs.items():
    # saved after the first run, per image set (data["KEY"])
    cache_file = CACHE_DIR / f"pyod_{feature_set}_{data['KEY']}.pkl"
    # progress bar: one step = one detector trained on one input in one run (5 x 2 x 4 = 40 steps)
    scores = load_or_compute(
        cache_file,
        lambda: run_one_class_detectors(pyod_detectors, whole_table, tile_table, eval_info, desc=f"PyOD {feature_set}"),
        USE_SAVED_RESULTS,
    )
    for (name, input_name), image_scores in scores.items():
        method = f"PyOD {feature_set} {name} | {input_name}"
        pyod_rows.append(summary_row(method, *evaluate_method(image_scores, eval_info, two_sided=False), eval_labels))
pyod_summary = pd.DataFrame(pyod_rows)

print("PyOD detectors next to the final one-class models (handcrafted and deep features), top 20 by AUROC:")
final_one_class = pd.concat([data["ONE_CLASS"], data["DEEP"][data["DEEP"]["family"] == "deep one-class"]])
pd.concat([final_one_class, pyod_summary]).sort_values("auroc", ascending=False)[
    ["method", "auroc", "auroc_sd", "balanced_accuracy", "har", "false_alarms_pct", "false_alarms_sd_pct"]
].head(20).round(3)

# %%
fig = plot_detection_heatmap(pyod_summary, eval_labels,
                             "Experiment: PyOD one-class detectors on handcrafted and deep features")
plt.show()
