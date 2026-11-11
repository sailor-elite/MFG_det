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
#     display_name: mfg-qc (3.11.x)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Config
#

# %%
import os
import pickle
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from skimage.feature import local_binary_pattern
from skimage.filters import threshold_otsu
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from tqdm import tqdm

# %%
RAW_DIR = Path("/home/qq/Pictures/MFG_det")  # raw micrographs, READ ONLY
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

# Label -> label group (CLAUDE.md §7.2); frozen and uht: other milk type (counted as defective)
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
}

# Expected number of raw images per label (CLAUDE.md §7.3)
EXPECTED_COUNTS = {
    "good": 640, "blur_light": 320, "blur_heavy": 320, "filter_yellow": 320, "filter_blue": 320,
    "bubbles": 320, "corner": 320, "overstained": 320, "undiluted": 320, "overdiluted": 320,
    "frozen": 240, "uht": 240,
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
}

# Fixed folds (CLAUDE.md §9 Step 6). F and U are not in any fold: they are test only.
FOLDS = {
    1: ["A01", "A02", "A03", "A13", "B01", "C01", "D01"],
    2: ["A04", "A05", "A06", "A16", "B02", "C02", "D02"],
    3: ["A07", "A08", "A09", "A14", "B03", "C03", "D03"],
    4: ["A10", "A11", "A12", "A15", "B04", "C04", "D04"],
}
# Evaluation runs: test on fold k, choose thresholds and settings on the next fold (cyclic),
# train on the other two. F and U (no fold) are tested in every run.
VALIDATION_FOLD = {k: k % len(FOLDS) + 1 for k in FOLDS}  # {1: 2, 2: 3, 3: 4, 4: 1}
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

data = {}

# %% [markdown]
# # Helper functions

# %%
def label_of(path: Path) -> str:
    """Return the label of one raw image, read from its file name (CLAUDE.md §7.2).

    The folder is ignored on purpose: some files are stored in the wrong folder
    (e.g. A11 and A14 are inside A08/).

    Args:
        path (Path): raw image file named "<PREP>_<NNNN>.png",
            e.g. Path("/home/qq/Pictures/MFG_det/A/A01/A01_0042.png").

    Returns:
        str: label of the image, e.g. "blur_light"
            (one of the values in LABELS_A or LABELS_BY_TYPE).

    Raises:
        ValueError: if the preparation type or image number is not in the mapping.
    """
    prep_id, image_idx = path.stem.split("_")  # "A01_0042" -> "A01", "0042"
    prep_type = prep_id[0]                       # "A01" -> "A"
    number = int(image_idx)                      # "0042" -> 42

    if prep_type == "A":
        for last, label in LABELS_A:
            if number <= last:
                return label
    elif prep_type in LABELS_BY_TYPE:
        return LABELS_BY_TYPE[prep_type]

    raise ValueError(f"No label defined for {path.name}")


def build_manifest(paths: list[Path], excluded_names: set[str]) -> pd.DataFrame:
    """Build the manifest: one row per raw image with its label, preparation details and fold.

    Uses label_of() and the Config globals LABEL_GROUPS, PREP_LOG and FOLDS.

    Args:
        paths (list[Path]): all raw image files, e.g. data["RAW_PICTURES"] (4000 paths).
        excluded_names (set[str]): file names removed for bad quality,
            e.g. {"A02_0009.png", "A08_0001.png", ...}.

    Returns:
        pd.DataFrame: one row per image, sorted by file name, with columns
            path (str): raw image path,
            file (str): file name, e.g. "A01_0042.png",
            prep_id (str): e.g. "A01"; prep_type (str): e.g. "A",
            image_idx (int): image number, e.g. 42,
            label (str): e.g. "blur_light"; label_group (str): e.g. "acquisition",
            is_defect (bool): True for every label except "good" (frozen and uht included),
            dilution_ml (float): distilled water added, e.g. 1.5,
            day (str): e.g. "2026-10-31"; milk_type (str): "fresh", "frozen" or "UHT",
            prep_time, photo_time (str): from the log, e.g. "2026-10-31 22:10",
            file_time (str): modification time of the raw file, e.g. "2026-10-31 22:35:12",
            fold (Int64): 1-4, <NA> for test-only preparations,
            test_only (bool): True for F and U (never used for training),
            excluded (bool): True if removed for bad quality.

    Raises:
        KeyError: if a preparation is missing from PREP_LOG.
    """
    fold_of = {prep: fold for fold, preps in FOLDS.items() for prep in preps}  # "A01" -> 1
    rows = []
    for path in paths:
        prep_id, image_idx = path.stem.split("_")  # "A01_0042" -> "A01", "0042"
        water_ml, milk_type, day, prep_time, photo_time = PREP_LOG[prep_id]
        label = label_of(path)
        rows.append({
            "path": str(path),
            "file": path.name,
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
            "fold": fold_of.get(prep_id),  # None for F and U
            "test_only": prep_id not in fold_of,
            "excluded": path.name in excluded_names,
        })
    manifest = pd.DataFrame(rows).sort_values("file", ignore_index=True)
    manifest["fold"] = manifest["fold"].astype("Int64")
    return manifest


def copy_pictures_by_label(
    pictures_by_label: dict[str, list[Path]],
    dst_dir: Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Copy raw images into one folder per label, keeping the original file names.

    Result: dst_dir/<label>/<original file name>, e.g. data/MFG/blur_light/A01_0042.png.
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
            n_in_folder (int): .png files now in the label folder,
            ok (bool): the folder contains exactly the expected file names.
    """
    rows = []
    for label, paths in pictures_by_label.items():
        label_dir = dst_dir / label
        folder_existed = label_dir.is_dir()
        label_dir.mkdir(parents=True, exist_ok=True)

        copied = skipped = 0
        for src in paths:
            dst = label_dir / src.name
            if dst.exists() and not overwrite:
                skipped += 1
                continue
            tmp = dst.with_name(dst.name + ".part")
            shutil.copyfile(src, tmp)
            src_stat = src.stat()
            os.utime(tmp, (src_stat.st_atime, src_stat.st_mtime))  # keep original timestamps
            tmp.replace(dst)
            copied += 1

        names_in_folder = {p.name for p in label_dir.glob("*.png")}
        rows.append({
            "label": label,
            "folder_existed": folder_existed,
            "copied": copied,
            "skipped": skipped,
            "n_expected": len(paths),
            "n_in_folder": len(names_in_folder),
            "ok": names_in_folder == {p.name for p in paths},
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


def evaluate_method(
    scores: pd.DataFrame, info: pd.DataFrame, two_sided: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one method in all runs with the common rules (percentile of good validation, 95 % cut).

    Args:
        scores (pd.DataFrame): index = file, one column per run (1-4): the method's score of every
            image in that run (for scores that need no training, the same column 4 times).
        info (pd.DataFrame): index = file (images that are not excluded), columns fold, label, is_defect.
        two_sided (bool): True for single features (a defect can move them either way).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            metrics: one row per run, on the test fold (A-D images): auroc, balanced_accuracy,
                har (harmonic mean of the good and defect recall), false_alarms_good,
                detected_defects; plus detected_frozen and detected_uht (test-only images);
            predictions: one row per image and run (test and test-only images): run, file, label,
                is_defect, role ("test" or "test_only"), score, unusualness, flagged, and
                ranking_score (what AUROC uses: the raw score, or for two-sided scores u with ties
                at 0/100 broken by the distance from the good median).
    """
    metric_rows, prediction_parts = [], []
    defect = info["is_defect"].astype(bool)
    for run in scores.columns:
        roles = run_roles(info["fold"], run)
        s = scores[run].reindex(info.index)
        good_val = s[(roles == "validation") & (info["label"] == "good")]
        u = pd.Series(unusualness(s, good_val, two_sided), index=info.index)
        flagged = u > FLAG_PERCENTILE
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
            "detected_frozen": flagged[test_only & (info["label"] == "frozen")].mean(),
            "detected_uht": flagged[test_only & (info["label"] == "uht")].mean(),
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


def one_class_scores(table: pd.DataFrame, info: pd.DataFrame, make_detector) -> pd.DataFrame:
    """Train a one-class detector on the good training rows of every run and score all rows.

    In every run: the features are standardised with the mean and SD of the good training rows,
    the detector learns from those rows only, and then scores every row.

    Args:
        table (pd.DataFrame): one row per whole picture or per tile, with a column "file" and the
            95 feature columns (a "tile_idx" column is ignored), e.g. data["WHOLE_FEATURES"].
        info (pd.DataFrame): index = file, columns fold and label (images that are not excluded).
        make_detector (callable): makes a new untrained detector, e.g. ONE_CLASS_DETECTORS["LOF"].

    Returns:
        pd.DataFrame: same index as table, one column per run (1-4) with the anomaly score.
    """
    feature_cols = [c for c in table.columns if c not in ("file", "tile_idx")]
    x_all = table[feature_cols].to_numpy(dtype=float)
    fold, label = table["file"].map(info["fold"]), table["file"].map(info["label"])
    scores = {}
    for run in FOLDS:
        train_good = ((run_roles(fold, run) == "train") & (label == "good")).to_numpy()
        scaler = StandardScaler().fit(x_all[train_good])
        x = scaler.transform(x_all)
        detector = make_detector()
        detector.fit(x[train_good])
        scores[run] = anomaly_scores(detector, x)
    return pd.DataFrame(scores, index=table.index)


def run_one_class_detectors(
    detectors: dict, whole: pd.DataFrame, tiles: pd.DataFrame, info: pd.DataFrame
) -> dict[tuple[str, str], pd.DataFrame]:
    """Run every detector on whole pictures and on tiles, and return image scores per run.

    Args:
        detectors (dict): name -> function that makes a new detector, e.g. ONE_CLASS_DETECTORS.
        whole (pd.DataFrame): whole-picture features, e.g. data["WHOLE_FEATURES"].
        tiles (pd.DataFrame): tile features, e.g. data["TILE_FEATURES"].
        info (pd.DataFrame): index = file, columns fold and label.

    Returns:
        dict[tuple[str, str], pd.DataFrame]: (detector name, input) -> image scores (index = file,
            columns = runs 1-4); input is "whole", "tiles max" (most unusual tile) or "tiles mean".
    """
    results = {}
    for name, make in tqdm(detectors.items(), desc="detectors"):
        results[(name, "whole")] = one_class_scores(whole, info, make).set_axis(whole["file"].to_numpy())
        tile_scores = one_class_scores(tiles, info, make)
        for agg in ("max", "mean"):
            results[(name, f"tiles {agg}")] = tile_scores.groupby(tiles["file"].to_numpy()).agg(agg)
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


# %% [markdown]
# # Loading data

# %%
data["RAW_PICTURES"] = sorted(p for p in RAW_DIR.rglob("*") if p.is_file())
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
unknown = excluded_names - {p.name for p in data["RAW_PICTURES"]}
if unknown:
    raise ValueError(f"Excluded files not found in the raw data: {sorted(unknown)}")

data["PICTURES_BY_LABEL"] = {
    label: [p for p in paths if p.name not in excluded_names]
    for label, paths in data["PICTURES_BY_LABEL"].items()
}
print(len(excluded_names), "images excluded")
{label: len(paths) for label, paths in data["PICTURES_BY_LABEL"].items()}

# %%
# Copy the raw images to MFG_DIR/<label>/ (about 38 GB; files that already exist are skipped)
copy_summary = load_or_compute(
    CACHE_DIR / "copy_summary.pkl",
    lambda: copy_pictures_by_label(data["PICTURES_BY_LABEL"], MFG_DIR, overwrite=False),
    USE_SAVED_RESULTS,
)
copy_summary

# %%
# First image of every label (full resolution, axes in pixels)
fig, axes = plt.subplots(3, 4, figsize=(20, 12), constrained_layout=True)
for ax, (label, paths) in zip(axes.flat, data["PICTURES_BY_LABEL"].items()):
    img = cv2.cvtColor(cv2.imread(str(paths[0])), cv2.COLOR_BGR2RGB)  # OpenCV reads BGR
    ax.imshow(img)
    ax.set_title(f"{label}  ({paths[0].name})")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")

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
# One row per raw image (all 4000; the excluded images are kept and flagged) with its label, the
# preparation details from the preparation log (CLAUDE.md §7.4) and the fold (§9 Step 6).
# F and U have no fold: they are test only. Saved to `data/manifest.csv`; later steps read it.

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
# The ROI cut into tiles of CROP_SIZE px, drawn on the first image of every label (12 labels)
x0, y0, x1, y1 = ROI
n_cols = (x1 - x0) // CROP_SIZE            # 4
n_tiles = n_cols * ((y1 - y0) // CROP_SIZE)  # 12

fig, axes = plt.subplots(3, 4, figsize=(24, 15), constrained_layout=True)
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
    ax.set_title(f"{label} ({path.name})", fontsize=14)
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
    "F and U are 'test_only' in every run": (image_roles[~in_folds] == "test_only").all(axis=None),
    "no preparation is in two roles within a run": all(
        images.assign(role=image_roles[run]).groupby("prep_id")["role"].nunique().max() == 1 for run in image_roles
    ),
}
pd.Series(checks, name="ok")

# %% [markdown]
# ## Plain variance of the Laplacian
#
# The evaluation tried on the simplest possible detector, before it is used for all methods:
# **"an image is defective if its variance of the Laplacian (VoL, sharpness of the whole picture) is
# unusual compared with good images"**. Nothing is trained. In every run:
# 1. the VoL of the good images of the validation fold shows what "normal" is;
# 2. a test image is flagged as defective if its VoL is below the 2.5th or above the 97.5th
#    percentile of them (two-sided: blur lowers VoL, undiluted raises it);
# 3. the decisions are compared with the true labels.
#
# Result: the metrics per run (table) and the figure (ROC curves, confusion matrix, flagged per label).

# %%
eval_info = data["MANIFEST"].loc[~data["MANIFEST"]["excluded"]].set_index("file")[["fold", "label", "is_defect"]]
eval_labels = list(data["PICTURES_BY_LABEL"])  # good, blur_light, ..., frozen, uht
demo_vol = pd.read_csv(WHOLE_FEATURES_CSV, usecols=["file", "vol"]).set_index("file")["vol"]
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
# (the ROI scaled to 768 x 576 px), for all 3,966 images that are not excluded. Computed once in
# `DATA_WORKERS` parallel processes (about 10-15 min) and saved to `data/features/`; later runs only
# read the files (unless `USE_SAVED_RESULTS = False`).

# %%
if TILE_FEATURES_CSV.exists() and WHOLE_FEATURES_CSV.exists() and USE_SAVED_RESULTS:
    data["TILE_FEATURES"] = pd.read_csv(TILE_FEATURES_CSV)
    data["WHOLE_FEATURES"] = pd.read_csv(WHOLE_FEATURES_CSV)
    print("Loaded the saved feature tables")
else:
    images = data["TILES"].drop_duplicates("file")[["file", "path"]]
    t0 = time.time()
    results = run_parallel(
        partial(image_features, roi=ROI, tile=CROP_SIZE, whole_size=WHOLE_SIZE,
                thresholds=(data["GLOBULE_THRESHOLD"], data["GLOBULE_THRESHOLD_WHOLE"]),
                sigmas=(GLOBULE_BLUR_SIGMA, GLOBULE_BLUR_SIGMA_WHOLE)),
        [Path(p) for p in images["path"]], DATA_WORKERS, desc="features",
    )
    print(f"Computed in {(time.time() - t0) / 60:.1f} min")
    data["TILE_FEATURES"] = pd.DataFrame([
        {"file": file, "tile_idx": k, **features}
        for file, (tile_list, _) in zip(images["file"], results)
        for k, features in enumerate(tile_list)
    ])
    data["WHOLE_FEATURES"] = pd.DataFrame([
        {"file": file, **whole} for file, (_, whole) in zip(images["file"], results)
    ])
    save_table(data["TILE_FEATURES"], TILE_FEATURES_CSV, overwrite=True)  # computed on purpose: replace
    save_table(data["WHOLE_FEATURES"], WHOLE_FEATURES_CSV, overwrite=True)
print(f"Tiles: {data['TILE_FEATURES'].shape}, whole pictures: {data['WHOLE_FEATURES'].shape}")

# %%
# Median of every single-value feature per label, over all tiles and over all whole pictures
label_of_file = data["TILES"].drop_duplicates("file").set_index("file")["label"]
label_order = list(data["PICTURES_BY_LABEL"])  # good, blur_light, ..., frozen, uht
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
    CACHE_DIR / "one_class_scores.pkl",
    lambda: run_one_class_detectors(ONE_CLASS_DETECTORS, data["WHOLE_FEATURES"], data["TILE_FEATURES"], eval_info),
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
# # Experiments
