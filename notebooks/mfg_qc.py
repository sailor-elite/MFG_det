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
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

# %%
RAW_DIR = Path("/home/qq/Pictures/MFG_det")  # raw micrographs, READ ONLY
MFG_DIR = Path("/home/qq/mfg_qc/data/MFG")    # copies of the raw images, one folder per label
EXCLUDED_CSV = Path("/home/qq/mfg_qc/data/excluded_images.csv")  # images removed for bad quality
MANIFEST_CSV = Path("/home/qq/mfg_qc/data/manifest.csv")  # one row per raw image (labels, preps, folds)
RESULTS_DIR = Path("/home/qq/mfg_qc/results")  # figures and tables (also for the paper)
OVERWRITE = False  # False = never overwrite files that already exist in RESULTS_DIR
SEED = 0           # random seed for everything

# ROI (CLAUDE.md §9 Step 1) and tiles (Step 2)
SHARP_TILE = 64            # px, cell size of the sharpness map (2048 x 1536 frame -> 32 x 24 cells)
ROI_THRESHOLD = 0.8        # sharp area = at least 80 % of the sharpness in the frame centre
CROP_SIZE = 384            # px, side of one tile
ROI = (256, 192, 1792, 1344)            # x0, y0, x1, y1 in px: 4 x 3 tiles of 384 px, centred
ROI_FIRST_GUESS = (250, 200, 1750, 1400)  # x0, y0, x1, y1: first box chosen by eye (for comparison)
WHOLE_SIZE = (768, 576)    # (width, height) in px of the whole-picture input: the ROI scaled to half
EXAMPLE_IMAGE = "A01_0141.png"  # corner image used to show what the models get

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
copy_summary = copy_pictures_by_label(data["PICTURES_BY_LABEL"], MFG_DIR, overwrite=False)
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
mean_map = np.mean([sharpness_map(p, SHARP_TILE) for p in data["ROI_IMAGES"]], axis=0)
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
# # Experiments
