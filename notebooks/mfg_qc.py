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
RESULTS_DIR = Path("/home/qq/mfg_qc/results")  # figures and tables (also for the paper)
OVERWRITE = False  # False = never overwrite files that already exist in RESULTS_DIR
SEED = 0           # random seed for everything

# ROI (CLAUDE.md §9 Step 1) and tiles (Step 2)
SHARP_TILE = 64            # px, cell size of the sharpness map (2048 x 1536 frame -> 32 x 24 cells)
ROI_IMAGES_PER_PREP = 2    # random good images per A preparation for the sharpness map (16 x 2 = 32)
ROI_THRESHOLD = 0.8        # sharp area = at least 80 % of the sharpness in the frame centre
CROP_SIZE = 384            # px, side of one tile
ROI = (256, 192, 1792, 1344)            # x0, y0, x1, y1 in px: 4 x 3 tiles of 384 px, centred
ROI_FIRST_GUESS = (250, 200, 1750, 1400)  # x0, y0, x1, y1: first box chosen by eye (for comparison)

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
# B, C, D, F, U preparations: one label for all images (1-80); F and U are test-only good images
LABELS_BY_TYPE = {
    "B": "overstained",
    "C": "undiluted",
    "D": "overdiluted",
    "F": "good_frozen",
    "U": "good_uht",
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
# | blur_heavy, corner, overstained, undiluted, overdiluted, good_frozen, good_uht | all OK |
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
plt.show()

# %% [markdown]
# # ROI
#
# The edges of every frame are softer than the centre (field curvature, vignetting), so only a
# central region of interest (ROI) is used. Sharpness map: variance of the Laplacian in 64 x 64 px
# cells, averaged over random good images (2 per A preparation), divided by the sharpness in the
# centre (central 256 x 256 px). The sharp area is where this relative sharpness is at least 80 %.
# The ROI (4 x 3 tiles of 384 px, centred) should lie inside the sharp area.

# %%
# Pick ROI_IMAGES_PER_PREP random good images from every A preparation
rng = np.random.default_rng(SEED)
good_by_prep = {}
for p in data["PICTURES_BY_LABEL"]["good"]:
    good_by_prep.setdefault(p.name[:3], []).append(p)  # "A01_0001.png" -> "A01"
data["ROI_IMAGES"] = [
    good_by_prep[prep][i]
    for prep in sorted(good_by_prep)
    for i in sorted(rng.choice(len(good_by_prep[prep]), ROI_IMAGES_PER_PREP, replace=False))
]

# Average sharpness map, relative to the centre of the frame
mean_map = np.mean([sharpness_map(p, SHARP_TILE) for p in data["ROI_IMAGES"]], axis=0)
rows, cols = mean_map.shape  # (24, 32)
centre = mean_map[rows // 2 - 2:rows // 2 + 2, cols // 2 - 2:cols // 2 + 2].mean()
data["SHARPNESS_MAP"] = mean_map / centre

print(len(data["ROI_IMAGES"]), "images:", ", ".join(p.name for p in data["ROI_IMAGES"]))

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

# %% [markdown]
# # Experiments
