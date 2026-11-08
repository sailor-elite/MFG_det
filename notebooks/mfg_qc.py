# ---
# jupyter:
#   jupytext:
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

import pandas as pd

# %%
RAW_DIR = Path("/home/qq/Pictures/MFG_det")  # raw micrographs, READ ONLY
MFG_DIR = Path("/home/qq/mfg_qc/data/MFG")    # copies of the raw images, one folder per label

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

# %%
# Copy the raw images to MFG_DIR/<label>/ (about 38 GB; files that already exist are skipped)
copy_summary = copy_pictures_by_label(data["PICTURES_BY_LABEL"], MFG_DIR, overwrite=False)
copy_summary

# %% [markdown]
# # Experiments
