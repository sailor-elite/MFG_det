# MFG-QC — Project context for Claude Code

Automated quality control of bright-field micrographs of milk fat globules (MFG):
classical focus/contrast measures vs. unsupervised anomaly detection vs. supervised CNNs.

This file is the single source of truth for the project. Read it fully before writing code.
Facts are marked as **[CONFIRMED]**, assumptions as **[ASSUMED]**, and open issues as **[TO VERIFY]**.
When something is marked [TO VERIFY], ask the user instead of guessing.

---

## 0. Safety rules (hard limits — read first)

These rules are also enforced by `.claude/settings.json` and by filesystem permissions where
possible, but follow them even where nothing technically stops you.

- Raw data in `/home/qq/Pictures/MFG_det` is **read-only**. Never write, move, rename or delete
  anything there — also not from Python scripts.
- **Never delete files anywhere** (no `rm`, `shutil.rmtree`, `os.remove`, `Path.unlink`, `find -delete`).
  If something should be removed, list it and ask the user.
- Do not edit `CLAUDE.md` or `.claude/settings.json` — the user maintains them manually.
- Write only inside `~/mfg_qc`. Before writing large caches (crops, features), report the expected
  size and the free disk space, and wait for the user's OK.
- Scripts that write outputs must not overwrite existing files unless an explicit `--overwrite`
  flag is given.
- No `sudo`, no system package installs, no changes outside the project. Python packages only via
  `uv add`, and only after the user agrees.
- Git: no `push`, `reset`, `rebase`, `clean`, `restore`, `checkout -- <file>`, force operations or
  branch deletion without the user's explicit OK.
- No uploading data anywhere and no downloads other than user-approved package installs.

## 1. Goal (TL;DR)

Build and compare methods that automatically flag defective microscope images of milk fat globules,
so that image quality can be checked for every image instead of a small manual sample.
Output: a journal article (pilot / feasibility study) plus reproducible code and a public dataset.

## 2. Motivation and source paper

- Source paper: "Molecular Remodeling of Milk Fat Globules Induced by Centrifugation: Insights from
  Deep Learning-Based Detection of Milk Adulteration", Int. J. Mol. Sci. 2025, 26(24), 11919
  (https://www.mdpi.com/1422-0067/26/24/11919).
- Their image QC was manual: 42 randomly selected micrographs were reviewed, 3 rejected as defocused
  or obscured, and a chi-square test (p = 0.52) was used to conclude that ~95% of ~25,000 images were
  of good quality. For 3/42 the 95% CI of the defect rate is roughly 2–19%, i.e. hundreds to
  ~4,800 bad images in 25k. Automated QC removes this uncertainty.
- Their acquisition protocol (for reference, ours differs where noted): Sudan III ethanol staining,
  bright field, 40× objective, pixel size 0.27 µm, several microscope/camera combos, 608×608 px crops,
  images converted to grayscale for training, metric HAR (harmonic mean of per-class recalls),
  samples warmed to 35 °C before imaging, simulated dust specks used as augmentation.
- Frame the paper as a constructive extension of their work, not a critique.

## 3. Target journal and constraints

- Applied Computer Science (ACS), Lublin University of Technology, https://ph.pollub.pl/index.php/acs
  - English, Word template, 8–20 pages, ≥15 references (≥60% journal articles).
  - Mandatory disclosure of generative-AI use (tool name). Similarity check (iThenticate):
    ≤30% overall, ≤10% from any single source.
  - Topic editor desk-check for scope; a single negative review leads to rejection.
  - Scope framing: computer application in technology / production engineering
    → "automated quality control in a laboratory image-analysis pipeline", not milk biology.
  - Quarterly issues: March, June, September, December.
- Closest methodological precedent in ACS: Karpiński & Syta (2026), ACS 22(3), "Anomaly detection in
  vibroarthrographic signals using handcrafted signal features and one-class methods":
  models trained only on normal data, thresholds from percentiles of normal validation scores,
  repeated participant-level splits, mean ± SD reporting. Mirror this rigor
  (our "participant" = one microscope preparation / slide).
- Timeline: first results for a meeting on **20 Nov 2026**; submission target **January 2027**
  (must appear by the June 2027 issue at the latest).

## 4. Research questions

- **RQ1** Which method family (classical / unsupervised anomaly detection / supervised CNN) best
  separates good from defective images, and best recognizes the defect type?
- **RQ2** How does detection depend on defect severity (light vs. heavy defocus)?
- **RQ3** Do detectors confuse a different milk type with an image defect? (false alarms on good
  images of frozen milk and UHT milk)
- **RQ4** How do defective images bias downstream analysis (e.g. fat-globule size distribution)?

Formulate results as answers ("which / how much"), not as hypotheses that must be confirmed.
Report all methods and all classes, including where methods fail.

## 5. Hardware and acquisition [CONFIRMED unless marked]

- Microscope: Levenhuk D320L PLUS 3.1M (monocular, achromatic objectives 4/10/40/100 oil).
  All images taken with the **40× objective**, bright field.
- Camera: 3.1 MP, 1/2" sensor, 3.2 µm pixels, **2048 × 1536 px**, software LevenhukLite (Linux),
  Quick Save with sequential numbering.
- Camera settings locked for the whole study: manual exposure, minimal gain, manual white balance
  set on an empty field, sharpening OFF, flat/dark-field corrections OFF, anti-flicker 50 Hz.
- Fine focus: 2 µm per division.
- Optics artefacts present in ALL images (normal, not defects): field curvature (edges softer than
  the centre), lateral chromatic aberration (purple/green fringes on globule edges), vignetting.
  → Use a central ROI (see §9 Step 1).
- Pixel size: ~0.08 µm/px **[ASSUMED]** (sensor pixel / 40, adapter assumed 1×; not calibrated with
  a stage micrometer). Do not rescale to 0.27 µm/px — the full frame would be only ~600×450 px.
- File format: **PNG** (lossless) [CONFIRMED]. Total size ~37.8 GB for ~4,000 files.

## 6. Sample preparation protocol

- One fresh whole-milk sample (single biological sample). Data collected on 31 Oct and 1 Nov 2026.
- Sudan III solution: 0.1 part Sudan III in 99.9 parts ethanol (≈0.1%, absolute ethanol) [CONFIRMED].
- Base recipe (class "good"): 1 ml milk + 3 drops Sudan III solution (~100 µl) → wait 15 min →
  dilute with distilled water **1.5, 2.5 or 3.5 ml** (all three levels count as "good"; the range
  represents acceptable lab variability) → one small drop on slide → coverslip →
  focus on the upper globule layer just under the coverslip.
- Staining time before dilution: 15 min [CONFIRMED]. Slide drop: one drop, ≈100 µl (estimated)
  [CONFIRMED].
- Observation: 3 drops of Sudan III give **no visible staining** of globules; a large dose coagulates
  milk proteins (haze) → used as a defect class ("over-stained").
- Over-stained recipe (B): 1 ml milk + 4.0 ml Sudan III solution → 1.5 ml distilled water [CONFIRMED].

## 7. Dataset

### 7.1 Locations (raw data — READ ONLY, never modify, move or delete)

```
/home/qq/Pictures/MFG_det/A   # base preparations A01–A16 (good + acquisition/field defects)
/home/qq/Pictures/MFG_det/B   # over-stained preparations B01–B04
/home/qq/Pictures/MFG_det/C   # undiluted preparations C01–C04
/home/qq/Pictures/MFG_det/D   # over-diluted preparations D01–D04
/home/qq/Pictures/MFG_det/F   # TEST ONLY: frozen milk, F01–F03, good images only
/home/qq/Pictures/MFG_det/U   # TEST ONLY: UHT milk 3.2%, U01–U03, good images only
```

**Warning: some images may be in the wrong folder** [CONFIRMED by user]. Do not trust folder
placement alone — see §9 Step 0.

Internal structure (sub-folders per preparation vs. flat files with prefix like `A01_0001.png`)
**[TO VERIFY]** — inspect before writing any loader. Image number = sequence number from
LevenhukLite Quick Save, starting at 1 for each preparation.

### 7.2 Image number → label mapping [CONFIRMED]

| Preparations | Image numbers | Label (folder-safe) | Group |
|---|---|---|---|
| A01–A16 | 1–40 | `good` (poprawne) | good |
| A01–A16 | 41–60 | `blur_light` (nieostre lekko) | acquisition |
| A01–A16 | 61–80 | `blur_heavy` (nieostre mocno) | acquisition |
| A01–A16 | 81–100 | `filter_yellow` (filtr żółty) | acquisition |
| A01–A16 | 101–120 | `filter_blue` (filtr niebieski) | acquisition |
| A01–A16 | 121–140 | `bubbles` (pęcherzyki powietrza) | preparation |
| A01–A16 | 141–160 | `corner` (róg/krawędź szkiełka) | field position |
| B01–B04 | 1–80 | `overstained` (przebarwione) | preparation |
| C01–C04 | 1–80 | `undiluted` (nierozcieńczone) | preparation |
| D01–D04 | 1–80 | `overdiluted` (zbyt rozcieńczone) | preparation |
| F01–F03 | 1–80 | `good` (frozen milk, test only) | good |
| U01–U03 | 1–80 | `good` (UHT milk, test only) | good |

Notes: blur levels are qualitative, set by the operator with the fine-focus knob
(light = globules soft but recognisable, heavy = globules merge into blobs); no exact focus offsets
were recorded. Bubbles and corners were searched for (not grid fields).

### 7.3 Expected counts

| Label | Images | Preparations |
|---|---|---|
| good (main) | 640 | 16 |
| blur_light / blur_heavy | 320 / 320 | 16 |
| filter_yellow / filter_blue | 320 / 320 | 16 |
| bubbles | 320 | 16 |
| corner | 320 | 16 |
| overstained | 320 | 4 |
| undiluted | 320 | 4 |
| overdiluted | 320 | 4 |
| **Main dataset total** | **3,520** | **28** |
| good – frozen (test) | 240 | 3 |
| good – UHT (test) | 240 | 3 |
| **Grand total** | **4,000** | **34** |

### 7.4 Preparation log (from `Dziennik_pomiary.xlsx`; times = preparation / start of imaging)

| ID | Type | Milk | Sudan III | Water | Images | Day | Prep | Photos |
|---|---|---|---|---|---|---|---|---|
| A01 | A | fresh | 3 drops | 1.5 ml | 160 | 31.10 | 22:10 | 22:32 |
| A02 | A | fresh | 3 drops | 2.5 ml | 160 | 01.11 | 08:48 | 09:08 |
| A03 | A | fresh | 3 drops | 3.5 ml | 160 | 31.10 | 19:42 | 19:47 |
| A04 | A | fresh | 3 drops | 1.5 ml | 160 | 31.10 | 21:10 | 21:34 |
| A05 | A | fresh | 3 drops | 2.5 ml | 160 | 01.11 | 09:09 | 09:16 |
| A06 | A | fresh | 3 drops | 3.5 ml | 160 | 31.10 | 19:49 | 20:06 |
| A07 | A | fresh | 3 drops | 1.5 ml | 160 | 31.10 | 21:35 | 21:50 |
| A08 | A | fresh | 3 drops | 2.5 ml | 160 | 01.11 | 09:17 | 09:24 |
| A09 | A | fresh | 3 drops | 3.5 ml | 160 | 31.10 | 20:07 | 20:20 |
| A10 | A | fresh | 3 drops | 1.5 ml | 160 | 31.10 | 21:50 | 22:03 |
| A11 | A | fresh | 3 drops | 2.5 ml | 160 | 01.11 | 09:40 | 09:46 |
| A12 | A | fresh | 3 drops | 3.5 ml | 160 | 31.10 | 20:22 | 20:31 |
| A13 | A | fresh | 3 drops | 1.5 ml | 160 | 01.11 | 08:08 | 08:22 |
| A14 | A | fresh | 3 drops | 2.5 ml | 160 | 01.11 | 09:48 | 10:07 |
| A15 | A | fresh | 3 drops | 3.5 ml | 160 | 31.10 | 20:34 | 20:45 |
| A16 | A | fresh | 3 drops | 1.5 ml | 160 | 01.11 | 08:23 | 08:34 |
| B01 | B | fresh | 4.0 ml | 1.5 ml | 80 | 01.11 | 12:49 | 12:52 |
| B02 | B | fresh | 4.0 ml | 1.5 ml | 80 | 01.11 | 12:53 | 12:58 |
| B03 | B | fresh | 4.0 ml | 1.5 ml | 80 | 01.11 | 12:58 | 13:01 |
| B04 | B | fresh | 4.0 ml | 1.5 ml | 80 | 01.11 | 13:01 | 13:04 |
| C01 | C | fresh | 3 drops | 0 ml | 80 | 01.11 | 10:29 | 10:37 |
| C02 | C | fresh | 3 drops | 0 ml | 80 | 01.11 | 10:39 | 10:46 |
| C03 | C | fresh | 3 drops | 0 ml | 80 | 01.11 | 10:50 | 10:56 |
| C04 | C | fresh | 3 drops | 0 ml | 80 | 01.11 | 10:57 | 11:05 |
| D01 | D | fresh | 3 drops | 20 ml | 80 | 01.11 | 11:50 | 11:59 |
| D02 | D | fresh | 3 drops | 20 ml | 80 | 01.11 | 12:00 | 12:14 |
| D03 | D | fresh | 3 drops | 20 ml | 80 | 01.11 | 12:30 | 12:35 |
| D04 | D | fresh | 3 drops | 20 ml | 80 | 01.11 | 12:36 | 12:37 |
| F01 | test | frozen (from main sample) | 3 drops | 3.5 ml | 80 | 01.11 | 13:08 | 13:13 |
| F02 | test | frozen (from main sample) | 3 drops | 3.5 ml | 80 | 01.11 | 13:15 | 13:17 |
| F03 | test | frozen (from main sample) | 3 drops | 3.5 ml | 80 | 01.11 | 13:19 | 13:22 |
| U01 | test | UHT 3.2% | 3 drops | 3.5 ml | 80 | 01.11? | 14:22 | 14:31 |
| U02 | test | UHT 3.2% | 3 drops | 3.5 ml | 80 | 01.11? | 14:32 | 14:36 |
| U03 | test | UHT 3.2% | 3 drops | 3.5 ml | 80 | 01.11? | 14:37 | 14:39 |

B recipe [CONFIRMED]: 1 ml milk + 4.0 ml Sudan III solution, then 1.5 ml water (strong overdose,
≈60% ethanol in the final mixture → protein coagulation). Pasteurised 3.2% milk (M01–M03) was planned but NOT
collected. Removed categories (not in the data): oil-contaminated optics, low contrast
(condenser aperture), unstained.

### 7.5 Known confounds and data caveats (must be handled and reported)

1. **Dilution ↔ day confound within "good"**: 2.5 ml preps were all made on day 2, 3.5 ml all on
   day 1; 1.5 ml on both days (A01, A04, A07, A10 day 1; A13, A16 day 2).
2. **Class ↔ day/time confound**: all B, C, D, F, U were made on day 2 (late morning / afternoon).
   Use A preps from day 2 (A02, A05, A08, A11, A13, A14, A16) as a time control: compare detector
   scores of good images day 1 vs. day 2.
3. **Independence**: every preparation (A–D, F, U) was made from a separate tube [CONFIRMED];
   the preparation ID is the grouping unit for all splits.
4. Single milk sample, single microscope → pilot study; state explicitly in Limitations.
5. Frozen milk: −18 °C for 12 h, thawed for 8 h at 30 °C [CONFIRMED]. Long warm thawing may add
   microbial/acidity changes on top of the freezing effect — mention in Limitations.
6. Some images may be misplaced between folders (see §9 Step 0).
7. Fields of view were chosen freely by the operator, not on a fixed grid [CONFIRMED] → possible
   selection bias; report in Limitations.

## 8. Tasks and labels for modelling

- **T1 (main)**: binary good vs. defective (all defect labels = defective).
- **T2**: multi-class defect type (10 fine labels in §7.2; also report the coarse groups
  acquisition / preparation / field position).
- **T3**: severity analysis for blur (light vs. heavy) — detection rate per level.
- Test-only sets F and U contain only good images → report false-alarm rate (1 − specificity).

## 9. Analysis pipeline (implement in this order)

**Step 0 — Inventory, verification and manifest.** Walk the raw folders, parse preparation ID and
image number from file names, assign labels from §7.2, read image size/mode, flag
missing/duplicate/extra files, compare counts with §7.3.
Folder placement is NOT reliable (some images may be misplaced), so also:
- cross-check file modification times against the preparation/photo times in §7.4;
- generate contact sheets (thumbnail grids, one per preparation, in image-number order) for the
  user to review visually;
- flag suspicious files, e.g. colour or sharpness outliers within a label block, or files whose
  timestamp falls outside their preparation's session;
- never move, rename or relabel raw files yourself — write proposed fixes to
  `data/corrections_proposed.csv` and wait for user approval; apply approved fixes only in the
  manifest.

Write `data/manifest.csv` with columns:
`path, prep_id, prep_type, image_idx, label, label_group, is_defect, dilution_ml, day, milk_type,
prep_time, photo_time, fold`. Print a summary table and stop for user review if counts differ.

**Step 1 — Useful ROI.** On ~20–30 good images, compute a tile-wise sharpness map (variance of
Laplacian per tile), average it, and choose the central region where sharpness ≥ 80% of the centre.
Save the figure (it goes into the paper). Use the same ROI for every image of every class.

**Step 2 — Crops.** Never downscale whole frames (fine blur cues disappear). Cut native-resolution
crops (e.g. 384–512 px) from the ROI. Cache crops on disk; keep raw data untouched. Keep RGB
(colour is needed for the filter classes); a grayscale variant is optional for comparison with the
source paper.

**Step 3 — Classical baselines (CPU).** Features: variance of Laplacian, Tenengrad, Brenner,
image entropy, mean/std intensity, RMS contrast, colour statistics (mean RGB/HSV, channel ratios),
globule area fraction (threshold segmentation). Two modes: (a) one-class thresholds = 95th
percentile of good validation scores; (b) supervised logistic regression / random forest.

**Step 4 — Unsupervised anomaly detection.** Anomalib PatchCore (backbone wide_resnet50_2, fall back
to resnet18 if memory is short; coreset ratio ~0.01) and PaDiM (resnet18). Train on good images
only; image-level score; threshold from good-only validation percentile; save anomaly maps.
Note: PaDiM models per-position statistics (designed for aligned objects); our images are
stationary textures — comment on this in the discussion. Check the installed Anomalib version and
its docs before writing code (API differs between major versions).

**Step 5 — Supervised CNNs.** timm ResNet-18/50 and Xception, ImageNet-pretrained, crop inputs,
class weights for imbalance, mixed precision (fall back to FP32 if NaNs), gradient accumulation to
an effective batch of 64. Binary (T1) and multi-class (T2) heads. Grad-CAM for explainability.

**Step 6 — Evaluation.** Grouped 4-fold CV by preparation. Fixed fold assignment (balanced by
dilution level and day):

| Fold | A preps (test) | B | C | D |
|---|---|---|---|---|
| 1 | A01, A02, A03, A13 | B01 | C01 | D01 |
| 2 | A04, A05, A06, A16 | B02 | C02 | D02 |
| 3 | A07, A08, A09, A14 | B03 | C03 | D03 |
| 4 | A10, A11, A12, A15 | B04 | C04 | D04 |

- Thresholds and hyperparameters are chosen only on preparations from the training part
  (inner grouped split), never on the test fold.
- F and U are never used for training; evaluate every fold model on them (mean ± SD).
- Metrics: AUROC, balanced accuracy, HAR, per-class recall, false-positive rate on good,
  confusion matrices; 95% CIs by cluster bootstrap over preparations; McNemar test for paired
  method comparisons; detection rate per blur level (RQ2); false-alarm rate on F/U (RQ3);
  inference time per image on CPU and GPU.
- RQ4: segment globules (e.g. Hough circles or threshold + watershed) on good vs. defective images
  of the same preparations; compare size distributions (Wasserstein distance, KS test).
- Sanity check first: mean-colour features must separate filter_yellow/filter_blue from good almost
  perfectly. If not, suspect a label/order bug, not the method.

**Step 7 — Outputs.** `results/` with per-fold predictions (CSV), metric tables, figures
(ROI map, example images per class, anomaly maps, Grad-CAM, confusion matrices, severity curve,
size distributions). Everything reproducible from config files with fixed seeds.

## 10. Compute and engineering conventions

- All processing runs **locally** on a Lenovo laptop (model reported as IHU156) with an NVIDIA
  GTX 1650 (4 GB VRAM), Linux. **No external or cloud compute** (no Colab, Kaggle, remote GPUs).
- Design everything for 4 GB VRAM: native-resolution crops, small batches + gradient accumulation,
  AMP with FP32 fallback, lighter backbones (resnet18) when memory is short, crop cache on disk,
  resumable runs with checkpoints (long trainings may run overnight).
- Check free disk space and RAM before caching crops; raw data alone is ~38 GB.
- Environment is managed with **uv** [CONFIRMED user preference] — never use `pip install`.
  Add packages with `uv add <pkg>` (dev tools: `uv add --dev <pkg>`), run everything with
  `uv run ...`. `pyproject.toml` and `uv.lock` are committed; `.venv/` is not.
  PyTorch/torchvision come from the PyTorch CUDA index configured in `pyproject.toml`
  (`[[tool.uv.index]]` + `[tool.uv.sources]`); after adding packages always re-check
  `torch.cuda.is_available()` (some packages can pull a CPU-only torch).
- Stack: Python 3.11, PyTorch, timm, anomalib, opencv-python, scikit-image, scikit-learn, pandas,
  numpy, matplotlib, grad-cam, statsmodels, pyyaml; dev: ipykernel, jupytext, nbstripout.
- Project layout (repo root `~/mfg_qc`):
  ```
  mfg_qc/
    CLAUDE.md
    pyproject.toml  # uv project definition (+ uv.lock)
    jupytext.toml   # formats = "ipynb,py:percent"
    .claude/        # settings.json with permission rules (maintained by the user)
    notebooks/      # mfg_qc.ipynb + mfg_qc.py: the ONE project notebook (jupytext pair, py:percent)
    configs/        # YAML configs per experiment
    data/           # manifest.csv, crop cache (git-ignored)
    results/        # tables/figures; checkpoints and caches git-ignored
  ```
- Development workflow [CONFIRMED user preference]:
  - **Everything is done in ONE notebook**: `notebooks/mfg_qc.ipynb`, paired with
    `notebooks/mfg_qc.py` in jupytext **py:percent** format (`jupytext.toml` at the repo root).
    Do not create other notebooks.
  - Notebook structure, top to bottom: `# Config` (imports first, then global variables)
    → `# Helper functions` (all function definitions; type hints + docstring with Args/Returns)
    → one `#` heading per pipeline step (Inventory, ROI, ...) → `# Experiments`
    (temporary code only, at the bottom).


  - The user writes and runs it **inside VS Code** (Microsoft Python + Jupyter extensions).
    Kernel = the project's uv environment `~/mfg_qc/.venv`
    (VS Code: Select Kernel → Python Environments → `.venv`). No JupyterLab server.
  - VS Code does not sync the pair automatically: after editing either file, run
    `uv run jupytext --sync notebooks/mfg_qc.ipynb`.
  - The notebook must show results inline: plots, pandas tables, example images, contact sheets.
    The user reviews everything visually in the notebook. Every figure/table meant for the paper is
    also saved to `results/`.
  - Editing the notebook: add new cells; do not rewrite or delete existing cells and never clear
    outputs. Never edit the notebook while the user has it open with unsaved changes — ask first.
  - Git: nbstripout strips outputs only from committed notebooks (outputs stay in local files).
    Never commit images, crops, checkpoints or the raw dataset. Remote: private GitHub repository
    (made public at submission).
  - Commit small, logical steps with clear English messages, only when the user asks.
  - Raw folders are read-only. Ask before deleting or moving anything.
  - ASCII-only file and folder names; English code comments; log every run (config + git hash).



## 11. Key references (verify every DOI before citing)

- Pertuz et al., 2013, focus measure operators, Pattern Recognition.
- Yang et al., 2018, microscope image focus quality with deep learning, BMC Bioinformatics.
- Bergmann et al., 2019, MVTec AD, CVPR.
- Defard et al., 2021, PaDiM, ICPR workshops.
- Roth et al., 2022, PatchCore, CVPR.
- Akcay et al., 2022, Anomalib, ICIP.
- Karpiński & Syta, 2026, ACS 22(3) (one-class anomaly detection precedent).
- Source paper: Int. J. Mol. Sci. 2025, 26(24), 11919.

## 12. Open questions for the user

1. Internal folder structure under `MFG_det/*` (inspect first; format is PNG).
