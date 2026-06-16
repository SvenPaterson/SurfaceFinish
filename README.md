# SurfaceFinish

![Overview](overview.png)

Surface roughness analysis toolkit with both CLI and Tkinter GUI front ends.

The current implementation targets:

- ISO 21920-3:2021 (profile acquisition / setting-class defaults)
- ISO 21920-2:2021 (parameter definitions)
- ISO 16610-21 (Gaussian S/L filtering)
- ISO 16610-31 (robust Gaussian regression L-filter for Rk-family source profile)

## Features

- TXT, CSV, Excel, and Digital Surf `.pro` input
- ISO setting classes Sc1-Sc5 via `iso21920.py`
- Automatic derivation of sampling section behavior:
	- `lsc = λc`
	- target `nsc = 5`
	- target `le = 5·λc`
- Automatic fallback when trace length is insufficient:
	- reduces `nsc` to the maximum that fits after filter edge buffers
	- emits warning text (`nsc_warning`) and flags this in plot banners/output
	- raises an error only when even one sampling section cannot fit
- Overview figure with:
	- raw + leveling fit
	- roughness + waviness (+ robust mean overlay)
	- bearing ratio curve with Rmr construction guides (`Cref`, `Rz/4` drop, target level, `Rmr`)
	- R-parameter table (`Ra`, `Rq`, `Rp`, `Rv`, `Rz`, `Rt`, `Rsk`, `Rku`, `Rmr`)
- Multi-standard comparison mode (GUI checkbox):
	- side-by-side table columns for ISO 21920, ISO 4287, and ASME B46.1
	- comparison uses each standard's intended roughness pipeline
	- ISO 4287 and ASME B46.1 are often numerically identical in this tool (Gaussian-based path), which is useful for customer-facing comparisons
- Batch processing (GUI):
	- recursive folder scan (depth ≤ 2) with automatic Part / Location grouping
	- adaptive plan-confirmation popup before processing
	- multi-sheet Excel workbook export (`Summary`, `Stats`, `Per-measurement`, `Settings`, plus `Failures` if any file errored)
	- supports `.xlsx`, `.csv`, `.pro` inputs

## Standards Comparison (GUI)

In the GUI, enable **Compare Standards** before clicking **Run Analysis** to render a side-by-side parameter table.

- ISO 21920 column: S-filter (`λs`) then L-filter (`λc`)
- ISO 4287 column: L-filter (`λc`) on primary profile (no S-filter)
- ASME B46.1 column: Gaussian path aligned to the ISO 4287 comparison path

Displayed parameters in comparison mode:

- `Ra`, `Rq`, `Rp`, `Rv`, `Rz`, `Rt`, `Rsk`, `Rku`, `Rmr`

Notes:

- `Rmr` is computed per column using the selected `Cref`.
- `Rt` is kept as the total-height metric in the table; `Rzx` is not shown.

## Batch Processing (GUI)

Click **Batch Process** in the GUI to analyze every supported file under a chosen folder using the same settings as a single-file run.

Folder layout:

- The picker scans up to **two levels deep** and groups files by parent folder.
- A folder one level under the root becomes a **Part**; a folder two levels deep becomes a **Location** within that part.
- Files sitting directly in the root are grouped under `(root)`.
- `~$*` Excel lock files, hidden folders, and any folder whose name starts with `TH_Template` are skipped.

Workflow:

1. Pick any one of your data files first so the GUI can infer the source format (`.pro` vs `.csv` vs `.xlsx`).
2. Tick **Save Excel report (Batch only)** if you want the workbook (default on).
3. Click **Batch Process** and select the root folder.
4. A confirmation popup shows the discovered Part / Location tree and total file count; click **Continue** to run.

Workbook output (sheets in order):

- **Summary** — one row per Part (and Location, if present), showing the mean of each parameter. Quick at-a-glance roughness comparison across all parts. Trailing **Units** row.
- **Stats** — one row per Part with `mean`, `stddev`, `min`, `max` columns for each parameter (e.g. `Ra mean`, `Ra stddev`, `Ra min`, `Ra max`, `Rq mean`, …).
- **Per-measurement** — one row per individual file, with parameters as columns.
- **Settings** — audit dump of the run settings (setting class, cutoffs, leveling order, units, etc.).
- **Failures** — present only if any file errored; lists path and error message.

## Quick Start

### GUI

```bash
python gui.py
```

GUI tip for inch data: use the **inch defaults** button (or Sc-class defaults with unit conversion) so `λs`/`λc` are in the same X-distance unit as your input trace.

### CLI

```bash
python main.py
```

If no `--file` is provided, CLI mode auto-prompts to choose an `.xlsx` in the project root.

## CLI Usage

Example using explicit file/sheet/columns/units:

```bash
python main.py --file "data/48743-004 trell sample t1.xlsx" --sheet "DATA" --x-col 4 --y-col 5 --x-unit in --y-unit "µin"
```

You can also pass column names instead of indexes:

```bash
python main.py --file "measurements.xlsx" --sheet "TraceData" --x-col "Distance" --y-col "Height" --x-unit mm --y-unit "µm"
```

### Important arguments

- `--setting-class {Sc1..Sc5,Custom}` (default `Sc3`)
- `--short-cutoff` and `--long-cutoff` (override setting-class defaults)
- `--order` (leveling order: 0, 1, 2, 3)
- `--plot-level` (show leveling preview)
- `--no-plot-roughness`
- `--no-plot-mr`

Note: evaluation length (`le`) and sampling section count (`nsc`) are backend-derived from `λc` and trace length. They are not user-configurable CLI options.

## Input Notes

- `--sheet` accepts a sheet name (for example `DATA`) or zero-based sheet index (`0`).
- `--x-col` and `--y-col` accept zero-based indexes or exact column names.
- `.txt`/`.csv` inputs use the first two columns unless pre-processed otherwise.
- `.pro` inputs are Digital Surf 1D profile files (`studiable_type == PROFILE`,
  magic `DIGITAL SURF` / `DSCOMPRESSED`). Parsing uses the `surfalize` package.
  X/Y units are taken from the file header (e.g. `in`/`in`) and override
  `--x-unit` / `--y-unit`; `--sheet`, `--x-col`, and `--y-col` are ignored.
  2D `.sur` surfaces and other studiable kinds are rejected with a clear error.

## Environment Setup

Use one of the following approaches.

### Option 1: venv + pip

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

### Option 2: conda

```bash
conda env create -f SurfFin.yml
conda activate SurfFin
```

`SurfFin.yml` is a Conda environment file and is not used with `pip install -r`.
